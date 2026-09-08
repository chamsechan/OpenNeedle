"""Cactus Quants, based on Cactus Compute's Apache-2.0 reference exporter.

Matrices use [out, in], quantized along the reduction dimension. This is
rotation + nonuniform codebook quantization, not uniform integer quantization.
"""
from functools import lru_cache
import math
import numpy as np

# Official Gaussian Lloyd-Max samples: RandomState(0), 400000, 200 iterations.
# Unit-sphere values for group=128 from the released archive's shared codebook.
_CB128 = np.array([
    -.1331721395254135, -.03990209102630615, .04003528133034706, .13346044719219208,
    -.18926136195659637, -.11788499355316162, -.06614968925714493, -.02101346105337143,
    .022101031616330147, .06678506731987, .11826207488775253, .18982602655887604,
    -.23953154683113098, -.18070605397224426, -.14095793664455414, -.10901255160570145,
    -.08151296526193619, -.056527651846408844, -.03293915092945099, -.010116109624505043,
    .012710800394415855, .03570791706442833, .059458255767822266, .08497702330350876,
    .112741619348526, .1446969211101532, .184239000082016, .2422737032175064,
], dtype=np.float32)

def codebook(bits, group_size=128):
    if bits not in (2, 3, 4):
        raise ValueError("CQ supports 2, 3, or 4 bits")
    _validate_group(group_size)
    start = {2: 0, 3: 4, 4: 12}[bits]
    return _CB128[start:start + (1 << bits)].copy() * np.float32(math.sqrt(128 / group_size))

def _validate_group(group):
    if group < 8 or group & (group - 1):
        raise ValueError("group_size must be a power of two >= 8")

@lru_cache(maxsize=8)
def hadamard_matrix(n):
    _validate_group(n)
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * np.float32(1 / math.sqrt(n))

def hadamard(x):
    """Normalized last-axis fast Walsh-Hadamard transform, O(n log n)."""
    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    _validate_group(n)
    step = 1
    while step < n:
        z = x.reshape(*x.shape[:-1], -1, 2, step)
        a, b = z[..., 0, :].copy(), z[..., 1, :].copy()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
        step *= 2
    return x * np.float32(1 / math.sqrt(n))

def pack_indices(indices, bits):
    idx = np.asarray(indices)
    if bits not in (2, 3, 4) or idx.ndim != 2 or idx.shape[1] % 8:
        raise ValueError("indices must be [rows, multiple of 8], bits in 2/3/4")
    if np.any(idx < 0) or np.any(idx >= 1 << bits):
        raise ValueError("index outside codebook")
    chunks = idx.astype(np.uint64).reshape(idx.shape[0], -1, 8)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for i in range(8):
        word |= chunks[..., i] << (i * bits)
    return np.stack([(word >> (8 * i)).astype(np.uint8) for i in range(bits)], -1).reshape(idx.shape[0], -1)

def unpack_indices(packed, bits, in_pad):
    if bits not in (2, 3, 4) or in_pad % 8:
        raise ValueError("invalid packed CQ geometry")
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != in_pad * bits // 8:
        raise ValueError("packed byte count does not match geometry")
    chunks = p.reshape(p.shape[0], -1, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for i in range(bits):
        word |= chunks[..., i] << (8 * i)
    return np.stack([((word >> (i * bits)) & ((1 << bits) - 1)).astype(np.uint8) for i in range(8)], -1).reshape(p.shape[0], in_pad)

def quantize_matrix(weight, bits=2, group_size=128, cb=None, chunk_rows=256):
    """Official export numerics (dense Hadamard), with bounded temporary memory."""
    _validate_group(group_size)
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or not all(w.shape) or not np.isfinite(w).all():
        raise ValueError("weight must be a finite nonempty matrix")
    if bits not in (2, 3, 4):
        raise ValueError("only CQ2/3/4 packing is supported")
    cb = codebook(bits, group_size) if cb is None else np.asarray(cb, dtype=np.float32)
    if cb.shape != (1 << bits,) or not np.isfinite(cb).all() or not np.all(np.diff(cb) > 0):
        raise ValueError("invalid sorted codebook")
    out, width = w.shape
    pad = (-width) % group_size
    in_pad = width + pad
    packed = np.empty((out, in_pad * bits // 8), np.uint8)
    norms = np.empty((out, in_pad // group_size), np.float16)
    h = hadamard_matrix(group_size)
    for row in range(0, out, chunk_rows):
        g = np.pad(w[row:row + chunk_rows], ((0, 0), (0, pad))).reshape(-1, in_pad // group_size, group_size)
        rot = g @ h
        norm = np.sqrt(np.sum(rot ** 2, axis=-1, keepdims=True))
        unit = rot / np.maximum(norm, 1e-12)
        pos = np.clip(np.searchsorted(cb, unit), 1, cb.size - 1)
        idx = np.where(np.abs(unit - cb[pos - 1]) <= np.abs(unit - cb[pos]), pos - 1, pos)
        packed[row:row + len(g)] = pack_indices(idx.reshape(len(g), in_pad), bits)
        with np.errstate(over="ignore"):
            scales = norm[..., 0].astype(np.float16)
        if not np.isfinite(scales).all():
            raise ValueError("group norm overflows FP16; rescale or retrain the weights")
        norms[row:row + len(g)] = scales
    return packed, norms

def dequantize_matrix(packed, norms, shape, bits=2, group_size=128, cb=None):
    cb = codebook(bits, group_size) if cb is None else np.asarray(cb, dtype=np.float32)
    out, width = shape
    in_pad = math.ceil(width / group_size) * group_size
    idx = unpack_indices(packed, bits, in_pad)
    rot = cb[idx].reshape(out, -1, group_size) * np.asarray(norms, dtype=np.float32)[..., None]
    return (rot @ hadamard_matrix(group_size)).reshape(out, in_pad)[:, :width].copy()

def fake_quantize(weight, bits=2, group_size=128, *, centroids=None):
    """Differentiable PyTorch CQ straight-through estimator for QAT.

    Quantization applies to the last axis and preserves the master parameters.
    Like upstream, norms are rounded to FP16 and indices use nearest/left ties.
    """
    import torch
    import torch.nn.functional as F
    _validate_group(group_size)
    w = weight.float()
    pad = (-w.shape[-1]) % group_size
    g = F.pad(w, (0, pad)).reshape(*w.shape[:-1], -1, group_size)
    h = torch.as_tensor(hadamard_matrix(group_size), device=w.device)
    cb = torch.as_tensor(codebook(bits, group_size) if centroids is None else centroids, device=w.device)
    rot = g @ h
    norm = rot.square().sum(-1, keepdim=True).sqrt()
    unit = rot / norm.clamp_min(1e-12)
    pos = torch.searchsorted(cb, unit.contiguous()).clamp(1, len(cb) - 1)
    idx = torch.where((unit - cb[pos - 1]).abs() <= (unit - cb[pos]).abs(), pos - 1, pos)
    dq = ((cb[idx] * norm.half().float()) @ h).reshape(*w.shape[:-1], w.shape[-1] + pad)[..., :w.shape[-1]].to(weight.dtype)
    return weight + (dq - weight).detach()
