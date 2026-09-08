"""Validated, named access to the official Needle 2 .cact format.

Format specification: cactus-compute/needle, needle/model/export.py (Apache-2.0).
All tensor payloads remain compressed until explicitly dequantized.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import numpy as np
from .quantize import dequantize_matrix, quantize_matrix

TAG = 0x05E12A83
HEADER = struct.Struct("<29If")
RECORD = struct.Struct("<BBHIIIIQQII")
FP16, FP32, CQ, RAW = 1, 2, 3, 4
LAYER_NAMES = ("norm_in", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate", "pre_hada", "d1", "d2", "d3")
MHC_NAMES = ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res", "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res")


def _validate_geometry(m):
    positive = ("d_model", "num_heads", "num_kv_heads", "num_layers", "head_dim",
                "mhc_lanes", "vocab_size", "max_seq_len", "hada_n")
    if any(m[k] <= 0 for k in positive) or m['num_heads'] % m['num_kv_heads']:
        raise ValueError("invalid model geometry")
    if m["head_dim"] % 2 or m["hada_n"] != 1 << (m["d_model"] - 1).bit_length():
        raise ValueError("invalid RoPE/Hadamard geometry")
    if not math.isfinite(m["rope_theta"]) or m["rope_theta"] <= 0:
        raise ValueError("rope_theta must be finite and positive")
    if m["kv_bits"] not in (2, 3, 4, 8):
        raise ValueError("unsupported KV bit width")
    sites, orders = m["engram_layers"], m["engram_orders"]
    if len(set(sites)) != len(sites) or any(i >= m["num_layers"] for i in sites):
        raise ValueError("invalid engram layer indices")
    if orders and min(orders) < 1:
        raise ValueError("engram orders must be positive")
    if sites and (not orders or m["num_engram_tables"] % len(orders)
                  or any(m[k] <= 0 for k in ("engram_slots", "engram_sub_dim", "num_engram_tables",
                                            "engram_conv_taps", "engram_conv_dilation"))):
        raise ValueError("invalid engram geometry")


def _canonical_shapes(m):
    """Pure-Python architecture geometry, checked before any numeric backend."""
    d, n, layers = m["d_model"], m["mhc_lanes"], m["num_layers"]
    a, kv = m["num_heads"] * m["head_dim"], m["num_kv_heads"] * m["head_dim"]
    h = m["hada_n"]
    shapes = {"embedding": (m["vocab_size"], d)}
    layer_shapes = ((d,), (a, d), (kv, d), (kv, d), (m["head_dim"],), (m["head_dim"],),
                    (a, d), (d, a), (d,), (1,), (d,), (h,), (h,), (h,))
    for layer in range(layers):
        shapes.update({f"layer{layer:02d}.{name}": shape for name, shape in zip(LAYER_NAMES, layer_shapes)})
    mhc_shapes = ((layers,), (layers,), (layers,), (layers, n), (layers, n), (layers, n, n),
                  (layers * n, n * d), (layers * n, n * d), (layers * n * n, n * d))
    shapes.update(zip(MHC_NAMES, mhc_shapes))
    width = m["num_engram_tables"] * m["engram_sub_dim"]
    for site in range(len(m["engram_layers"])):
        shapes.update({f"engram{site}.tables": (m["num_engram_tables"] * m["engram_slots"], m["engram_sub_dim"]),
                       f"engram{site}.key_proj": (d, width), f"engram{site}.value_proj": (d, width),
                       f"engram{site}.taps": (m["engram_conv_taps"], d)})
    shapes["final_norm"] = (d,)
    return shapes


def _validate_named_shapes(tensors, metadata):
    expected = _canonical_shapes(metadata)
    for head, probes in (("contrastive_head", 4), ("confidence_head", 8)):
        if head + ".probes" not in tensors:
            continue
        projection = tensors[head + ".proj"]
        if len(projection.shape) != 2:
            raise ValueError(f"{head}.proj: expected a matrix")
        output = projection.shape[0] if head == "contrastive_head" else 1
        expected.update({head + ".probes": (probes, metadata["d_model"]),
                         head + ".proj": (output, probes * metadata["d_model"]),
                         head + ".bias": (output,)})
    for name, shape in expected.items():
        record = tensors[name]
        if record.dtype == RAW or record.shape != shape:
            raise ValueError(f"{name}: expected numeric tensor {shape}, got dtype={record.dtype} shape={record.shape}")

@dataclass
class TensorRecord:
    name: str
    dtype: int
    shape: tuple
    blob: bytes
    group_size: int = 0
    bits: int = 0
    codebook: np.ndarray | None = None

    @property
    def in_pad(self):
        return math.ceil(self.shape[1] / self.group_size) * self.group_size

    @property
    def packed(self):
        physical_bits = 2 if self.bits == 5 else self.bits
        n = self.shape[0] * self.in_pad * physical_bits // 8
        return np.frombuffer(self.blob, dtype=np.uint8, count=n).reshape(self.shape[0], -1)

    @property
    def norms(self):
        offset = self.packed.size
        return np.frombuffer(self.blob, dtype="<f2", offset=offset).reshape(self.shape[0], -1)

    def dequantize(self):
        if self.dtype == RAW:
            raise TypeError("RAW tokenizer is not a numeric tensor")
        if self.dtype in (FP16, FP32):
            return np.frombuffer(self.blob, dtype="<f2" if self.dtype == FP16 else "<f4").reshape(self.shape).astype(np.float32)
        if self.bits == 5:
            from .quantize import unpack_indices, hadamard_matrix
            idx = unpack_indices(self.packed, 2, self.in_pad)
            if np.any(idx == 2):
                raise ValueError("invalid ternary crumb 2")
            trits = np.where(idx == 3, -1, idx).astype(np.float32)
            rot = trits.reshape(self.shape[0], -1, self.group_size) * np.float32(1.2240064 / math.sqrt(self.group_size)) * self.norms.astype(np.float32)[..., None]
            return (rot @ hadamard_matrix(self.group_size)).reshape(self.shape[0], -1)[:, :self.shape[1]].copy()
        return dequantize_matrix(self.packed, self.norms, self.shape, self.bits, self.group_size, self.codebook)

    def quantized_like(self, weight, bits=None):
        w = np.asarray(weight, dtype=np.float32)
        if w.shape != self.shape or not np.isfinite(w).all():
            raise ValueError(f"{self.name}: expected finite tensor {self.shape}, got {w.shape}")
        if self.dtype != CQ:
            dt = "<f2" if self.dtype == FP16 else "<f4"
            with np.errstate(over="ignore"):
                values = w.astype(dt)
            if not np.isfinite(values).all():
                raise ValueError(f"{self.name}: storage dtype overflow")
            return TensorRecord(self.name, self.dtype, self.shape, values.tobytes())
        bits = self.bits if bits is None else bits
        if bits != self.bits:
            raise ValueError("use Archive.rebuild with an explicit precision map")
        p, s = quantize_matrix(w, bits, self.group_size, self.codebook)
        return TensorRecord(self.name, CQ, self.shape, p.tobytes() + s.astype("<f2").tobytes(), self.group_size, bits, self.codebook)

class Archive:
    @classmethod
    def load(cls, path):
        return cls(Path(path).read_bytes())

    def __init__(self, raw):
        self.raw = bytes(raw)
        if len(raw) < HEADER.size:
            raise ValueError("truncated .cact header")
        self.header = h = HEADER.unpack_from(raw)
        if h[0] != TAG:
            raise ValueError(f"unsupported .cact tag {h[0]:#x}; expected Needle 2 {TAG:#x}")
        if h[2] != 28 or h[19] > 4 or h[24] > 4:
            raise ValueError("unsupported codebook or engram header geometry")
        end_dir = HEADER.size + h[2] * 4 + h[1] * RECORD.size
        if end_dir > len(raw):
            raise ValueError("truncated .cact directory")
        fields = ("vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers", "head_dim", "max_seq_len", "hada_n", "mhc_lanes", "engram_slots", "engram_sub_dim", "num_engram_tables", "engram_conv_taps", "engram_conv_dilation")
        self.metadata = dict(zip(fields, h[5:19]))
        self.metadata.update(kv_window=h[3], kv_bits=h[4], engram_orders=list(h[20:20+h[19]]), engram_layers=list(h[25:25+h[24]]), rope_theta=h[29], act_bits=8)
        m = self.metadata
        _validate_geometry(m)
        minimum_tensors = 1 + m["num_layers"] * len(LAYER_NAMES) + len(MHC_NAMES) + 4 * len(m["engram_layers"]) + 1
        if h[1] not in {minimum_tensors + extra for extra in (0, 1, 4, 5, 7, 8)}:
            raise ValueError("tensor count does not match model geometry")
        self.codebook = np.frombuffer(raw, dtype="<f4", count=h[2], offset=HEADER.size).copy()
        if not np.isfinite(self.codebook).all():
            raise ValueError("nonfinite codebook")
        records = []
        intervals = []
        for i in range(h[1]):
            r = RECORD.unpack_from(raw, HEADER.size + h[2] * 4 + i * RECORD.size)
            dt, ndim, reserved, *rest = r
            if ndim > 4 or reserved != 0 or dt not in (FP16, FP32, CQ, RAW):
                raise ValueError(f"invalid tensor record {i}")
            shape = tuple(r[3:3+ndim])
            offset, nbytes, group, bits = r[7:]
            if offset % 64 or offset < end_dir or offset + nbytes > len(raw):
                raise ValueError(f"out-of-range or unaligned tensor {i}")
            if dt != RAW and (not shape or not all(shape)):
                raise ValueError("empty numeric tensor")
            if dt != CQ and (group != 0 or bits != 0):
                raise ValueError("non-CQ tensor carries quantization metadata")
            if dt == RAW and ndim != 0:
                raise ValueError("RAW tokenizer must have an empty shape")
            if dt in (FP16, FP32):
                expected = math.prod(shape) * (2 if dt == FP16 else 4)
            elif dt == CQ:
                if ndim != 2 or group < 8 or group & (group - 1) or bits not in (2, 3, 4, 5):
                    raise ValueError("unsupported CQ geometry")
                padded = math.ceil(shape[1] / group) * group
                expected = shape[0] * (padded * (2 if bits == 5 else bits) // 8 + padded // group * 2)
            else:
                expected = nbytes
            if nbytes != expected:
                raise ValueError(f"tensor {i}: wrong payload size")
            cb = self.codebook[{2:0,3:4,4:12}[bits]:{2:4,3:12,4:28}[bits]] if dt == CQ and bits != 5 else None
            if cb is not None and not np.all(np.diff(cb) > 0):
                raise ValueError("codebook must be sorted")
            records.append(TensorRecord("", dt, shape, bytes(raw[offset:offset+nbytes]), group, bits, cb))
            intervals.append((offset, offset+nbytes))
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("overlapping tensor payloads")
        names = ["embedding"] + [f"layer{i:02d}.{n}" for i in range(m['num_layers']) for n in LAYER_NAMES] + list(MHC_NAMES)
        names += [f"engram{i}.{n}" for i in range(len(m['engram_layers'])) for n in ("tables", "key_proj", "value_proj", "taps")]
        names += ["final_norm"]
        nraw = int(bool(records) and records[-1].dtype == RAW)
        extra = len(records) - len(names) - nraw
        if extra:
            if extra < 4 or (extra-1) % 3 or len(names) >= len(records):
                raise ValueError("invalid optional probe heads layout")
            manifest_record = records[len(names)]
            if manifest_record.dtype not in (FP16, FP32) or len(manifest_record.shape) != 1:
                raise ValueError("head manifest must be a floating-point vector")
            manifest = manifest_record.dequantize().reshape(-1)
            if len(manifest) != (extra-1)//3 or len(set(manifest.tolist())) != len(manifest):
                raise ValueError("invalid head manifest")
            names.append("heads.manifest")
            for c in manifest:
                if c not in (1, 2):
                    raise ValueError("unknown probe head code")
                head = {1:'contrastive_head', 2:'confidence_head'}[int(c)]
                names.extend(head + '.' + n for n in ('probes','proj','bias'))
        if nraw:
            names.append("tokenizer")
        if len(names) != len(records):
            raise ValueError("tensor count does not match model geometry")
        self.tensors = {}
        for name, record in zip(names, records):
            record.name = name
            self.tensors[name] = record
        _validate_named_shapes(self.tensors, self.metadata)

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()

    def stats(self):
        numeric = [t for t in self.tensors.values() if t.dtype != RAW]
        quant = [t for t in numeric if t.dtype == CQ]
        params = sum(math.prod(t.shape) for t in numeric)
        qp = sum(math.prod(t.shape) for t in quant)
        return dict(bytes=len(self.raw), tensors=len(self.tensors), parameters=params,
                    quantized_parameters=qp,
                    code_bits_per_quantized_weight=sum(math.prod(t.shape)* (2 if t.bits == 5 else t.bits) for t in quant)/qp if qp else 0,
                    numeric_payload_bits_per_weight=sum(len(t.blob)*8 for t in numeric)/params,
                    file_bits_per_weight=len(self.raw)*8/params, sha256=self.sha256,
                    precision_counts={str(b):sum(t.bits==b for t in quant) for b in sorted({t.bits for t in quant})})

    def rebuild(self, replacements=None):
        replacements = replacements or {}
        if set(replacements) - self.tensors.keys():
            raise ValueError("unknown replacement tensors")
        for name, record in replacements.items():
            if record.name != name:
                raise ValueError(f"replacement name mismatch: key {name!r}, record {record.name!r}")
        records = [replacements.get(n,t) for n,t in self.tensors.items()]
        # Exact untouched round trip, including padding bytes.
        if all((t.dtype, t.shape, t.blob, t.group_size, t.bits)
               == (self.tensors[t.name].dtype, self.tensors[t.name].shape,
                   self.tensors[t.name].blob, self.tensors[t.name].group_size,
                   self.tensors[t.name].bits) for t in records):
            return self.raw
        buf = bytearray(HEADER.pack(*self.header) + self.codebook.astype('<f4').tobytes())
        directory = bytearray()
        pos = len(buf) + len(records)*RECORD.size
        payload = bytearray()
        for t in records:
            aligned = (pos + 63) & ~63
            payload.extend(b'\0' * (aligned-pos))
            directory.extend(RECORD.pack(t.dtype, len(t.shape), 0, *(list(t.shape)+[0]*4)[:4], aligned, len(t.blob), t.group_size, t.bits))
            payload.extend(t.blob)
            pos = aligned + len(t.blob)
        result = bytes(buf+directory+payload)
        Archive(result)  # Validate before writing an externally consumable archive.
        return result
