"""Trainable PyTorch implementation of Needle's Simple Attention Network.

The parameter names and matrix orientations are the canonical deployment names
from upstream ``needle/model/export.py``.  In particular every projection is
``[out_features, in_features]`` and the three mHC projections concatenate layers
along the output dimension.  The default arithmetic follows upstream's FP32
``decode.py``; activation fake quantization is an explicit, separate option.

This implementation is independent of the closed runtime.  It implements GQA,
split-half RoPE, zero-centred RMSNorm, gated attention, Hadamard MLPs, mHC with
20 Sinkhorn iterations, and the hashed/dilated Engram memory.  Optional exported
confidence and retrieval heads are supported as well.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class NeedleConfig:
    vocab_size: int = 8192
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 27
    head_dim: int = 64
    max_seq_len: int = 2048
    rope_theta: float = 100000.0
    mhc_lanes: int = 4
    engram_orders: tuple[int, ...] = (2, 3)
    engram_layers: tuple[int, ...] = (2, 15)
    engram_slots: int = 8192
    engram_sub_dim: int = 128
    num_engram_tables: int = 4
    engram_conv_taps: int = 4
    engram_conv_dilation: int = 3
    kv_window: int = 0
    kv_bits: int = 8
    act_bits: int = 8
    pad_token_id: int = 0

    def __post_init__(self):
        self.engram_orders = tuple(self.engram_orders)
        self.engram_layers = tuple(self.engram_layers)
        if min(self.d_model, self.vocab_size, self.num_heads, self.num_kv_heads,
               self.num_layers, self.mhc_lanes, self.head_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError("GQA requires H divisible by KV heads and an even RoPE head dimension")
        if any(i < 0 or i >= self.num_layers for i in self.engram_layers):
            raise ValueError("engram layer index outside the transformer stack")
        if len(set(self.engram_layers)) != len(self.engram_layers):
            raise ValueError("duplicate engram sites are unsupported")
        if self.engram_layers and (not self.engram_orders or min(self.engram_orders) < 1
                                   or self.num_engram_tables % len(self.engram_orders)):
            raise ValueError("invalid engram orders/table geometry")
        if self.engram_layers and min(self.engram_slots, self.engram_sub_dim,
                                      self.num_engram_tables, self.engram_conv_taps,
                                      self.engram_conv_dilation) <= 0:
            raise ValueError("engram dimensions must be positive")
        if self.kv_window < 0:
            raise ValueError("kv_window cannot be negative")

    @property
    def attn_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def engram_heads(self) -> int:
        return self.num_engram_tables // len(self.engram_orders)

    @classmethod
    def from_dict(cls, values: Mapping) -> "NeedleConfig":
        """Accept a CACT header, upstream training config, or HF config.json."""
        if isinstance(values, cls):
            return values
        v = dict(values)
        aliases = {"hidden_size": "d_model", "num_attention_heads": "num_heads",
                   "num_key_value_heads": "num_kv_heads", "num_hidden_layers": "num_layers",
                   "max_position_embeddings": "max_seq_len", "sliding_window": "kv_window"}
        for source, destination in aliases.items():
            if destination not in v and source in v:
                v[destination] = v[source]
        extras = v.get("extras", {})
        if "mhc_lanes" in extras:
            v.setdefault("mhc_lanes", extras["mhc_lanes"])
        for source, destination in {"sites": "engram_layers", "orders": "engram_orders",
                                    "slots": "engram_slots", "sub_dim": "engram_sub_dim",
                                    "conv_taps": "engram_conv_taps",
                                    "conv_dilation": "engram_conv_dilation"}.items():
            if source in extras.get("engram", {}):
                v.setdefault(destination, extras["engram"][source])
        quant = v.get("quantization", {})
        v.setdefault("kv_bits", quant.get("kv_cache_bits", 8))
        v.setdefault("act_bits", quant.get("activation_bits", 8))
        d, h = int(v.get("d_model", 512)), int(v.get("num_heads", 8))
        v.setdefault("head_dim", (int(v.get("attn_dim", 0)) or d) // h)
        orders = tuple(v.get("engram_orders", (2, 3)))
        heads = int(v.get("engram_heads", 0)) or max(1, d // max(1, len(orders) * 128))
        v.setdefault("num_engram_tables", len(orders) * heads)
        v.setdefault("engram_sub_dim", d // max(1, v["num_engram_tables"]))
        v.setdefault("engram_conv_dilation", max(orders) if orders else 1)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: value for k, value in v.items() if k in valid})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NeedleCache:
    """Incremental state; ``position`` counts all previously consumed tokens.

    With a sliding window only the recent KV entries are retained.  Token
    history is separately bounded by Engram's receptive field.  The cache can
    retain autograd graphs; use ``detach()`` between truncated training chunks.
    """
    keys: list[Tensor]
    values: list[Tensor]
    key_valid: Tensor
    token_history: Tensor
    history_valid: Tensor
    position: int
    key_positions: Tensor | None = None
    key_sink: Tensor | None = None
    history_sink: Tensor | None = None

    def detach(self) -> "NeedleCache":
        return NeedleCache([x.detach() for x in self.keys], [x.detach() for x in self.values],
                           self.key_valid.detach(), self.token_history.detach(),
                           self.history_valid.detach(), self.position,
                           self.key_positions.detach() if self.key_positions is not None else None,
                           self.key_sink.detach() if self.key_sink is not None else None,
                           self.history_sink.detach() if self.history_sink is not None else None)


def rms_unit(x: Tensor, epsilon: float = 1e-6) -> Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + epsilon)


def hadamard(x: Tensor) -> Tensor:
    """Normalized Sylvester/Walsh transform in O(n log n), with autograd."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    shape = x.shape
    y = x
    stride = 1
    while stride < n:
        z = y.reshape(*shape[:-1], -1, 2, stride)
        a, b = z.unbind(-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        stride *= 2
    return y * (n ** -0.5)


def _shift_right(x: Tensor, offset: int) -> Tensor:
    if offset == 0:
        return x
    if offset >= x.shape[1]:
        return torch.zeros_like(x)
    return torch.cat((torch.zeros_like(x[:, :offset]), x[:, :-offset]), dim=1)


def engram_indices(tokens: Tensor, orders: tuple[int, ...], heads: int, slots: int) -> Tensor:
    """Upstream FNV-style uint32 hash; int64 + explicit wrap works on CPU/CUDA."""
    u = tokens.to(torch.int64)
    indices = []
    for oi, order in enumerate(orders):
        for head in range(heads):
            seed = (0x9E3779B9 * (oi * heads + head + 1)) & 0xFFFFFFFF
            acc = torch.full_like(u, seed)
            for offset in range(order):
                acc = ((acc ^ _shift_right(u, offset)) * 0x01000193) & 0xFFFFFFFF
            acc = acc ^ (acc >> 15)
            indices.append(acc.remainder(slots))
    return torch.stack(indices, dim=-1)


def _sinkhorn(logits: Tensor, iterations: int = 20) -> Tensor:
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


def _required_shapes(c: NeedleConfig) -> dict[str, tuple[int, ...]]:
    d, a, kv, n, layers = c.d_model, c.attn_dim, c.num_kv_heads * c.head_dim, c.mhc_lanes, c.num_layers
    hada_n = 1 << (d - 1).bit_length()
    shapes = {"embedding": (c.vocab_size, d)}
    for i in range(layers):
        for name, shape in {"norm_in": (d,), "q_proj": (a, d), "k_proj": (kv, d),
                            "v_proj": (kv, d), "q_norm": (c.head_dim,), "k_norm": (c.head_dim,),
                            "gate_proj": (a, d), "out_proj": (d, a), "post_norm": (d,),
                            "attn_gate": (1,), "pre_hada": (d,), "d1": (hada_n,),
                            "d2": (hada_n,), "d3": (hada_n,)}.items():
            shapes[f"layer{i:02d}.{name}"] = shape
    for kind in ("pre", "post", "res"):
        width = n * n if kind == "res" else n
        shapes[f"mhc_phi_{kind}"] = (layers * width, n * d)
        shapes[f"mhc_a_{kind}"] = (layers,)
        shapes[f"mhc_b_{kind}"] = (layers, n, n) if kind == "res" else (layers, n)
    for i in range(len(c.engram_layers)):
        shapes[f"engram{i}.tables"] = (c.num_engram_tables * c.engram_slots, c.engram_sub_dim)
        for name in ("key_proj", "value_proj"):
            shapes[f"engram{i}.{name}"] = (d, c.num_engram_tables * c.engram_sub_dim)
        shapes[f"engram{i}.taps"] = (c.engram_conv_taps, d)
    shapes["final_norm"] = (d,)
    return shapes


class NeedleModel(nn.Module):
    """Needle SAN with canonical, directly serializable trainable parameters.

    ``forward(ids)`` returns ``[batch, time, vocabulary]`` logits.
    ``forward(ids, use_cache=True)`` returns ``(logits, NeedleCache)``.
    ``attention_mask`` is a boolean ``[batch, time]`` key-validity mask for
    the current chunk, or ``[batch, total_time]`` including cached tokens.
    ``sink_mask`` uses the same shapes and pins prefix KV entries outside the
    sliding window. Set sink entries before they would otherwise be evicted.
    """

    def __init__(self, config: NeedleConfig | Mapping,
                 tensors: Mapping[str, Tensor] | None = None, *,
                 quant_activations: bool = False, quant_kv: bool = False):
        super().__init__()
        self.config = NeedleConfig.from_dict(config)
        self.quant_activations = quant_activations
        self.quant_kv = quant_kv
        self._linear_backend = None
        self._kv_quantizer = None
        required = _required_shapes(self.config)
        if tensors is None:
            tensors = self._initial_weights(required)
        missing = set(required) - set(tensors)
        if missing:
            raise ValueError(f"missing canonical tensors: {sorted(missing)}")
        for name, shape in required.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensors[name].shape)}")
        for name, value in tensors.items():
            if isinstance(value, (bytes, bytearray)):
                continue
            value = torch.as_tensor(value).detach().clone()
            if not value.is_floating_point():
                value = value.float()
            module = self
            parts = name.split(".")
            for part in parts[:-1]:
                if part not in module._modules:
                    module.add_module(part, nn.Module())
                module = module._modules[part]
            if name == "heads.manifest":
                module.register_buffer(parts[-1], value)
            else:
                module.register_parameter(parts[-1], nn.Parameter(value))

    def _initial_weights(self, shapes: Mapping[str, tuple]) -> dict[str, Tensor]:
        result = {}
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, shape in shapes.items():
            if name.endswith((".d1", ".d2")):
                value = torch.ones(shape)
            elif name.endswith(".d3"):
                value = torch.full(shape, 0.02)
            elif name.startswith("mhc_a_"):
                value = torch.full(shape, 0.01)
            elif name == "mhc_b_res":
                value = 4 * torch.eye(self.config.mhc_lanes).expand(shape).clone()
            elif name.endswith(".taps"):
                value = torch.zeros(shape)
                value[0] = 1
            elif len(shape) == 1 or name.startswith("mhc_b_"):
                value = torch.zeros(shape)
            else:
                std = residual_std if name.endswith(("out_proj", "value_proj")) else 0.02
                value = torch.randn(shape) * std
            result[name] = value
        return result

    @classmethod
    def from_archive(cls, archive, **kwargs) -> "NeedleModel":
        tensors = {name: torch.from_numpy(record.dequantize().copy())
                   for name, record in archive.tensors.items() if name != "tokenizer"}
        return cls(archive.metadata, tensors, **kwargs)

    def canonical_state_dict(self, *, keep_vars: bool = False) -> dict[str, Tensor]:
        return dict(self.state_dict(keep_vars=keep_vars))

    def set_linear_backend(self, backend: Callable[[str, Tensor, Tensor], Tensor] | None):
        """Install an inference-only packed backend: ``backend(name, x, weight)``.

        It should fall back to ``F.linear`` for unsupported names/devices and
        whenever autograd is needed.  Layer slices of mHC are named
        ``mhc_phi_pre.layer00`` etc. and can use the supplied sliced weight.
        """
        self._linear_backend = backend
        return self

    def set_kv_quantizer(self, quantizer: Callable[[Tensor, int, int], Tensor] | None):
        """Install CQ fake quantization ``quantizer(x, bits, group_size=64)``."""
        self._kv_quantizer = quantizer
        return self

    def _w(self, name: str) -> Tensor:
        module_path, _, parameter_name = name.rpartition(".")
        return getattr(self.get_submodule(module_path), parameter_name)

    def _linear(self, name: str, x: Tensor, weight: Tensor | None = None) -> Tensor:
        weight = self._w(name) if weight is None else weight
        if self._linear_backend is not None:
            return self._linear_backend(name, x, weight)
        return F.linear(x, weight)

    def _aq(self, x: Tensor) -> Tensor:
        if not self.quant_activations:
            return x
        qmax = 2 ** (self.config.act_bits - 1) - 1
        absmax = x.float().abs().amax(dim=-1, keepdim=True)
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        quantized = (x.float() / scale).round().clamp(-qmax - 1, qmax) * scale
        return x + (quantized.to(x.dtype) - x).detach()

    def _kvq(self, x: Tensor) -> Tensor:
        # Upstream configure_deploy maps KV8 to KV_BITS=0 (no CQ fake quant).
        if not self.quant_kv or self.config.kv_bits >= 8:
            return x
        if self._kv_quantizer is None:
            from .quantize import fake_quantize
            return fake_quantize(x, self.config.kv_bits, 64)
        return self._kv_quantizer(x, self.config.kv_bits, 64)

    def _norm(self, x: Tensor, name: str) -> Tensor:
        return (rms_unit(x) * (1 + self._w(name).float())).to(x.dtype)

    def _rope(self, x: Tensor, start: int) -> Tensor:
        dim = x.shape[-1]
        exponent = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
        frequencies = self.config.rope_theta ** -exponent
        positions = torch.arange(start, start + x.shape[2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)

    def _attention(self, x: Tensor, layer: int, start: int, valid: Tensor,
                   key_positions: Tensor, key_sink: Tensor,
                   cache: NeedleCache | None) -> tuple[Tensor, Tensor, Tensor]:
        c = self.config
        b, t, _ = x.shape
        prefix = f"layer{layer:02d}"
        x = self._aq(x)
        q = self._linear(prefix + ".q_proj", x).view(b, t, c.num_heads, c.head_dim).transpose(1, 2)
        k = self._linear(prefix + ".k_proj", x).view(b, t, c.num_kv_heads, c.head_dim).transpose(1, 2)
        v = self._linear(prefix + ".v_proj", x).view(b, t, c.num_kv_heads, c.head_dim).transpose(1, 2)
        q = self._rope(self._norm(q, prefix + ".q_norm"), start)
        k = self._kvq(self._rope(self._norm(k, prefix + ".k_norm"), start))
        v = self._kvq(v)
        past = cache.keys[layer].shape[2] if cache is not None else 0
        if past:
            k = torch.cat((cache.keys[layer], k), dim=2)
            v = torch.cat((cache.values[layer], v), dim=2)
        qpositions = torch.arange(start, start + t, device=x.device)
        kpositions = key_positions
        mask = kpositions[None, :] <= qpositions[:, None]
        if c.kv_window:
            recent = qpositions[:, None] - kpositions[None, :] < c.kv_window
            mask = mask[None] & (recent[None] | key_sink[:, None, :])
        else:
            mask = mask[None]
        mask = mask[:, None, None] & valid[:, None, None, None, :]
        repeats = c.num_heads // c.num_kv_heads
        qg = q.reshape(b, c.num_kv_heads, repeats, t, c.head_dim)
        scores = torch.einsum("bgrtd,bgkd->bgrtk", qg.float(), k.float()) / math.sqrt(c.head_dim)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).to(v.dtype)
        out = torch.einsum("bgrtk,bgkd->bgrtd", weights, v)
        out = out.reshape(b, c.num_heads, t, c.head_dim).transpose(1, 2).reshape(b, t, c.attn_dim)
        out = out * torch.sigmoid(self._linear(prefix + ".gate_proj", x))
        return self._linear(prefix + ".out_proj", self._aq(out)), k, v

    def _engram_pairs(self, tokens: Tensor, current_valid: Tensor, current_sink: Tensor,
                      cache: NeedleCache | None) -> list[tuple[Tensor, Tensor]]:
        c = self.config
        if not c.engram_layers:
            return []
        history = cache.token_history if cache is not None else tokens[:, :0]
        old_valid = cache.history_valid if cache is not None else current_valid[:, :0]
        all_tokens = torch.cat((history, tokens), dim=1)
        valid = torch.cat((old_valid, current_valid), dim=1)
        old_sink = (cache.history_sink if cache is not None and cache.history_sink is not None
                    else torch.zeros_like(old_valid))
        sink = torch.cat((old_sink, current_sink), dim=1)
        indices = engram_indices(all_tokens, c.engram_orders, c.engram_heads, c.engram_slots)
        ngram_ok = torch.stack([_shift_right(valid, order - 1)
                                & ((not c.kv_window or order - 1 < c.kv_window)
                                   | _shift_right(sink, order - 1))
                                for order in c.engram_orders for _ in range(c.engram_heads)], -1)
        table_offsets = torch.arange(c.num_engram_tables, device=tokens.device) * c.engram_slots
        flat_indices = indices + table_offsets
        result = []
        for site in range(len(c.engram_layers)):
            prefix = f"engram{site}"
            fetched = F.embedding(flat_indices, self._w(prefix + ".tables")) * ngram_ok[..., None]
            e = self._aq(fetched.flatten(-2))
            k = self._linear(prefix + ".key_proj", e)
            raw_v = self._linear(prefix + ".value_proj", e)
            taps = self._w(prefix + ".taps")
            v = torch.zeros_like(raw_v)
            for tap in range(c.engram_conv_taps):
                offset = tap * c.engram_conv_dilation
                tap_ok = _shift_right(valid, offset)
                if c.kv_window and offset >= c.kv_window:
                    tap_ok = tap_ok & _shift_right(sink, offset)
                v = v + taps[tap] * _shift_right(raw_v, offset) * tap_ok[..., None]
            result.append((k[:, history.shape[1]:], v[:, history.shape[1]:]))
        return result

    def _block(self, x: Tensor, layer: int, start: int, valid: Tensor,
               key_positions: Tensor, key_sink: Tensor,
               cache: NeedleCache | None) -> tuple[Tensor, Tensor, Tensor]:
        prefix = f"layer{layer:02d}"
        attn, k, v = self._attention(self._norm(x, prefix + ".norm_in"), layer, start,
                                   valid, key_positions, key_sink, cache)
        x = x + torch.sigmoid(self._w(prefix + ".attn_gate")) * self._norm(attn, prefix + ".post_norm")
        z = self._norm(x, prefix + ".pre_hada")
        n = self._w(prefix + ".d1").numel()
        if n != z.shape[-1]:
            z = F.pad(z, (0, n - z.shape[-1]))
        z = hadamard(self._w(prefix + ".d1") * z)
        z = hadamard(F.silu(self._w(prefix + ".d2") * z))
        z = (self._w(prefix + ".d3") * z)[..., :self.config.d_model]
        return x + z, k, v

    def _mhc_projection(self, x: Tensor, layer: int, kind: str) -> Tensor:
        width = self.config.mhc_lanes ** (2 if kind == "res" else 1)
        name = "mhc_phi_" + kind
        weight = self._w(name)[layer * width:(layer + 1) * width].float()
        return self._linear(f"{name}.layer{layer:02d}", x.float(), weight)

    def _run(self, tokens: Tensor, cache: NeedleCache | None, attention_mask: Tensor | None,
             sink_mask: Tensor | None,
             collect_hidden: bool) -> tuple[Tensor, NeedleCache, Tensor | None]:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty_time]")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("tokens must be integer IDs")
        c = self.config
        b, t = tokens.shape
        start = cache.position if cache is not None else 0
        if start + t > c.max_seq_len:
            raise ValueError(f"sequence exceeds max_seq_len={c.max_seq_len}")
        if cache is not None and (len(cache.keys) != c.num_layers or cache.key_valid.shape[0] != b):
            raise ValueError("cache batch/layer geometry does not match the model")
        current_valid = torch.ones_like(tokens, dtype=torch.bool)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != b:
                raise ValueError("attention_mask must be [batch, current_time or total_time]")
            if attention_mask.shape[1] not in (t, start + t):
                raise ValueError("attention_mask width must cover the current chunk or full history")
            current_valid = attention_mask[:, -t:].to(device=tokens.device, dtype=torch.bool)
        old_valid = cache.key_valid if cache is not None else current_valid[:, :0]
        valid = torch.cat((old_valid, current_valid), dim=1)
        old_positions = (cache.key_positions if cache is not None and cache.key_positions is not None
                         else torch.arange(start - old_valid.shape[1], start, device=tokens.device))
        key_positions = torch.cat((old_positions, torch.arange(start, start + t, device=tokens.device)))
        old_sink = (cache.key_sink if cache is not None and cache.key_sink is not None
                    else torch.zeros_like(old_valid))
        current_sink = torch.zeros_like(current_valid)
        if sink_mask is not None:
            if sink_mask.ndim != 2 or sink_mask.shape[0] != b or sink_mask.shape[1] not in (t, start + t):
                raise ValueError("sink_mask must be [batch, current_time or total_time]")
            sink_mask = sink_mask.to(device=tokens.device, dtype=torch.bool)
            current_sink = sink_mask[:, -t:]
            if sink_mask.shape[1] == start + t:
                old_sink = sink_mask.index_select(1, old_positions)
        key_sink = torch.cat((old_sink, current_sink), dim=1)
        x0 = F.embedding(tokens, self._w("embedding")) * math.sqrt(c.d_model)
        x = x0[:, :, None].expand(b, t, c.mhc_lanes, c.d_model)
        hidden = [x0] if collect_hidden else None
        pairs = self._engram_pairs(tokens, current_valid, current_sink, cache)
        site_for_layer = {layer: site for site, layer in enumerate(c.engram_layers)}
        keys, values = [], []
        keep_cache = torch.ones_like(key_positions, dtype=torch.bool)
        if c.kv_window:
            keep_cache = (key_positions >= start + t - c.kv_window) | key_sink.any(dim=0)
        lane_ids = torch.arange(c.mhc_lanes, device=x.device)
        for layer in range(c.num_layers):
            nx = rms_unit(x.flatten(-2))
            lane = (lane_ids == layer % c.mhc_lanes).float()
            hpre = torch.sigmoid(self._w("mhc_a_pre")[layer].float() * self._mhc_projection(nx, layer, "pre")
                                 + self._w("mhc_b_pre")[layer].float() + 8 * lane - 4)
            u = torch.einsum("btn,btnc->btc", hpre, x.float()).to(x.dtype)
            bx = u
            if layer in site_for_layer:
                ek, ev = pairs[site_for_layer[layer]]
                alpha = torch.sigmoid((rms_unit(u) * rms_unit(ek)).sum(-1) / math.sqrt(c.d_model))
                bx = u + (alpha[..., None] * ev.float()).to(u.dtype)
            y, k, v = self._block(bx, layer, start, valid, key_positions, key_sink, cache)
            y = y - u
            hpost = 2 * torch.sigmoid(self._w("mhc_a_post")[layer].float() * self._mhc_projection(nx, layer, "post")
                                      + self._w("mhc_b_post")[layer].float() - 4 * (1 - lane))
            residual = self._mhc_projection(nx, layer, "res").reshape(b, t, c.mhc_lanes, c.mhc_lanes)
            hres = _sinkhorn(self._w("mhc_a_res")[layer].float() * residual
                             + self._w("mhc_b_res")[layer].float())
            x = (torch.einsum("btij,btjc->btic", hres, x.float())
                 + hpost[..., None] * y.float()[:, :, None]).to(x.dtype)
            if hidden is not None:
                hidden.append(x.mean(dim=2))
            # Preserve pinned prefix entries in addition to the recent window.
            keys.append(k[:, :, keep_cache])
            values.append(v[:, :, keep_cache])
        x = self._norm(x.mean(dim=2), "final_norm")
        history = torch.cat((cache.token_history, tokens), 1) if cache is not None else tokens
        hist_valid = torch.cat((cache.history_valid, current_valid), 1) if cache is not None else current_valid
        old_history_sink = (cache.history_sink if cache is not None and cache.history_sink is not None
                            else torch.zeros_like(cache.history_valid) if cache is not None else current_sink[:, :0])
        history_sink = torch.cat((old_history_sink, current_sink), 1)
        history_width = ((c.engram_conv_taps - 1) * c.engram_conv_dilation + max(c.engram_orders)
                         if c.engram_layers else 0)
        history = history[:, -history_width:] if history_width else history[:, :0]
        hist_valid = hist_valid[:, -history_width:] if history_width else hist_valid[:, :0]
        history_sink = history_sink[:, -history_width:] if history_width else history_sink[:, :0]
        next_cache = NeedleCache(keys, values, valid[:, keep_cache], history, hist_valid, start + t,
                                 key_positions[keep_cache], key_sink[:, keep_cache], history_sink)
        return x, next_cache, torch.stack(hidden, dim=2) if hidden is not None else None

    def forward(self, tokens: Tensor, cache: NeedleCache | None = None, *,
                use_cache: bool = False, attention_mask: Tensor | None = None,
                sink_mask: Tensor | None = None):
        hidden, next_cache, _ = self._run(tokens, cache, attention_mask, sink_mask, False)
        logits = self._linear("embedding", self._aq(hidden).float(), self._w("embedding").float())
        return (logits, next_cache) if use_cache else logits

    def hidden_cells(self, tokens: Tensor, attention_mask: Tensor | None = None,
                     sink_mask: Tensor | None = None) -> Tensor:
        if attention_mask is None:
            attention_mask = tokens != self.config.pad_token_id
        return self._run(tokens, None, attention_mask, sink_mask, True)[2]

    def _probe_pool(self, cells: Tensor, probes: Tensor, keep: Tensor) -> Tensor:
        b, t, layers, d = cells.shape
        cells = cells.reshape(b, t * layers, d)
        keep = keep.repeat_interleave(layers, dim=1)
        if not bool(keep.any(dim=1).all()):
            raise ValueError("probe pooling requires at least one non-padding token per sample")
        scores = torch.einsum("bcd,kd->bkc", cells.float(), probes.float()) / math.sqrt(d)
        scores = scores.masked_fill(~keep[:, None], -torch.inf)
        return torch.einsum("bkc,bcd->bkd", scores.softmax(-1), cells.float()).flatten(1).to(cells.dtype)

    def encode_contrastive(self, tokens: Tensor, *, train_backbone: bool = False) -> Tensor:
        cells = self.hidden_cells(tokens)
        if not train_backbone:
            cells = cells.detach()
        name = "contrastive_head"
        pooled = self._probe_pool(cells, self._w(name + ".probes"), tokens != self.config.pad_token_id)
        projected = self._linear(name + ".proj", pooled)
        return projected / (projected.float().square().sum(-1, keepdim=True) + 1e-12).sqrt().to(projected.dtype)

    def forward_confidence(self, tokens: Tensor, *, train_backbone: bool = False) -> Tensor:
        cells = self.hidden_cells(tokens)
        if not train_backbone:
            cells = cells.detach()
        name = "confidence_head"
        pooled = self._probe_pool(cells, self._w(name + ".probes"), tokens != self.config.pad_token_id)
        return (self._linear(name + ".proj", pooled) + self._w(name + ".bias"))[..., 0].float()

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int = 32, eos_token_id: int | None = None,
                 prefix_len: int = 0) -> Tensor:
        """Greedy generation returning the prompt followed by generated IDs."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if prefix_len < 0 or prefix_len > tokens.shape[1]:
            raise ValueError("prefix_len must be between zero and the prompt length")
        output = tokens
        cache = None
        current = tokens
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        for _ in range(min(max_new_tokens, self.config.max_seq_len - tokens.shape[1])):
            sink_mask = (torch.arange(current.shape[1], device=current.device)[None].expand_as(current) < prefix_len
                         if cache is None else None)
            logits, cache = self(current, cache, use_cache=True, sink_mask=sink_mask)
            nxt = logits[:, -1].argmax(dim=-1)
            if eos_token_id is not None:
                nxt = torch.where(finished, eos_token_id, nxt)
                finished |= nxt == eos_token_id
            current = nxt[:, None]
            output = torch.cat((output, current), dim=1)
            if bool(finished.all()):
                break
        return output


# Descriptive aliases for users porting the upstream JAX implementation.
SimpleAttentionNetwork = NeedleModel
TransformerConfig = NeedleConfig
