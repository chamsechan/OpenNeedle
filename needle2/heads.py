"""Streaming Needle retrieval/confidence heads over the independent CPU engine.

Probe attention pools token × layer cells, including scaled input embeddings.
Online log-sum-exp accumulation retains O(probes × hidden_size) state instead
of storing the sequence's hidden cells.  No dense transformer copy is created.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .archive import Archive, CQ
from .native import NativeCQ, NativeEngine


class _OnlineProbePool:
    def __init__(self, probes):
        self.probes = np.ascontiguousarray(probes, dtype=np.float32)
        self.maximum = np.full(len(probes), -np.inf, dtype=np.float32)
        self.denominator = np.zeros(len(probes), dtype=np.float32)
        self.numerator = np.zeros_like(self.probes)

    def update(self, cells):
        """Add any nonempty ``[cells, hidden_size]`` block."""
        cells = np.asarray(cells, dtype=np.float32)
        scores = (self.probes @ cells.T) * np.float32(1 / math.sqrt(cells.shape[-1]))
        maximum = np.maximum(self.maximum, scores.max(axis=-1))
        previous_scale = np.exp(self.maximum - maximum)
        weights = np.exp(scores - maximum[:, None])
        self.numerator = self.numerator * previous_scale[:, None] + weights @ cells
        self.denominator = self.denominator * previous_scale + weights.sum(axis=-1)
        self.maximum = maximum

    def result(self):
        if not np.all(self.denominator > 0):
            raise ValueError("probe pooling requires at least one cell")
        return (self.numerator / self.denominator[:, None]).reshape(-1)


class NativeProbeEncoder:
    """Evaluate exported heads without retaining full-precision SAN weights.

    Each call resets the independent engine and consumes one unpadded token
    sequence. ``both(ids)`` shares a single hidden-state pass for retrieval and
    confidence. ``prefix_len`` pins prefix attention keys, as in NativeEngine.
    The confidence method returns the raw logit used by the upstream model.
    """

    def __init__(self, archive, threads: int = 1, activation_bits: int = 0):
        if isinstance(archive, (str, Path)):
            archive = Archive.load(archive)
        self.archive = archive
        self.metadata = archive.metadata
        self.heads = {}
        for name, count in (("contrastive_head", 4), ("confidence_head", 8)):
            if name + ".probes" not in archive.tensors:
                continue
            probes, projection, bias = (archive.tensors[name + "." + suffix].dequantize()
                                        for suffix in ("probes", "proj", "bias"))
            d = self.metadata["d_model"]
            if probes.shape != (count, d) or projection.ndim != 2 or projection.shape[1] != count * d:
                raise ValueError(f"invalid {name} probe/projection geometry")
            if bias.shape != (projection.shape[0],) or (name == "confidence_head" and projection.shape[0] != 1):
                raise ValueError(f"invalid {name} output geometry")
            self.heads[name] = (probes, projection, bias)
        if not self.heads:
            raise ValueError("archive contains no exported retrieval or confidence head")
        self.engine = NativeEngine(archive, threads=threads, activation_bits=activation_bits)
        embedding = archive.tensors["embedding"]
        self._embedding = NativeCQ.from_record(embedding) if embedding.dtype == CQ else embedding.dequantize()

    def _run(self, token_ids, requested, prefix_len):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError("probe encoding requires a nonempty one-dimensional integer sequence")
        if np.any((ids < 0) | (ids >= self.metadata["vocab_size"])):
            raise ValueError("token outside vocabulary")
        if np.any(ids == self.metadata.get("pad_token_id", 0)):
            raise ValueError("NativeProbeEncoder requires unpadded token IDs")
        if ids.size > self.metadata["max_seq_len"]:
            raise ValueError("probe sequence exceeds max_seq_len")
        if not 0 <= prefix_len <= ids.size:
            raise ValueError("prefix_len must be between zero and the sequence length")
        missing = set(requested) - self.heads.keys()
        if missing:
            raise ValueError(f"archive is missing requested heads: {sorted(missing)}")
        pools = {name: _OnlineProbePool(self.heads[name][0]) for name in requested}
        self.engine.reset(prefix_len=prefix_len)
        d = self.metadata["d_model"]
        cells = np.empty((self.metadata["num_layers"] + 1, d), dtype=np.float32)
        for token in ids:
            _, hidden = self.engine.step(int(token), compute_logits=False, return_hidden=True)
            if isinstance(self._embedding, NativeCQ):
                embedding = self._embedding.rows(np.asarray([token], dtype=np.int64))[0]
            else:
                embedding = self._embedding[int(token)]
            cells[0] = embedding * np.float32(math.sqrt(d))
            cells[1:] = hidden
            for pool in pools.values():
                pool.update(cells)
        outputs = {}
        for name, pool in pools.items():
            _, projection, bias = self.heads[name]
            projected = projection @ pool.result()
            if name == "contrastive_head":
                # The tied upstream retrieval projection has use_bias=False;
                # export stores a dummy zero bias to keep the head layout fixed.
                norm = np.sqrt(np.sum(projected * projected, dtype=np.float32) + np.float32(1e-12))
                outputs["embedding"] = (projected / norm).astype(np.float32)
            else:
                outputs["confidence_logit"] = float(projected[0] + bias[0])
        return outputs

    def encode(self, token_ids, *, prefix_len: int = 0) -> np.ndarray:
        """Return the unit-length contrastive retrieval embedding."""
        return self._run(token_ids, ("contrastive_head",), prefix_len)["embedding"]

    def confidence_logit(self, token_ids, *, prefix_len: int = 0) -> float:
        """Return the confidence head's raw logit."""
        return self._run(token_ids, ("confidence_head",), prefix_len)["confidence_logit"]

    def both(self, token_ids, *, prefix_len: int = 0) -> dict:
        """Return ``embedding`` and ``confidence_logit`` in a single model pass."""
        return self._run(token_ids, ("contrastive_head", "confidence_head"), prefix_len)
