"""Numerical and layout tests against an independent dense Walsh construction."""
import unittest

import numpy as np

from needle2.native import NativeCQ


def example(bits, rows=9, dim=257, group=128):
    rng = np.random.default_rng(104 + bits)
    padded = (dim + group - 1) // group * group
    storage = 2 if bits == 5 else bits
    cb = np.linspace(-2, 2, 3 if bits == 5 else 1 << bits, dtype=np.float32) / np.sqrt(np.float32(group))
    indices = rng.integers(0, len(cb), (rows, padded), dtype=np.uint8)
    crumbs = np.where(indices == 0, 3, indices - 1).astype(np.uint8) if bits == 5 else indices
    packed = np.zeros((rows, padded * storage // 8), dtype=np.uint8)
    for col in range(padded):
        bit = col * storage
        packed[:, bit // 8] |= crumbs[:, col] << (bit % 8)
        if bit % 8 + storage > 8:
            packed[:, bit // 8 + 1] |= crumbs[:, col] >> (8 - bit % 8)
    norms = rng.uniform(0.1, 2, (rows, padded // group)).astype(np.float16)
    norms[0] = 0  # Cover exact zero groups and rows.
    hadamard = np.ones((1, 1), dtype=np.float32)
    while len(hadamard) < group:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    hadamard /= np.sqrt(np.float32(group))
    dense = ((cb[indices].reshape(rows, -1, group) * norms.astype(np.float32)[..., None]) @ hadamard).reshape(rows, padded)[:, :dim]
    return NativeCQ(packed, norms, (rows, dim), bits, group, cb), dense


class NativeTests(unittest.TestCase):
    def test_packed_linear_matches_dense(self):
        for bits in (2, 3, 4, 5):
            with self.subTest(bits=bits):
                q, dense = example(bits)
                x = np.random.default_rng(9).normal(size=(2, 3, 257)).astype(np.float32)
                for threads in (1, 2):
                    for lookup in (0, 1, 2, 3):
                        np.testing.assert_allclose(q.linear(x, threads, lookup=lookup), x @ dense.T, atol=3e-6, rtol=2e-5)
                np.testing.assert_allclose(q.dequantize(), dense, atol=2e-7, rtol=3e-5)
                np.testing.assert_allclose(q.rows(2), dense[2], atol=2e-7, rtol=3e-5)
                np.testing.assert_allclose(q.rows(np.array([[8, 2], [0, 2]])), dense[[[8, 2], [0, 2]]], atol=2e-7, rtol=3e-5)

    def test_bounds_and_shape(self):
        q, _ = example(2)
        with self.assertRaises(IndexError):
            q.rows([-1])
        with self.assertRaises(IndexError):
            q.rows([q.shape[0]])
        with self.assertRaises(TypeError):
            q.rows([1.5])
        with self.assertRaises(ValueError):
            q.linear(np.ones(256))
        self.assertEqual(q.linear(np.empty((0, 257))).shape, (0, 9))


if __name__ == "__main__":
    unittest.main()


# This test uses a randomized small architecture so sliding-window eviction,
# frozen prefix sinks, and delayed nonidentity Engram taps are all exercised.
def test_native_engine_matches_torch_and_keeps_prefix_sinks():
    from types import SimpleNamespace
    import torch
    from needle2.archive import TensorRecord, FP32, CQ
    from needle2.native import NativeEngine
    from needle2.model import NeedleModel
    from needle2.quantize import quantize_matrix, codebook
    from test_model import tiny_model

    torch.set_num_threads(1)
    for bits, lookup in ((0, 0), (8, 0), (0, 3), (0, 4)):
        model = tiny_model(window=5, quant=bool(bits))
        records = {}
        for name, tensor in model.canonical_state_dict().items():
            a = tensor.numpy()
            if a.ndim == 2 and (name == "embedding" or name.endswith("_proj") or name.endswith(".tables")):
                packed, norms = quantize_matrix(a, bits=2)
                records[name] = TensorRecord(name, CQ, a.shape, packed.tobytes() + norms.tobytes(), 128, 2, codebook(2, 128))
            else:
                records[name] = TensorRecord(name, FP32, a.shape, a.tobytes())
        meta = model.config.to_dict()
        meta["hada_n"] = 16
        archive = SimpleNamespace(metadata=meta, tensors=records)
        dense_model = NeedleModel(model.config, {n: torch.from_numpy(r.dequantize()) for n, r in records.items()}, quant_activations=bool(bits))
        engine = NativeEngine(archive, activation_bits=bits, projection_lookup=lookup)
        ids = torch.randint(1, 31, (1, 25))
        for prefix in (0, 3):
            engine.reset(prefix_len=prefix)
            mask = torch.arange(ids.shape[1])[None] < prefix
            with torch.no_grad():
                expected = dense_model(ids, sink_mask=mask).numpy()[0]
            actual = engine.prefill(ids.numpy()[0])
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=5e-5)
            engine.reset(prefix_len=prefix)
            last = engine.prefill(ids.numpy()[0, :17], last_only=True, backend="torch")
            np.testing.assert_allclose(last, expected[16], atol=2e-6, rtol=5e-5)
            tail = engine.prefill(ids.numpy()[0, 17:])
            np.testing.assert_allclose(tail, expected[17:], atol=2e-6, rtol=5e-5)
            engine.release_prefill_model()


def test_native_rejects_corrupt_norms_and_codebooks():
    import pytest
    q, _ = example(2)
    for value in (-1.0, np.inf, np.nan):
        norms = q.norms.copy()
        norms[0, 0] = value
        with pytest.raises(ValueError, match="norms must be finite and nonnegative"):
            NativeCQ(q.packed, norms, q.shape, q.bits, q.group_size, q.codebook)
    for codebook in ([np.nan, 0, 1, 2], [-2, -1, 0, np.inf], [2, 1, 0, -1], [0, 0, 1, 2]):
        with pytest.raises(ValueError, match="codebook must be finite and strictly sorted"):
            NativeCQ(q.packed, q.norms, q.shape, q.bits, q.group_size, codebook)


def test_native_generate_caps_context_before_prefill():
    # Exercise generation's public length policy without constructing weights.
    from needle2.native import NativeEngine
    import pytest
    engine = object.__new__(NativeEngine)
    engine.metadata = {"max_seq_len": 5}
    calls = []
    engine.reset = lambda prefix_len=0: calls.append(("reset", prefix_len))
    engine.prefill = lambda ids, **kw: np.array([0., 1., 0.], np.float32)
    engine.step = lambda token: np.array([0., 1., 0.], np.float32)
    assert engine.generate([2, 2, 2], max_new_tokens=999) == [2, 2, 2, 1, 1]
    assert engine.generate([2] * 5, max_new_tokens=999) == [2] * 5
    calls.clear()
    with pytest.raises(ValueError, match="prompt exceeds max_seq_len"):
        engine.generate([2] * 6, max_new_tokens=999)
    assert not calls


def test_sdot_tail_groups_match_independent_integer_oracle():
    import pytest
    from needle2.native import sdot_available
    from needle2.quantize import hadamard_matrix, unpack_indices
    if not sdot_available():
        pytest.skip("ARM DotProd unavailable")
    for bits in (2, 4):
        for group in (64, 128):
            q, _ = example(bits, dim=257, group=group)
            x = np.random.default_rng(315).normal(size=(5, 257)).astype(np.float32)
            x[0] = 0
            x[1] = 0
            x[1, 0] = 1
            padded = q.norms.shape[1] * group
            rot = np.pad(x, ((0, 0), (0, padded-257))).reshape(5, -1, group) @ hadamard_matrix(group)
            scale = np.max(np.abs(rot), axis=-1) / np.float32(127)
            scale = np.where(scale > 0, scale, np.float32(1))
            act = np.rint(rot / scale[..., None]).clip(-127, 127).astype(np.int32)
            centroid_scale = np.max(np.abs(q.codebook)) / np.float32(127)
            centroids = np.rint(q.codebook / centroid_scale).clip(-127, 127).astype(np.int32)
            indices = unpack_indices(q.packed, bits, padded)
            weights = centroids[indices].reshape(q.shape[0], -1, group)
            dot = np.einsum("bgd,ogd->bog", act, weights).astype(np.float32)
            expected = (dot * (q.norms.astype(np.float32) * centroid_scale)[None] * scale[:, None, :]).sum(-1)
            for threads in (1, 2):
                np.testing.assert_allclose(q.linear_sdot(x, threads), expected, atol=2e-6, rtol=2e-5)


def test_sdot_whole_tiny_error_budget_and_threads():
    import pytest
    import torch
    from types import SimpleNamespace
    from needle2.archive import TensorRecord, FP32, CQ
    from needle2.native import NativeEngine, sdot_available
    from needle2.quantize import quantize_matrix, codebook
    from test_model import tiny_model
    if not sdot_available():
        pytest.skip("ARM DotProd unavailable")
    torch.set_num_threads(1)
    model = tiny_model(window=5)
    records = {}
    for name, tensor in model.canonical_state_dict().items():
        a = tensor.numpy()
        if a.ndim == 2 and (name == "embedding" or name.endswith(("_proj", ".tables"))):
            bits = 4 if name == "embedding" else 2
            packed, norms = quantize_matrix(a, bits=bits)
            records[name] = TensorRecord(name, CQ, a.shape, packed.tobytes()+norms.tobytes(), 128, bits, codebook(bits))
        else:
            records[name] = TensorRecord(name, FP32, a.shape, a.tobytes())
    meta = model.config.to_dict()
    meta["hada_n"] = 16
    archive = SimpleNamespace(metadata=meta, tensors=records)
    ids = np.random.default_rng(25).integers(1, 31, 29)
    reference = NativeEngine(archive)
    reference.reset(prefix_len=3)
    expected = reference.prefill(ids)
    previous = None
    for threads in (1, 2):
        engine = NativeEngine(archive, threads=threads, matmul="sdot")
        engine.reset(prefix_len=3)
        actual = engine.prefill(ids)
        assert np.isfinite(actual).all()
        # SDOT intentionally adds two INT8 rounding steps; FP32 tolerances do
        # not apply. Bound accumulated error over the full three-layer model.
        assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < .03
        if previous is not None:
            np.testing.assert_array_equal(actual, previous)
        previous = actual
    with pytest.raises(ValueError, match="already rounds rotated activations"):
        NativeEngine(archive, matmul="sdot", activation_bits=8)


def test_explicit_prefix_snapshot_restores_two_requests_exactly():
    import pytest
    import torch
    from types import SimpleNamespace
    from needle2.archive import TensorRecord, FP32, CQ
    from needle2.native import NativeEngine, sdot_available
    from needle2.quantize import quantize_matrix, codebook
    from test_model import tiny_model
    torch.set_num_threads(1)
    model = tiny_model(window=5)
    records = {}
    for name, tensor in model.canonical_state_dict().items():
        a = tensor.numpy()
        if a.ndim == 2 and (name == "embedding" or name.endswith(("_proj", ".tables"))):
            bits = 4 if name == "embedding" else 2
            packed, norms = quantize_matrix(a, bits=bits)
            records[name] = TensorRecord(name, CQ, a.shape, packed.tobytes()+norms.tobytes(), 128, bits, codebook(bits))
        else:
            records[name] = TensorRecord(name, FP32, a.shape, a.tobytes())
    meta = model.config.to_dict()
    meta["hada_n"] = 16
    archive = SimpleNamespace(metadata=meta, tensors=records)
    rng = np.random.default_rng(1209)
    prefix = rng.integers(1, 31, 9)
    suffixes = [rng.integers(1, 31, 21), rng.integers(1, 31, 17)]
    modes = ["fp32"] + (["sdot"] if sdot_available() else [])
    for matmul in modes:
        reused = NativeEngine(archive, threads=2, matmul=matmul)
        with pytest.raises(RuntimeError, match="no cached prefix"):
            reused.reset_to_prefix()
        with pytest.raises(ValueError, match="position == prefix_len"):
            reused.cache_prefix()
        reused.reset(prefix_len=len(prefix))
        reused.prefill(prefix[:-1], last_only=True)
        with pytest.raises(ValueError, match="position == prefix_len"):
            reused.cache_prefix()
        reused.step(int(prefix[-1]), compute_logits=False)
        reused.cache_prefix()  # Exactly one explicit snapshot for both requests.
        for suffix in suffixes:
            reused.reset_to_prefix()
            assert reused.position == reused.prefix_len == len(prefix)
            actual = reused.prefill(suffix)
            fresh = NativeEngine(archive, threads=2, matmul=matmul)
            fresh.reset(prefix_len=len(prefix))
            fresh.prefill(prefix, last_only=True)
            expected = fresh.prefill(suffix)
            # Snapshot/restore introduces no numerical approximation, including
            # in SDOT mode: compare to recomputation using the same math mode.
            np.testing.assert_array_equal(actual, expected)
            with pytest.raises(ValueError, match="position == prefix_len"):
                reused.cache_prefix()
        # Even reset with an identical prefix length invalidates old content.
        reused.reset(prefix_len=len(prefix))
        with pytest.raises(RuntimeError, match="no cached prefix"):
            reused.reset_to_prefix()
