"""Numerical coverage for paired prompt projection and tiled GQA accumulation."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from needle2.archive import TensorRecord, FP32, CQ
from needle2.model import NeedleConfig, NeedleModel
from needle2.native import NativeEngine, sdot_available
from needle2.quantize import quantize_matrix, codebook


@pytest.mark.parametrize('head_dim', [32, 38, 64])
@pytest.mark.parametrize('kv', ['fp32', 'int8'])
def test_tiled_gqa_prefill_and_window_restore(head_dim, kv):
    torch.manual_seed(430)
    config = NeedleConfig(vocab_size=37, d_model=48, num_heads=6, num_kv_heads=2,
                          num_layers=2, head_dim=head_dim, mhc_lanes=2,
                          engram_layers=(0, 1), engram_slots=19, engram_sub_dim=12,
                          num_engram_tables=4, max_seq_len=128, kv_window=17)
    model = NeedleModel(config, quant_activations=False)
    records = {}
    for name, tensor in model.canonical_state_dict().items():
        a = tensor.numpy()
        if a.ndim == 2 and (name == 'embedding' or name.endswith(('_proj', '.tables'))):
            bits = 4 if name == 'embedding' or name.startswith('layer01') else 2
            packed, norms = quantize_matrix(a, bits, 64)
            records[name] = TensorRecord(name, CQ, a.shape, packed.tobytes()+norms.tobytes(), 64, bits, codebook(bits, 64))
        else:
            records[name] = TensorRecord(name, FP32, a.shape, a.tobytes())
    meta = config.to_dict(); meta['hada_n'] = 64
    archive = SimpleNamespace(metadata=meta, tensors=records)
    tokens = np.random.default_rng(81).integers(1, 37, 79, dtype=np.int32)
    # GQA ratio 3 exercises paired and unpaired heads; 38 exercises SIMD tails,
    # and 64 exercises the INT8-K fixed-dimension path across short and wrapped contexts.
    for mode in ['fp32'] + (['sdot'] if sdot_available() else []):
        incremental = NativeEngine(archive, threads=1, matmul=mode, kv_cache=kv)
        batched = NativeEngine(archive, threads=4, matmul=mode, kv_cache=kv)
        for e in (incremental, batched): e.reset(prefix_len=7)
        expected = np.stack([incremental.step(int(t)) for t in tokens[:7]])
        np.testing.assert_allclose(batched.prefill(tokens[:7]), expected, atol=3e-6, rtol=5e-5)
        for e in (incremental, batched): e.cache_prefix()
        expected = np.stack([incremental.step(int(t)) for t in tokens[7:]])
        actual = batched.prefill(tokens[7:])
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=5e-5)
        if kv == 'fp32' and mode == 'fp32':
            reference = NeedleModel.from_archive(archive).eval()
            sink = torch.zeros((1, len(tokens)), dtype=torch.bool); sink[:, :7] = True
            with torch.inference_mode():
                oracle = reference(torch.tensor(tokens[None].astype(np.int64)), sink_mask=sink)[0, 7:].numpy()
            np.testing.assert_allclose(actual, oracle, atol=5e-6, rtol=1e-4)
        for e in (incremental, batched): e.reset_to_prefix()
        expected = np.stack([incremental.step(int(t)) for t in tokens[7:26]])
        np.testing.assert_allclose(batched.prefill(tokens[7:26]), expected, atol=3e-6, rtol=5e-5)
