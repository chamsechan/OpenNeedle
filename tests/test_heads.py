"""Streaming probes must agree with full PyTorch token/layer attention pooling."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from needle2.archive import CQ, FP32, TensorRecord
from needle2.heads import NativeProbeEncoder, _OnlineProbePool
from needle2.model import NeedleModel
from needle2.quantize import codebook, quantize_matrix


def test_online_pool_is_stable_and_partition_invariant():
    rng = np.random.default_rng(810)
    cells = rng.normal(size=(67, 12)).astype(np.float32)
    probes = rng.normal(size=(4, 12)).astype(np.float32) * 40
    scores = probes @ cells.T / np.sqrt(np.float32(12))
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    expected = ((weights / weights.sum(-1, keepdims=True)) @ cells).reshape(-1)
    pool = _OnlineProbePool(probes)
    for block in np.array_split(cells, [3, 17, 18, 40]):
        pool.update(block)
    np.testing.assert_allclose(pool.result(), expected, atol=1e-6, rtol=1e-5)
    assert np.isfinite(pool.result()).all()


def _small_head_archive():
    from test_model import tiny_model
    model = tiny_model(window=5)
    weights = model.canonical_state_dict()
    rng = np.random.default_rng(1234)
    weights["heads.manifest"] = torch.tensor([1., 2.])
    for name, probes, output in (("contrastive_head", 4, 7), ("confidence_head", 8, 1)):
        weights[name + ".probes"] = torch.from_numpy(rng.normal(0, .2, (probes, 12)).astype(np.float32))
        weights[name + ".proj"] = torch.from_numpy(rng.normal(0, .1, (output, probes * 12)).astype(np.float32))
        weights[name + ".bias"] = torch.zeros(output) if output != 1 else torch.tensor([.123])
    records = {}
    for name, tensor in weights.items():
        array = tensor.numpy()
        if name == "embedding" or name.endswith(("_proj", ".tables")):
            packed, norms = quantize_matrix(array, 2, 128)
            records[name] = TensorRecord(name, CQ, array.shape, packed.tobytes() + norms.tobytes(),
                                         128, 2, codebook(2, 128))
        else:
            records[name] = TensorRecord(name, FP32, array.shape, array.tobytes())
    metadata = model.config.to_dict()
    metadata["hada_n"] = 16
    archive = SimpleNamespace(metadata=metadata, tensors=records)
    reference = NeedleModel.from_archive(archive).eval()
    return archive, reference


def test_native_probe_heads_match_torch():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        archive, model = _small_head_archive()
        encoder = NativeProbeEncoder(archive)
        ids = torch.tensor([[2, 5, 7, 8, 11, 29, 7, 4, 8, 11, 19, 14, 25, 9, 3, 1, 8]])
        with torch.no_grad():
            embedding = model.encode_contrastive(ids).numpy()[0]
            confidence = model.forward_confidence(ids).item()
        result = encoder.both(ids.numpy()[0])
        np.testing.assert_allclose(result["embedding"], embedding, atol=2e-6, rtol=5e-5)
        np.testing.assert_allclose(result["confidence_logit"], confidence, atol=2e-6, rtol=5e-5)
        assert encoder.engine.position == ids.shape[1]
        np.testing.assert_allclose(np.linalg.norm(result["embedding"]), 1., atol=1e-6)
        # One-head methods have the same semantics and reset previous state.
        np.testing.assert_allclose(encoder.encode(ids.numpy()[0]), result["embedding"], atol=2e-6)
        np.testing.assert_allclose(encoder.confidence_logit(ids.numpy()[0]), result["confidence_logit"], atol=2e-6)
        with pytest.raises(ValueError, match="unpadded"):
            encoder.encode([2, 0, 3])
    finally:
        torch.set_num_threads(previous_threads)


def test_tool_retrieval_and_prompt_pruning():
    from pathlib import Path
    model_path = Path("artifacts/official/needle2.cact")
    if not model_path.exists():
        pytest.skip("official cact artifact not present")

    from needle2.inference import generate, retrieve_tools
    from needle2.archive import Archive
    from needle2.tokenizer import RefTokenizer

    archive = Archive.load(model_path)
    tokenizer = RefTokenizer.from_cact(model_path)

    tools = [
        {
            "name": "get_weather",
            "description": "Fetch current weather conditions for a given city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
        {
            "name": "send_email",
            "description": "Send an electronic mail message to a recipient",
            "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]},
        },
        {
            "name": "calculator",
            "description": "Evaluate arbitrary mathematical expressions and calculate formulas",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        },
    ]

    # Test retrieve_tools ranking
    selected, scored = retrieve_tools(archive, tokenizer, "What is the temperature in London?", tools, top_k=1)
    assert len(selected) == 1
    assert selected[0]["name"] == "get_weather"
    assert scored[0][0] > scored[1][0]
    assert scored[0][0] > scored[2][0]

    # Test precomputed embedding bypasses encode
    dummy_emb = np.zeros(128, dtype=np.float32)
    dummy_emb[0] = 1.0
    tools_with_cached = [
        {"name": "cached_tool", "description": "some tool", "_embedding": dummy_emb},
        {"name": "other_tool", "description": "other tool"},
    ]
    selected_cached, _ = retrieve_tools(archive, tokenizer, "test", tools_with_cached, top_k=1)
    assert len(selected_cached) == 1

    # Test end-to-end generate() with retrieval enabled
    full_res = generate(model_path, "What is the temperature in London?", tools=tools, max_new_tokens=10, threads=2, retrieval=False)
    assert full_res["retrieval_enabled"] is False
    assert full_res["retrieved_tools"] is None

    pruned_res = generate(model_path, "What is the temperature in London?", tools=tools, max_new_tokens=10, threads=2, retrieval=True, top_k_tools=1)
    assert pruned_res["retrieval_enabled"] is True
    assert pruned_res["retrieved_tools"] == ["get_weather"]
    assert pruned_res["prompt_tokens"] < full_res["prompt_tokens"]
