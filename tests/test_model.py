"""Behavioral tests for SAN, including an optional independent upstream oracle."""
import importlib
from pathlib import Path
import sys
import types
from dataclasses import asdict

import numpy as np
import pytest
import torch

from needle2.model import NeedleConfig, NeedleModel, engram_indices, hadamard


@pytest.fixture(autouse=True)
def single_torch_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(window=0, quant=False):
    torch.manual_seed(117)
    config = NeedleConfig(vocab_size=31, d_model=12, num_heads=2, num_kv_heads=1,
                          num_layers=3, head_dim=6, mhc_lanes=2,
                          engram_layers=(0, 2), engram_slots=19, engram_sub_dim=3,
                          num_engram_tables=4, max_seq_len=64, kv_window=window)
    model = NeedleModel(config, quant_activations=quant)
    # Nonidentity convolution taps are essential to exercise history at chunk
    # boundaries and catch implementations that omit the Engram convolution.
    with torch.no_grad():
        model.engram0.taps[1:].normal_(0, 0.1)
        model.engram1.taps[1:].normal_(0, 0.1)
    return model


@pytest.mark.parametrize("window", [0, 5, 16])
@pytest.mark.parametrize("quant", [False, True])
def test_prefill_chunk_and_incremental_agree(window, quant):
    model = tiny_model(window, quant)
    tokens = torch.randint(1, 31, (2, 29))
    with torch.no_grad():
        expected = model(tokens)
        cache = None
        pieces = []
        for start, end in [(0, 4), (4, 5), (5, 18), (18, 23), (23, 29)]:
            logits, cache = model(tokens[:, start:end], cache, use_cache=True)
            pieces.append(logits)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=2e-6, rtol=2e-5)
        assert cache.position == tokens.shape[1]
        if window:
            assert all(k.shape[2] <= window for k in cache.keys)
        assert cache.token_history.shape[1] <= 12


def test_padding_mask_preserves_valid_chunk_logits():
    model = tiny_model(5)
    tokens = torch.randint(1, 31, (2, 20))
    tokens[0, :3] = 0
    mask = tokens != 0
    with torch.no_grad():
        expected = model(tokens, attention_mask=mask)
        first, cache = model(tokens[:, :4], attention_mask=mask[:, :4], use_cache=True)
        second = model(tokens[:, 4:], cache, attention_mask=mask)
    torch.testing.assert_close(torch.cat((first, second), 1)[mask], expected[mask], atol=2e-6, rtol=2e-5)


def test_pinned_prefix_survives_window_eviction():
    model = tiny_model(5)
    tokens = torch.randint(1, 31, (2, 33))
    sink = torch.zeros_like(tokens, dtype=torch.bool)
    sink[0, :3] = True
    sink[1, :2] = True
    with torch.no_grad():
        expected = model(tokens, sink_mask=sink)
        cache, chunks = None, []
        for start, end in [(0, 3), (3, 15), (15, 16), (16, 29), (29, 33)]:
            logits, cache = model(tokens[:, start:end], cache, use_cache=True,
                                  sink_mask=sink[:, start:end])
            chunks.append(logits)
        torch.testing.assert_close(torch.cat(chunks, 1), expected, atol=2e-6, rtol=2e-5)
        assert cache.key_positions.tolist() == [0, 1, 2, 28, 29, 30, 31, 32]
        assert cache.keys[0].shape[2] == 8
        assert cache.key_sink[1, 2].item() is False
        assert not torch.allclose(expected[:, -1], model(tokens)[:, -1], atol=1e-5)


def test_backprop_reaches_engram_mhc_and_attention():
    model = tiny_model()
    tokens = torch.randint(1, 31, (2, 16))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    for name in ["embedding", "layer00.q_proj", "layer02.d1", "engram0.tables",
                 "engram1.value_proj", "engram1.taps", "mhc_phi_pre", "mhc_phi_res"]:
        gradient = model.get_parameter(name).grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_hadamard_matches_sylvester_matrix_and_is_self_inverse():
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 4
    x = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(hadamard(x), x @ h.double())
    torch.testing.assert_close(hadamard(hadamard(x)), x)
    assert torch.autograd.gradcheck(hadamard, (x,))


def test_canonical_serialization_and_input_checks():
    model = tiny_model()
    state = model.canonical_state_dict()
    assert "layer00.q_proj" in state and "mhc_phi_pre" in state
    loaded = NeedleModel(model.config.to_dict(), state)
    tokens = torch.tensor([[2, 8, 10]])
    torch.testing.assert_close(model(tokens), loaded(tokens), rtol=0, atol=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones(1, 65, dtype=torch.long))
    with pytest.raises(ValueError, match="nonempty_time"):
        model(tokens[:, :0])
    with pytest.raises(ValueError, match="missing canonical"):
        NeedleModel(model.config, {})


def test_greedy_generation_matches_full_recompute():
    model = tiny_model(5)
    prompt = torch.tensor([[2, 5, 9]])
    expected = prompt
    with torch.no_grad():
        for _ in range(5):
            expected = torch.cat((expected, model(expected)[:, -1].argmax(-1)[:, None]), dim=1)
    torch.testing.assert_close(model.generate(prompt, max_new_tokens=5), expected)
    torch.testing.assert_close(model.generate(prompt, max_new_tokens=0), prompt)


def _upstream_modules():
    """Import research modules without importing upstream's hosted-agent API."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    source = Path(__file__).parents[1] / "third_party/needle/needle/model"
    if not (source / "architecture.py").exists():
        pytest.skip("upstream research checkout not installed")
    name = "_needle2_test_upstream"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(source)]
        sys.modules[name] = package
    return (importlib.import_module(name + ".architecture"),
            importlib.import_module(name + ".decode"))


def test_matches_independent_upstream_jax_architecture():
    upstream, _ = _upstream_modules()
    import jax
    import jax.numpy as jnp
    from needle2.convert import canonical_weights

    config = upstream.TransformerConfig(vocab_size=31, d_model=12, num_heads=2,
                                       num_kv_heads=1, num_layers=3, mhc_lanes=2,
                                       engram_layers=(0, 2), engram_slots=19,
                                       engram_heads=2, dtype="float32", flash=False,
                                       remat=False, max_seq_len=64)
    tokens = np.random.default_rng(199).integers(1, 31, (2, 16), dtype=np.int32)
    network = upstream.SimpleAttentionNetwork(config)
    params = network.init(jax.random.PRNGKey(19), jnp.asarray(tokens))["params"]
    # Exercise learned convolution and nontrivial norm scales as well as the
    # architecture's initialization defaults.
    for site in range(2):
        params[f"engrams_{site}"]["taps"] = jnp.asarray(
            np.random.default_rng(site).normal(0, 0.2, (4, 12)).astype(np.float32))
    params["stack"]["layers"]["block"]["self_attn"]["q_norm"]["scale"] = jnp.full((3, 6), 0.13)
    metadata = asdict(config)
    weights = canonical_weights(params, metadata)
    model = NeedleModel(metadata, {n: torch.from_numpy(w.copy()) for n, w in weights.items()})
    expected = np.asarray(network.apply({"params": params}, jnp.asarray(tokens)))
    with torch.no_grad():
        actual = model(torch.from_numpy(tokens)).numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    expected_indices = np.asarray(upstream.engram_indices(jnp.asarray(tokens), (2, 3), 2, 19))
    np.testing.assert_array_equal(engram_indices(torch.from_numpy(tokens), (2, 3), 2, 19).numpy(),
                                  expected_indices)
