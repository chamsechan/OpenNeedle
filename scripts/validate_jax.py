#!/usr/bin/env python3
"""Compare the PyTorch port against upstream JAX on the official master weights.

Optional reference dependencies: ``pip install 'jax[cpu]' flax``.
This imports only the vendored research model, never the proprietary runtime.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/official/checkpoints/needle2.pkl")
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/needle/needle/model")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_jax_parity.json")
    parser.add_argument("--tokens", default="2,176,45,3,128,52,765,1298,40,819,901,123,51,2001,97,1")
    args = parser.parse_args()
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from needle2.convert import canonical_weights, read_official_checkpoint
    from needle2.model import NeedleModel

    package = types.ModuleType("_needle2_validation_upstream")
    package.__path__ = [str(args.upstream.resolve())]
    sys.modules[package.__name__] = package
    upstream = importlib.import_module(package.__name__ + ".architecture")
    checkpoint = read_official_checkpoint(args.checkpoint)
    metadata = dict(checkpoint["config"])
    metadata.update(dtype="float32", flash=False, remat=False)
    config = upstream.TransformerConfig(**metadata)
    params = jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.float32), checkpoint["params"])
    weights = canonical_weights(checkpoint["params"], metadata)
    torch.set_num_threads(1)
    model = NeedleModel(metadata, {name: torch.from_numpy(value.copy()) for name, value in weights.items()})
    tokens = np.asarray([[int(token) for token in args.tokens.split(",")]], dtype=np.int32)
    # This comparison is deliberately shorter than the window: the upstream
    # __call__ uses a full causal mask, while the deployment config has a window.
    if metadata.get("kv_window", 0) and tokens.shape[1] > metadata["kv_window"]:
        raise ValueError("use at most kv_window tokens with the architecture.__call__ oracle")
    expected = np.asarray(upstream.SimpleAttentionNetwork(config).apply({"params": params}, jnp.asarray(tokens)))
    with torch.no_grad():
        actual = model(torch.from_numpy(tokens)).numpy()
    difference = np.abs(expected - actual)
    metrics = {
        "source": "official checkpoints/needle2.pkl FP16 master cast FP32",
        "oracle": "upstream architecture.SimpleAttentionNetwork.apply dtype=float32 flash=False",
        "shape": list(expected.shape),
        "max_abs": float(difference.max()),
        "rms_error": float(np.sqrt(np.mean(difference * difference))),
        "cosine_similarity": float(np.sum(expected * actual) / np.sqrt(np.sum(expected * expected) * np.sum(actual * actual))),
        "top1_agreement": float(np.mean(expected.argmax(-1) == actual.argmax(-1))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0005)
    if metrics["top1_agreement"] != 1.0:
        raise AssertionError("reference next-token predictions differ")


if __name__ == "__main__":
    main()
