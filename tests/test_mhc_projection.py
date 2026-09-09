"""Dense mHC decode must agree with the independent batched projection path."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from needle2.archive import FP32, TensorRecord
from needle2.model import NeedleModel
from needle2.native import NativeEngine
from test_model import tiny_model


@pytest.mark.parametrize('lanes', [1, 2, 3, 4])
@pytest.mark.parametrize('kv_cache', ['fp32', 'int8'])
def test_mhc_dense_projection_handles_odd_lanes_and_window(lanes, kv_cache):
    config = replace(tiny_model(window=5).config, mhc_lanes=lanes)
    model = NeedleModel(config)
    records = {}
    for name, tensor in model.canonical_state_dict().items():
        data = tensor.numpy()
        records[name] = TensorRecord(name, FP32, data.shape, data.tobytes())
    metadata = config.to_dict()
    metadata['hada_n'] = 16
    archive = SimpleNamespace(metadata=metadata, tensors=records)
    decode = NativeEngine(archive, threads=4, kv_cache=kv_cache)
    reference = NativeEngine(archive, threads=1, kv_cache=kv_cache)
    # One-token prefill uses the existing separate matrix projection code.
    for token in [2, 7, 13, 4, 29, 8, 16, 3, 9, 20, 11, 5]:
        actual = decode.step(token)
        expected = reference.prefill([token], last_only=True)
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
    assert decode.position == reference.position == 12
