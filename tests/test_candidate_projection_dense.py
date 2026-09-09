"""The near-full SDOT projection must preserve ordered candidate scores exactly."""
from pathlib import Path

import numpy as np
import pytest

from needle2.native import NativeEngine, sdot_available

MODEL = Path('artifacts/official/needle2.cact')


@pytest.mark.integration
@pytest.mark.parametrize('threads', [1, 2, 4])
@pytest.mark.parametrize('kv_cache', ['fp32', 'int8'])
def test_dense_candidates_preserve_scores_order_duplicates_and_state(threads, kv_cache):
    if not MODEL.is_file() or not sdot_available():
        pytest.skip('Requires official model and ARM SDOT')
    full = NativeEngine(MODEL, threads=threads, matmul='sdot', kv_cache=kv_cache)
    selected = NativeEngine(MODEL, threads=threads, matmul='sdot', kv_cache=kv_cache)
    rng = np.random.default_rng(51)
    for count in [142, 7679, 7680, 8037, 8192]:
        ids = rng.permutation(8192)[:count].astype(np.int32)
        ids[-1] = ids[0]  # Ordering and duplicate candidates must survive gather.
        full.reset()
        selected.reset()
        for token in [2, 51, 802, 17]:
            expected, expected_hidden = full.step(token, return_hidden=True)
            actual, actual_hidden = selected.step_candidates(token, ids, return_hidden=True)
            np.testing.assert_array_equal(actual, expected[ids])
            np.testing.assert_array_equal(actual_hidden, expected_hidden)
            assert full.position == selected.position
        # Subsequent full projections also expose any accidental state changes.
        np.testing.assert_array_equal(selected.step(33), full.step(33))
