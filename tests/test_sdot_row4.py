"""Four-row SDOT must preserve the existing single-row arithmetic exactly."""
import numpy as np
import pytest

from needle2.native import NativeCQ, sdot_available
from needle2.quantize import codebook, quantize_matrix


@pytest.mark.parametrize('bits', [2, 4])
@pytest.mark.parametrize('group', [64, 128])
@pytest.mark.parametrize('threads', [1, 2, 4])
@pytest.mark.parametrize('rows', [9, 129])
def test_four_row_sdot_matches_single_row_with_padding_and_tail(bits, group, threads, rows):
    if not sdot_available():
        pytest.skip('ARM DotProd is unavailable')
    rng = np.random.default_rng(42)
    # Both shapes have a scalar tail; 129 rows also cross the parallel threshold.
    # 129 columns exercise padding in both groups. Row norms differ deliberately.
    weights = rng.normal(size=(rows, 129)).astype(np.float32)
    weights *= np.arange(1, rows + 1, dtype=np.float32)[:, None]
    packed, norms = quantize_matrix(weights, bits=bits, group_size=group)
    cb = codebook(bits, group_size=group)
    block = NativeCQ(packed, norms, weights.shape, bits, group, cb)
    singles = [NativeCQ(packed[i:i + 1], norms[i:i + 1], (1, 129), bits, group, cb)
               for i in range(rows)]
    for x in (rng.normal(size=129).astype(np.float32),
              np.zeros(129, dtype=np.float32), np.eye(1, 129, dtype=np.float32)[0]):
        expected = np.concatenate([row.linear_sdot(x, threads=1) for row in singles])
        np.testing.assert_array_equal(block.linear_sdot(x, threads=threads), expected)
