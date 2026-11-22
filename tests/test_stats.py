import numpy as np
import pytest

from netcal import hpdi, is_in_quantile
from netcal.metrics import PICP


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
@pytest.mark.parametrize("draws", [1, 7])
def test_hpdi_full_probability(axis, dtype, draws):
    samples = np.arange(draws * 6, dtype=dtype).reshape(draws, 2, 3)[::-1]
    samples = np.moveaxis(samples, 0, axis)
    original = samples.copy()

    result = hpdi(samples, prob=1.0, axis=axis)

    expected = np.stack([samples.min(axis=axis), samples.max(axis=axis)], axis=axis)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(samples, original)
    assert result.dtype == samples.dtype


@pytest.mark.parametrize(
    "prob, expected",
    [(0.0, [0.0, 0.0]), (0.2, [0.0, 1.0]), (0.6, [0.0, 3.0]), (0.9, [0.0, 100.0])],
)
def test_hpdi_partial_probability(prob, expected):
    samples = np.array([100.0, 3.0, 0.0, 2.0, 1.0])

    np.testing.assert_array_equal(hpdi(samples, prob=prob), expected)


def test_is_in_quantile_full_empirical_interval():
    samples = np.array([[-2.0, -1.0], [0.0, 2.0], [3.0, 5.0]])
    targets = np.array([-2.0, 6.0])

    covered, _, (lower, upper), _, _ = is_in_quantile(
        samples, targets, q=[0.5, 1.0], kind="meanstd"
    )

    np.testing.assert_array_equal(lower[-1, :, 0], samples.min(axis=0))
    np.testing.assert_array_equal(upper[-1, :, 0], samples.max(axis=0))
    np.testing.assert_array_equal(covered[-1, :, 0], [True, False])


@pytest.mark.parametrize("reduction", ["none", "mean", "batchmean"])
def test_picp_full_empirical_interval(reduction):
    samples = np.array([[-2.0, -1.0], [0.0, 2.0], [3.0, 5.0]])
    targets = np.array([-2.0, 6.0])

    result = PICP().measure(samples, targets, q=[1.0], reduction=reduction)

    np.testing.assert_allclose(result.picp, 0.5)
    np.testing.assert_allclose(result.mpiw, 5.5)
