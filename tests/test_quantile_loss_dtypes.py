import unittest
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from netcal.metrics.regression import PinballLoss, QuantileLoss


class QuantileLossDtypeTest(unittest.TestCase):
    metrics = (QuantileLoss, PinballLoss)
    reductions = ('none', None, 'mean', 'sum')

    def assert_reductions(self, X, y, q, expected, kind='cumulative'):
        for metric in self.metrics:
            for reduction in self.reductions:
                with self.subTest(metric=metric.__name__, reduction=reduction):
                    actual = metric().measure(X, y, q, kind=kind, reduction=reduction)
                    target = expected if reduction in ('none', None) else getattr(expected, reduction)()
                    assert_allclose(actual, target, rtol=1e-14, atol=0.)
                    if reduction in ('none', None):
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertNotEqual(actual.dtype, np.dtype(object))
                    else:
                        self.assertIsInstance(actual, float)

    @staticmethod
    def integer_oracle(grid, cdf, y, q):
        # Python integers and rational arithmetic avoid sharing NumPy's fixed-
        # width subtraction or the implementation's signed-residual formula.
        grid = grid[..., None] if grid.ndim == 2 else grid
        cdf = cdf[..., None] if cdf.ndim == 2 else cdf
        y = y[..., None] if y.ndim == 1 else y
        expected = np.empty((len(q), *y.shape))
        for k, tau in enumerate(q):
            level = Fraction(float(tau))
            for n, d in np.ndindex(y.shape):
                i = min(range(len(grid)), key=lambda j: abs(float(cdf[j, n, d]) - float(tau)))
                prediction, target = int(grid[i, n, d]), int(y[n, d])
                value = (target - prediction) * level if target >= prediction else (prediction - target) * (1 - level)
                expected[k, n, d] = float(value)
        return expected

    def test_integer_cdf_public_api_and_reductions(self):
        X = (np.array([[0], [2]]), np.array([[0.], [1.]]))
        for q in (.5, [.5], np.array([.5])):
            with self.subTest(q=q):
                self.assert_reductions(X, np.array([1]), q, np.array([[[.5]]]))

    def test_integer_overflow_and_large_offset_residuals(self):
        i64 = np.iinfo(np.int64)
        u64 = np.iinfo(np.uint64)
        cases = [
            ('uint8-negative', [0, 2], np.uint8, 1, np.uint8, .75, .25),
            ('int8-positive', [-128, 127], np.int8, 127, np.int8, .25, 63.75),
            ('int8-negative', [-128, 127], np.int8, -128, np.int8, .75, 63.75),
            ('int64-full-positive', [i64.min, i64.max], np.int64, i64.max, np.int64, .25, float(Fraction(2**64 - 1, 4))),
            ('int64-full-negative', [i64.min, i64.max], np.int64, i64.min, np.int64, .75, float(Fraction(2**64 - 1, 4))),
            ('int64-adjacent-positive', [i64.max - 1, i64.max], np.int64, i64.max, np.int64, .25, .25),
            ('int64-adjacent-negative', [i64.max - 1, i64.max], np.int64, i64.max - 1, np.int64, .75, .25),
            ('int64-min-adjacent', [i64.min, i64.min + 1], np.int64, i64.min + 1, np.int64, .25, .25),
            ('uint64-adjacent', [u64.max - 1, u64.max], np.uint64, u64.max, np.uint64, .25, .25),
            ('mixed-adjacent-positive', [i64.max - 1, i64.max], np.int64, i64.max + 1, np.uint64, .75, .75),
            ('mixed-adjacent-negative', [i64.max + 1, i64.max + 2], np.uint64, i64.max, np.int64, .25, .75),
            ('mixed-full-positive', [i64.min, i64.max], np.int64, u64.max, np.uint64, .25, float(Fraction(3 * 2**63 - 1, 4))),
            ('mixed-full-negative', [0, u64.max], np.uint64, i64.min, np.int64, .75, float(Fraction(3 * 2**63 - 1, 4))),
        ]
        for name, grid, grid_dtype, y, y_dtype, tau, value in cases:
            with self.subTest(case=name):
                self.assert_reductions(
                    (np.array(grid, dtype=grid_dtype)[:, None], np.array([[0.], [1.]])),
                    np.array([y], dtype=y_dtype), np.array([tau]), np.array([[[value]]]),
                )

    def test_all_standard_integer_dtype_pairs(self):
        dtypes = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64)
        q = np.array([0., .25, .5, .75, 1.])
        for grid_dtype in dtypes:
            for target_dtype in dtypes:
                with self.subTest(grid=grid_dtype, target=target_dtype):
                    g, t = np.iinfo(grid_dtype), np.iinfo(target_dtype)
                    grid = np.array([[g.min, g.min], [g.max, g.max]], dtype=grid_dtype)
                    cdf = np.array([[0., 0.], [1., 1.]])
                    y = np.array([t.min, t.max], dtype=target_dtype)
                    expected = self.integer_oracle(grid, cdf, y, q)
                    self.assert_reductions((grid, cdf), y, q, expected)

    def test_integer_residual_respects_quantile_precision(self):
        grid = np.array([[np.iinfo(np.int64).min], [np.iinfo(np.int64).max]])
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            with self.subTest(dtype=dtype):
                q = np.array([.25], dtype=dtype)
                actual = QuantileLoss().measure(
                    (grid, np.array([[0.], [1.]])), np.array([np.iinfo(np.int64).max]),
                    q, kind='cumulative', reduction='none',
                )
                expected_dtype = np.result_type(grid.dtype, q.dtype)
                self.assertEqual(actual.dtype, expected_dtype)
                # Decimal parsing also works where longdouble equals double.
                expected = np.array([[['4611686018427387903.75']]], dtype=expected_dtype)
                assert_array_equal(actual, expected)

    def test_integer_finite_support_endpoints(self):
        grid = np.array([[-2, -2, -2], [2, 2, 2]])
        cdf = np.array([[0., 0., 0.], [1., 1., 1.]])
        expected = np.array([[[1.], [0.], [0.]], [[0.], [0.], [1.]]])
        with np.errstate(invalid='raise', over='raise'):
            self.assert_reductions((grid, cdf), np.array([-3, 0, 3]), np.array([0., 1.]), expected)

    def test_cdf_multiple_samples_dimensions_and_ties(self):
        grid = np.array([
            [[-9, 0], [-2, 8], [100, -100]],
            [[-4, 2], [0, 10], [101, -90]],
            [[1, 4], [5, 20], [109, -70]],
            [[10, 8], [9, 30], [120, -60]],
        ], dtype=np.int64)
        cdf = np.array([
            [[0., 0.], [0., 0.], [0., 0.]],
            [[.125, .25], [.5, .125], [.25, .375]],
            [[.5, .75], [.75, .5], [.875, .75]],
            [[1., 1.], [1., 1.], [1., 1.]],
        ])
        y = np.array([[0, 3], [2, 40], [99, -65]], dtype=np.int64)
        q = np.array([0., .25, .5, .75, 1.])
        expected = self.integer_oracle(grid, cdf, y, q)
        self.assert_reductions((grid, cdf), y, q, expected)

    def test_floating_result_dtype_matches_weighted_product(self):
        for value_dtype in (np.float16, np.float32, np.float64, np.longdouble):
            for q_dtype in (np.float16, np.float32, np.float64, np.longdouble):
                with self.subTest(value=value_dtype, quantile=q_dtype):
                    grid = np.array([[0., 0.], [2., 2.]], dtype=value_dtype)
                    y = np.array([1. / 3., 1.5], dtype=value_dtype)
                    q = np.array([.123456789012345, .876543210987655], dtype=q_dtype)
                    cdf = np.array([[0., 0.], [1., 1.]])
                    error = y[None, :, None] - grid[:, :, None]
                    expected = np.maximum(q[:, None, None] * error, (q[:, None, None] - 1.) * error)
                    for metric in self.metrics:
                        actual = metric().measure((grid, cdf), y, q, kind='cumulative', reduction='none')
                        self.assertEqual(actual.dtype, expected.dtype)
                        assert_array_equal(actual, expected)

    def test_mixed_floating_and_integer_inputs(self):
        for grid_dtype, target_dtype in ((np.int16, np.float32), (np.float32, np.int16)):
            with self.subTest(grid=grid_dtype, target=target_dtype):
                grid = np.array([[0], [2]], dtype=grid_dtype)
                y = np.array([1], dtype=target_dtype)
                q = np.array([.25, .75], dtype=np.float64)
                self.assert_reductions((grid, np.array([[0.], [1.]])), y, q, np.array([[[.25]], [[.25]]]))

    def test_nan_targets_all_kinds_reductions_and_alias(self):
        cases = [
            ('meanstd', (np.zeros(1), np.ones(1))),
            ('cauchy', (np.zeros(1), np.ones(1))),
            ('meanstd', np.array([[-2.], [2.]])),
            ('cumulative', (np.array([[-2], [2]]), np.array([[0.], [1.]]))),
        ]
        for kind, X in cases:
            with self.subTest(kind=kind), np.errstate(invalid='raise'):
                self.assert_reductions(X, np.array([np.nan]), np.array([0., .25, .75, 1.]), np.full((4, 1, 1), np.nan), kind)

    def test_nan_selected_quantiles_not_hidden(self):
        cases = [
            ('meanstd', np.array([[np.nan], [np.nan]])),
            ('cumulative', (np.array([[np.nan], [np.nan]]), np.array([[0.], [1.]]))),
        ]
        for kind, X in cases:
            with self.subTest(kind=kind), np.errstate(invalid='raise'):
                self.assert_reductions(X, np.array([0.]), np.array([0., .5, 1.]), np.full((3, 1, 1), np.nan), kind)

    def test_unbounded_endpoints_all_reductions_and_alias(self):
        for kind in ('meanstd', 'cauchy'):
            with self.subTest(kind=kind), np.errstate(invalid='raise'):
                self.assert_reductions((np.zeros(3), np.ones(3)), np.array([-3., 0., 3.]), np.array([0., 1.]), np.zeros((2, 3, 1)), kind)

    def test_empirical_finite_support_endpoints(self):
        expected = np.array([[[1.], [0.], [0.]], [[0.], [0.], [1.]]])
        self.assert_reductions(np.array([[-2., -2., -2.], [2., 2., 2.]]), np.array([-3., 0., 3.]), np.array([0., 1.]), expected, 'meanstd')

    def test_inputs_not_mutated(self):
        grid = np.array([[np.iinfo(np.int64).max - 1], [np.iinfo(np.int64).max]])
        cdf, y, q = np.array([[0.], [1.]]), grid[-1].copy(), np.array([.25, .75])
        original = [x.copy() for x in (grid, cdf, y, q)]
        QuantileLoss().measure((grid, cdf), y, q, kind='cumulative')
        for actual, expected in zip((grid, cdf, y, q), original):
            assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
