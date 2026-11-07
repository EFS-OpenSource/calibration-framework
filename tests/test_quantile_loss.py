import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import cauchy, norm

from netcal import is_in_quantile
from netcal.metrics.regression import PinballLoss, QuantileLoss


class QuantileLossTest(unittest.TestCase):
    def test_requested_normal_quantile_has_zero_loss(self):
        for tau in (0.2, 0.5, 0.8):
            with self.subTest(tau=tau):
                actual = PinballLoss().measure(
                    (np.array([0.]), np.array([1.])),
                    np.array([norm.ppf(tau)]), np.array([tau]), reduction='none',
                )
                self.assertEqual(actual.shape, (1, 1, 1))
                assert_allclose(actual, 0., atol=1e-14)

    def test_parametric_quantiles_and_reductions(self):
        location = np.array([[0., 2.], [-1., 3.]])
        scale = np.array([[1., 2.], [0.5, 3.]])
        y = np.array([[-2., 5.], [0., -1.]])
        q = np.array([0.2, 0.5, 0.8])
        for kind, distribution in [('meanstd', norm), ('cauchy', cauchy)]:
            quantiles = distribution.ppf(q[:, None, None], loc=location, scale=scale)
            error = y - quantiles
            expected = np.maximum(q[:, None, None] * error, (q[:, None, None] - 1.) * error)
            for metric in (QuantileLoss(), PinballLoss()):
                for reduction in ('none', None, 'mean', 'sum'):
                    with self.subTest(kind=kind, metric=type(metric).__name__, reduction=reduction):
                        actual = metric.measure((location, scale), y, q, kind=kind, reduction=reduction)
                        target = expected if reduction in ('none', None) else getattr(expected, reduction)()
                        assert_allclose(actual, target)
                        if reduction in ('none', None):
                            self.assertEqual(actual.shape, (3, 2, 2))
                        else:
                            self.assertIsInstance(actual, float)

    def test_scalar_and_list_quantile_levels(self):
        for q in (0.8, [0.8], np.array([0.8])):
            with self.subTest(q=q):
                actual = QuantileLoss().measure(
                    (np.array([0.]), np.array([1.])), np.array([norm.ppf(.8)]), q,
                )
                assert_allclose(actual, 0., atol=1e-14)

    def test_empirical_quantiles(self):
        # A skewed sample distribution distinguishes quantiles from HPDI bounds
        # and from a Gaussian approximation to the sample moments.
        samples = np.array([0., 1., 2., 3., 20.])[:, None]
        for multidimensional in (False, True):
            X = np.stack((samples, samples + 10.), axis=-1) if multidimensional else samples
            y = np.array([[3., 13.]]) if multidimensional else np.array([3.])
            actual = QuantileLoss().measure(X, y, np.array([.25, .75]), reduction='none')
            expected = np.array([[[.5, .5]], [[0., 0.]]]) if multidimensional else np.array([[[.5]], [[0.]]])
            assert_allclose(actual, expected)

    def test_cumulative_quantiles(self):
        t = np.array([-4., -1., 0., 2., 8.])[:, None]
        cdf = np.array([.05, .2, .5, .8, .95])[:, None]
        for multidimensional in (False, True):
            X = (np.stack((t, t + 10.), axis=-1), np.stack((cdf, cdf), axis=-1)) if multidimensional else (t, cdf)
            y = np.array([[2., 12.]]) if multidimensional else np.array([2.])
            actual = QuantileLoss().measure(X, y, np.array([.2, .8]), kind='cumulative', reduction='none')
            expected = np.array([[[.6, .6]], [[0., 0.]]]) if multidimensional else np.array([[[.6]], [[0.]]])
            assert_allclose(actual, expected)

    def test_endpoint_quantiles(self):
        q = np.array([0., 1.])
        cases = [
            ('meanstd', (np.zeros(2), np.ones(2))),
            ('cauchy', (np.zeros(2), np.ones(2))),
            ('meanstd', np.array([[-2., -2.], [2., 2.]])),
            ('cumulative', (np.array([[-2., -2.], [2., 2.]]), np.array([[0., 0.], [1., 1.]]))),
        ]
        for kind, X in cases:
            with self.subTest(kind=kind, X=X), np.errstate(invalid='raise'):
                actual = QuantileLoss().measure(X, np.array([-1., 1.]), q, kind=kind, reduction='none')
                assert_allclose(actual, np.zeros((2, 2, 1)))

    def test_nan_targets_are_not_hidden_at_endpoints(self):
        actual = QuantileLoss().measure(
            (np.zeros(1), np.ones(1)), np.array([np.nan]), np.array([0., 1.]), reduction='none',
        )
        self.assertTrue(np.isnan(actual).all())

    def test_correlated_input_still_rejected(self):
        X = (np.zeros((2, 2)), np.repeat(np.eye(2)[None, ...], 2, axis=0))
        for kind in ('meanstd', 'cauchy'):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                QuantileLoss().measure(X, np.zeros((2, 2)), np.array([.8]), kind=kind)

    def test_unknown_reduction_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unknown reduction'):
            QuantileLoss().measure((np.zeros(2), np.ones(2)), np.zeros(2), np.array([.8]), reduction='invalid')

    def test_interval_helper_remains_two_sided(self):
        coverage, _, (lower, upper), _, _ = is_in_quantile(
            (np.zeros(2), np.ones(2)), np.array([-1., 1.]), np.array([.8]), 'meanstd',
        )
        assert_allclose(lower, norm.ppf(.1))
        assert_allclose(upper, norm.ppf(.9))
        self.assertTrue(coverage.all())


if __name__ == '__main__':
    unittest.main()
