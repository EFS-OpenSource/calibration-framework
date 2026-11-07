# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, Tuple
from typing import Union
import numpy as np
from scipy.stats import norm, cauchy
from netcal import squeeze_generic, meanvar


class QuantileLoss(object):
    """
    Pinball aka quantile loss within regression calibration to test for *quantile calibration* of a probabilistic
    regression model. The Pinball loss is an asymmetric loss that measures the quality of the predicted
    quantiles. Given a probabilistic regression model that outputs a probability density function (PDF) :math:`f_Y(y)`
    targeting the ground-truth :math:`y`, we further denote the cumulative as :math:`F_Y(y)` and the (inverse)
    percent point function (PPF) as :math:`F_Y^{-1}(\\tau)` for a certain quantile level :math:`\\tau \\in [0, 1]`.

    The Pinball loss is given by

    .. math::
       L_{\\text{Pin}}(\\tau) =
       \\begin{cases}
            \\big( y-F_Y^{-1}(\\tau) \\big)\\tau \\quad &\\text{if } y \\geq F_Y^{-1}(\\tau)\\\\
            \\big( F_Y^{-1}(\\tau)-y \\big)(1-\\tau) \\quad &\\text{if } y < F_Y^{-1}(\\tau)
       \\end{cases} .
    """

    def measure(
            self, X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            q: Union[float, Iterable[float], np.ndarray],
            *,
            kind: str = 'meanstd',
            reduction: str = 'mean',
    ):
        """
        Measure quantile loss for given input data either as tuple consisting of mean and stddev estimates or as
        NumPy array consisting of a sample distribution. The loss is computed for several quantiles
        given by parameter q.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        q : np.ndarray of shape (q,)
            Quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
        kind : str, either "meanstd", "cauchy" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, empirical quantiles are computed over the stochastic forward passes.
            - cauchy: assert X as a tuple of location and scale arrays of shape (n, [d]).
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions). Quantiles use the nearest CDF value.
        reduction : str, one of 'none', 'mean' or 'sum' default: 'mean'
            Specifies the reduction to apply to the output:
            - none : no reduction is performed. Return quantile loss for each sample, each
                     quantile and for each dim separately.
            - mean : calculate mean over all quantiles, all samples and all dimensions.
            - sum : calculate sum over all quantiles, all samples and all dimensions

        Returns
        -------
        np.ndarray of shape (q, n, d) or float
            Quantile loss for each quantile, sample and dimension, or its reduction.
            See parameter "reduction" for a detailed description of the return type.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], "Unknown input kind: '%s'" % kind
        q = squeeze_generic(np.array(q), axes_to_keep=0).reshape((-1,))

        # Pinball loss uses the one-sided quantile F^{-1}(q), not a boundary
        # of the central prediction interval used by PICP and QCE.
        if kind == 'cumulative':
            t, cdf = X
            t = np.expand_dims(t, axis=2) if t.ndim == 2 else t
            cdf = np.expand_dims(cdf, axis=2) if cdf.ndim == 2 else cdf
            nearest = np.argmin(np.abs(cdf[None, ...] - q[:, None, None, None]), axis=1)
            qbounds = np.take_along_axis(t, nearest, axis=0)

        elif kind == 'cauchy':
            location, scale = X
            if scale.ndim > location.ndim:
                raise RuntimeError("Multivariate Cauchy with correlations is currently not supported.")
            location = location.reshape((-1, 1)) if location.ndim == 1 else location
            scale = scale.reshape((-1, 1)) if scale.ndim == 1 else scale
            qbounds = cauchy.ppf(q[:, None, None], loc=location, scale=scale)

        elif isinstance(X, np.ndarray):
            assert X.ndim >= 2, "Sample distributions must have shape (r, n, [d])."
            X = np.expand_dims(X, axis=2) if X.ndim == 2 else X
            qbounds = np.quantile(X, q, axis=0)

        else:
            mean, var, cov = meanvar(X)
            if cov:
                raise RuntimeError("QuantileLoss is currently not defined for multivariate data with correlation.")
            mean = mean.reshape((-1, 1)) if mean.ndim == 1 else mean
            var = var.reshape((-1, 1)) if var.ndim == 1 else var
            qbounds = norm.ppf(q[:, None, None], loc=mean, scale=np.sqrt(var))

        # make y at least 2d and weight the signed quantile residual
        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)
        error = y[None, ...] - qbounds  # (q, n, d)
        q = q[:, None, None]
        weights = np.where(error >= 0., q, q - 1.)
        # At q=0 or q=1 an unbounded distribution has an infinite residual
        # with zero weight. Preserve this zero loss without hiding NaN inputs.
        loss = np.zeros_like(error)
        np.multiply(weights, error, out=loss, where=(weights != 0.) | np.isnan(error))

        # no reduction is applied
        if reduction is None or reduction == 'none':
            return loss

        # 'mean' is mean over all quantiles and all dimensions
        elif reduction == "mean":
            return float(np.mean(loss))

        # 'sum' is sum over all quantiles and all dimensions
        elif reduction == "sum":
            return float(np.sum(loss))

        # unknown reduction method
        else:
            raise RuntimeError("Unknown reduction: \'%s\'" % reduction)


class PinballLoss(QuantileLoss):
    """ Synonym for Quantile loss. For documentation, see :class:`netcal.metrics.regression.QuantileLoss`. """
    pass
