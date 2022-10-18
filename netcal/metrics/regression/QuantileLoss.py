# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, Tuple
from typing import Union
import numpy as np
from netcal import accepts, squeeze_generic, is_in_quantile


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
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        reduction : str, one of 'none', 'mean' or 'sum' default: 'mean'
            Specifies the reduction to apply to the output:
            - none : no reduction is performed. Return quantile loss for each sample, each
                     quantile and for each dim separately.
            - mean : calculate mean over all quantiles, all samples and all dimensions.
            - sum : calculate sum over all quantiles, all samples and all dimensions

        Returns
        -------
        np.ndarray of shape (q, d)
            Quantile loss for quantile q in dimension d over all samples in input X.
            See parameter "reduction" for a detailed description of the return type.
        """

        # get quantile boundaries
        in_quantile, qbounds, _, _, _ = is_in_quantile(X, y, q, kind)  # (q, n, [d]), (q, n, d)

        # qbounds is None if input is given with covariance matrices
        if qbounds is None:
            raise RuntimeError("QuantileLoss is currently not defined for multivariate data with correlation.")

        # make y at least 2d
        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

        # broadcast q array to all dimensions
        n_samples, n_dims = y.shape

        # standard case: use L1 (absolute) distance between ground-truth and quantile boundary
        q = np.broadcast_to(q[:, None, None], (q.shape[0], n_samples, n_dims))  # (q, n, d)
        distance = np.abs(y[None, :] - qbounds)  # (q, n, d)

        # compute weights for pinball loss
        weights = np.where(in_quantile, 1.-q, q)  # (q, n, [d])

        # finally, compute loss
        loss = distance * weights  # (q, n, [d])

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
