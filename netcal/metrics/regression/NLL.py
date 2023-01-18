# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple
from typing import Union
import numpy as np
from scipy.stats import norm, cauchy

import torch
from torch.distributions import MultivariateNormal

from netcal import meanvar, mv_cauchy_log_density, density_from_cumulative


class NLL(object):
    """
    Negative log likelihood (NLL) for probabilistic regression models.
    If a probabilistic forecaster outputs a probability density function (PDF) :math:`f_Y(y)` targeting the ground-truth
    :math:`y`, the negative log likelihood is defined by

    .. math::
        \\text{NLL} = -\\sum^N_{n=1} \\log\\big(f_Y(y)\\big) ,

    with :math:`N` as the number of samples within the examined data set.

    **Note:** the input field for the standard deviation might also be given as quadratic NumPy arrays of shape
    (n, d, d) with d dimensions. In this case, this method asserts covariance matrices as input
    for each sample and the NLL is calculated for multivariate distributions.
    """

    def measure(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            reduction: str = 'batchmean',
    ) -> Union[float, np.ndarray]:
        """
        Measures the negative log likelihood (NLL) for probabilistic regression models.
        If the input standard deviation is given with shape (n, d, d), this method asserts covariance matrices as input
        for each sample. In this case, the NLL is calculated for multivariate distributions.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                       for each sample. In this case, the NLL is calculated for multivariate distributions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        reduction : str, one of 'none', 'mean', 'batchmean', 'sum' or 'batchsum', default: 'batchmean'
            Specifies the reduction to apply to the output:
            - none : no reduction is performed. Return NLL for each sample and for each dim separately.
            - mean : calculate mean over all samples and all dimensions.
            - batchmean : calculate mean over all samples but for each dim separately.
                          If input has covariance matrices, 'batchmean' is the same as 'mean'.
            - sum : calculate sum over all samples and all dimensions.
            - batchsum : calculate sum over all samples but for each dim separately.
                         If input has covariance matrices, 'batchsum' is the same as 'sum'.

        Returns
        -------
        float or np.ndarray
            Negative Log Likelihood for regression input. See parameter 'reduction' for detailed return type.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'

        if kind == "cumulative":

            assert isinstance(X, (tuple, list)), "If kind=\'cumulative\', assert input X as tuple of sampling points" \
                                                 " and the respective cumulative scores."
            t, cdf = X[0], X[1]  # (t, n, [d])

            # make t, CDF at least 3d, y at least 2d
            t = np.expand_dims(t, axis=2) if t.ndim == 2 else t  # (t, n, d)
            cdf = np.expand_dims(cdf, axis=2) if cdf.ndim == 2 else cdf  # (t, n, d)
            y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

            # get density using the cumulative
            pdf = density_from_cumulative(t, cdf)

            # for the moment, use the nearest point of the curve to determine the actual likelihood
            nearest = np.argmin(
                np.abs(t - y[None, ...]),  # (t, n, d)
                axis=0,
            )  # (n, d)

            # take likelihood om PDF and compute negative log of likelihood
            likelihood = np.squeeze(
                np.take_along_axis(pdf, nearest[None, ...], axis=0),  # (1, n, d)
                axis=0
            )  # (n, d)

            # clip likelihood to avoid 0
            likelihood = np.clip(likelihood, a_min=1e-6, a_max=None)

            nll = -np.log(likelihood)  # (n, d)

        # use Cauchy distribution to measure NLL
        elif kind == "cauchy":

            mode, scale = X[0], X[1]  # (n, [d])

            mode = np.expand_dims(mode, axis=1) if mode.ndim == 1 else mode  # (n, d)
            scale = np.expand_dims(scale, axis=1) if scale.ndim == 1 else scale  # (n, d)
            y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

            # scale is multivariate?
            if scale.ndim == 3:
                with torch.no_grad():
                    nll = -mv_cauchy_log_density(
                        torch.from_numpy(y),
                        torch.from_numpy(mode),
                        torch.from_numpy(scale)
                    ).numpy()  # (n,)

            else:
                nll = -cauchy.logpdf(y, loc=mode, scale=scale)

        # otherwise, use parametric Gaussian expression
        else:
            (mean, var), y, correlations = meanvar(X, y)

            # make mean, var and y at least 2d, for covariances 3d
            mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
            y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

            # in the multivariate case, use a multivariate normal distribution and get logpdf
            if correlations:
                var = np.reshape(var, (-1, 1, 1)) if var.ndim == 1 else var  # (n, d, d)

                # torch's MVN is capable of batched computation in contrast to SciPy's variant
                mvn = MultivariateNormal(loc=torch.from_numpy(mean), covariance_matrix=torch.from_numpy(var))
                nll = -mvn.log_prob(torch.from_numpy(y)).numpy()  # (n,)

            # in the independent case, use a standard normal distribution and get logpdf
            else:
                var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)
                std = np.sqrt(var)
                nll = -norm.logpdf(y, loc=mean, scale=std)

        # perform reduction of (n, d) to single float if requested
        if reduction is None or reduction == 'none':
            return nll

        # 'mean' is mean over all samples and all dimensions
        elif reduction == "mean":
            return float(np.mean(nll))

        # 'batchmean' is mean over all samples but for each dim separately.
        # If correlations == True, 'batchmean' is the same as 'mean'.
        elif reduction == "batchmean":
            return np.mean(nll, axis=0)  # (d,) or scalar

        # 'sum' is sum over all samples and all dimensions
        elif reduction == "sum":
            return float(np.sum(nll))

        # 'batchsum' is sum over all samples but for each dim separately.
        # If correlations == True, 'batchsum' is the same as 'sum'.
        elif reduction == "batchsum":
            return np.sum(nll, axis=0)  # (d,) or scalar

        # unknown reduction method
        else:
            raise RuntimeError("Unknown reduction: \'%s\'" % reduction)
