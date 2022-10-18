# Copyright (C) 2021-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import warnings
from typing import Iterable, Tuple, List
from typing import Union
import numpy as np

from netcal import accepts, meanvar, cumulative_moments
from netcal.metrics.Miscalibration import _Miscalibration


class ENCE(_Miscalibration):
    """
    Expected Normalized Calibration Error (ENCE) for a regression calibration evaluation to test for
    *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    mean :math:`\\mu_Y(X)` and a standard deviation :math:`\\sigma_Y(X)` targeting the ground-truth :math:`y`.
    Similar to the :class:`netcal.metrics.confidence.ECE`, the ENCE applies a binning scheme with :math:`B` bins
    over the predicted standard deviation :math:`\\sigma_Y(X)` and measures the absolute (normalized) difference
    between root mean squared error (RMSE) and root mean variance (RMV) [1]_.
    Thus, the ENCE [1]_ is defined by

    .. math::
        \\text{ENCE} := \\frac{1}{B} \\sum^B_{b=1} \\frac{|RMSE(b) - RMV(b)|}{RMV(b)} ,

    where :math:`RMSE(b)` and :math:`RMV(b)` are the root mean squared error and the root mean variance within
    bin :math:`b`, respectively.

    If multiple dimensions are given, the ENCE is measured for each dimension separately.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the ENCE binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    sample_threshold : int, optional, default: 1
        Bins with an amount of samples below this threshold are not included into the miscalibration metrics.

    References
    ----------
    .. [1] Levi, Dan, et al.:
       "Evaluating and calibrating uncertainty prediction in regression tasks."
       arXiv preprint arXiv:1905.11659 (2019).
       `Get source online <https://arxiv.org/pdf/1905.11659.pdf>`__
    """

    @accepts((int, tuple, list), int)
    def __init__(
            self,
            bins: Union[int, Iterable[int]] = 10,
            sample_threshold: int = 1
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        super().__init__(bins=bins, equal_intervals=True, detection=False, sample_threshold=sample_threshold)

    def measure(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            range_: List[Tuple[float, float]] = None,
    ):
        """
        Measure the ENCE for given input data either as tuple consisting of mean and stddev estimates or as
        NumPy array consisting of a sample distribution.
        If multiple dimensions are given, the ENCE is measured for each dimension separately.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the standard deviation for each dimension separately.
            For example, if input data is given with only a few samples having high standard deviations,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.

        Returns
        -------
        NumPy array of shape (d,)
            NumPy array with the ENCE for each input dimension separately.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'
        if kind == "meanstd":
            (mean, var), y, cov = meanvar(X, y)

            # check if correlated input is given
            if cov:
                raise RuntimeError("UCE is not defined for multivariate data with correlation.")

            mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
            var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)

        # Cauchy distribution has no variance - ENCE is not applicable
        elif kind == "cauchy":

            n_dims = y.shape[1] if y.ndim == 2 else 1

            warnings.warn("ENCE is not applicable for Cauchy distributions.")
            return np.full(shape=(n_dims,), fill_value=float('nan'))

        else:

            # extract sampling points t and cumulative
            # get differences of cumulative and intermediate points of t
            t, cdf = X
            mean, var = cumulative_moments(t, cdf)  # (n, d) and (n, d)

        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y
        std = np.sqrt(var)
        n_samples, n_dims = y.shape

        # squared error
        error = np.square(mean - y)  # (n, d)

        # prepare binning boundaries for regression
        bin_bounds = self._prepare_bins_regression(std, n_dims=n_dims, range_=range_)

        ence = []
        for dim in range(n_dims):

            # perform binning over 1D stddev
            (mv_hist, mse_hist), n_samples, _, _ = self.binning(
                [bin_bounds[dim]],
                std[:, dim],
                var[:, dim],
                error[:, dim]
            )

            rmv_hist = np.sqrt(mv_hist)  # (b,)
            rmse_hist = np.sqrt(mse_hist)  # (b,)

            # ENCE for current dim is equally weighted (but normalized) mean over all bins
            ence.append(
                np.nanmean(
                    np.divide(
                        np.abs(rmv_hist - rmse_hist), rmv_hist,
                        out=np.full_like(rmv_hist, fill_value=float('nan')),
                        where=rmv_hist != 0
                    )
                )
            )

        ence = np.array(ence).squeeze()
        return ence
