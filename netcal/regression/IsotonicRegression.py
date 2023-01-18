# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, List, Tuple, Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression as sklearn_iso
from scipy.stats import norm
from matplotlib import pyplot as plt

from netcal import AbstractCalibration, cumulative, meanvar, density_from_cumulative


class IsotonicRegression(AbstractCalibration):
    """
    Isotonic regression calibration for probabilistic regression models with multiple independent
    output dimensions (optionally).
    Isotonic regression is a piecewise constant, monotonically increasing mapping function used to recalibrate
    the estimated cumulative density function (CDF) of a probabilistic forecaster [1]_. The goal of regression
    calibration using Isotonic regression is to achieve quantile calibration.

    On the one hand, this method accepts as input X either a tuple X = (mean, stddev) using two NumPy arrays of
    shape N with N number of samples that express the estimated mean and standard deviation of a probabilistic
    forecaster. On the other hand, a NumPy array of shape (R, N) is also accepted where R denotes the number of
    probabilistic forecasts. For example, if probabilistic outputs are obtained by Monte-Carlo sampling using N samples
    and R stochastic forward passes, it is possible to pass all outputs to the calibration function in a single
    NumPy array.

    This method is capable of multiple independent data dimensions where separate calibration models are fitted for
    each data dimension. This method outputs a tuple consisting of three NumPy arrays:

    - 1st array: T points where the density/cumulative distribution functions are defined, shape: (T, N, D)
    - 2nd array: calibrated probability density function, shape: (T, N, D)
    - 3rd array: calibrated cumulative density function, shape: (T, N, D)

    **Mathematical background:** In [1]_, regression calibration is defined in terms of *quantile calibration*.
    A probabilistic forecaster :math:`h(X)` outputs for any input :math:`X \\in \\mathbb{R}` a probability density
    distribution :math:`f_Y(y)` for the target domain :math:`Y \\in \\mathcal{Y} = \\mathbb{R}`. The according
    cumulative density function (CDF) is denoted as :math:`F_Y(y)`, the respective (inverse) quantile function
    :math:`F^{-1}_Y(\\tau)` for a certain confidence level :math:`\\tau \\in [0, 1]`. The quantile function denotes
    the quantile boundaries in :math:`\\mathcal{Y}` given a certain confidence level :math:`\\tau`.
    Using this notation, *quantile calibration* [1]_ is defined as

    .. math::
        \\mathbb{P}(Y \\leq F^{-1}_Y(\\tau)) = \\tau, \\quad \\forall \\tau \\in [0, 1] ,

    which is equivalent to

    .. math::
        \\mathbb{P}(F^{-1}_Y(\\tau_1) \\leq Y \\leq F^{-1}_Y(\\tau_2)) = \\tau_2 - \\tau_1,
        \\quad \\forall \\tau_1, \\tau_2 \\in [0, 1] .

    In other words, the estimated quantiles should match the observed quantiles. For example, if we inspect the 90%
    quantiles of a forecaster over multiple samples, we would expect that 90% of all ground-truth estimates fall into
    these quantiles.

    The Isotonic Regression consumes the input cumulative distribution function (CDF) and compares it with the
    empirical data CDF. With this comparison, it is possible to map the uncalibrated CDF estimates to calibrated
    ones using a monotonically increasing step function.

    References
    ----------
    .. [1] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon:
       "Accurate uncertainties for deep learning using calibrated regression."
       International Conference on Machine Learning. PMLR, 2018.
       `Get source online <http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf>`__
    """

    def __init__(self):
        """ Constructor. """

        super().__init__(detection=False, independent_probabilities=False)
        self._iso = None

    def clear(self):
        """ Clear model parameters. """

        self._iso = None

    def fit(
            self,
            X: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            tensorboard: Optional[SummaryWriter] = None
    ) -> 'IsotonicRegression':
        """
        Fit a isotonic regression calibration method to the provided data. If multiple dimensions are provided,
        multiple independent regression models are fitted for each dimension.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or Tuple of two np.ndarray, each of shape (n, [d])
            Input data for calibration regression obtained by a model that performs inference with uncertainty.
            Depending on the input format, this method handles the input differently:
            If X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
            first array as mean and the second one as the according stddev predictions with d dimensions (optionally).
            If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
            inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
            case, the mean and stddev is computed automatically.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        tensorboard: torch.utils.tensorboard.SummaryWriter, optional, default: None
            Instance of type "SummaryWriter" to log training statistics.

        Returns
        -------
        IsotonicRegression
            Instance of class :class:`netcal.regression.IsotonicRegression`.
        """

        # add a preceeding (sample) dimension to y as "cumulative" expects multiple points on the
        # cumulative in the first dimension
        y = np.expand_dims(y, axis=0)  # (1, n, [d])
        cdf_predicted, y = cumulative(X, y)  # (1, n, d)

        # squeeze out the first data dim
        cdf_predicted = np.squeeze(cdf_predicted, axis=0)  # (n, d)
        n_samples, n_dims = cdf_predicted.shape

        # fit the isotonic regression for each dimension independently
        self._iso = []
        for dim in range(n_dims):
            iso = sklearn_iso(out_of_bounds='clip')

            # get the empirical cumulative for the current dim
            # use sort algorithm to determine how many samples are less or equal given a certain sample
            # simply use linspace to give the fraction of the relative amount of less or equal
            # samples in a sorted array
            cdf = np.sort(cdf_predicted[:, dim])  # (n,)
            cdf_empirical = np.linspace(1./n_samples, 1., n_samples)  # (n,)

            # this method might introduce a small error when consecutive samples have the same value
            # in this case, identify equal samples and reassign the new cumulative score in reversed order
            equal, = np.where(cdf[:-1] == cdf[1:])  # (n-1, )
            for idx in reversed(equal):
                cdf_empirical[idx] = cdf_empirical[idx+1]

            iso.fit(cdf, cdf_empirical)
            self._iso.append(iso)

            # draw calibration curve for current dimension and add to SummaryWriter
            if tensorboard is not None:

                # draw matplotlib figure
                fig, ax = plt.subplots()
                ax.plot(iso.X_thresholds_, iso.y_thresholds_)
                ax.grid(True)
                ax.set_xlim([0., 1.])
                ax.set_ylim([0., 1.])
                ax.set_title("Isotonic recalibration curve dim %02d" % dim)

                # add matplotlib figure to SummaryWriter
                tensorboard.add_figure("isotonic/train/curve/dim%02d" % dim, fig, close=True)

        # add number of training samples to tensorboard
        if tensorboard is not None:
            tensorboard.add_scalar("isotonic/train/n_samples", n_samples)

        return self

    def transform(
            self,
            X: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray],
            t: Union[int, np.ndarray] = 512
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform uncalibrated distributional estimates (mean and stddev or stochastic samples) to calibrated ones
        by applying isotonic regression calibration of quantiles. Use the parameter "t" to control the base points
        for the returned density/cumulative functions.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or Tuple of two np.ndarray, each of shape (n, [d])
            Input data for calibration regression obtained by a model that performs inference with uncertainty.
            Depending on the input format, this method handles the input differently:
            If X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
            first array as mean and the second one as the according stddev predictions with d dimensions (optionally).
            If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
            inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
            case, the mean and stddev is computed automatically.
        t : int or np.ndarray of shape (t, [n, [d]])
            Parameter to control the output sample distribution. If integer, the given amount of
            sample points for each input X is created. If np.ndarray, use this as distribution values for each
            sample and each dim in X at location t (either for all samples equally with shape (t,) or for each sample
            individually with shape (t, n, [d])).

        Returns
        -------
        tuple of 3 np.ndarray, each of shape (t, n, d)
            First array holds the points where the density and cumulative functions are defined, shape: (t, n ,d).
            Second array is the recalibrated probability density function (PDF), shape: (t, n, d).
            Third array is the recalibrated cumulative density function (CDF), shape: (t, n, d).
            Note that the data dimension d is always present (also for d=1).
        """

        # check if method has already been trained
        if self._iso is None:
            raise RuntimeError("IsotonicRegression: call \'fit()\' method first before transform.")

        # get number of dimension used for training
        n_dims = len(self._iso)
        mean, variance, cov = meanvar(X)

        # catch covariance input
        if cov:
            raise RuntimeError("IsotonicRegression: covariance input is currently not supported for quantile computation.")

        # make mean and variance at least 2d
        mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
        variance = np.expand_dims(variance, axis=1) if variance.ndim == 1 else variance  # (n, d)

        # initialize sampling points to define the recalibrated PDF/CDF
        n_samples = mean.shape[0]
        if isinstance(t, int):

            target_quantile = 1e-7
            stddev = np.sqrt(variance)

            # initialize empty boundaries for cumulative x-scores
            lb_cdf = np.zeros(len(self._iso))  # (d,)
            ub_cdf = np.zeros(len(self._iso))  # (d,)

            # iterate over all dimensions and examine the according isotonic regression model
            for dim, iso in enumerate(self._iso):

                # get the indices of the isotonic calibration curve where the desired relevance level is reached
                lb_idx = np.argmax(iso.y_thresholds_ >= target_quantile)
                ub_idx = len(iso.y_thresholds_) - np.argmax(np.flip(iso.y_thresholds_) < (1-target_quantile)) - 1

                # get the according x-scores of the (recalibrated) cumulative
                lb_cdf[dim] = iso.X_thresholds_[lb_idx]
                ub_cdf[dim] = iso.X_thresholds_[ub_idx]

            # finally, convert these scores back to the uncalibrated cumulative to obtain the sampling points
            # with boundaries for the uncalibrated cumulative
            lb = norm.ppf(lb_cdf[None, :], loc=mean, scale=stddev)  # (n, d)
            ub = norm.ppf(ub_cdf[None, :], loc=mean, scale=stddev)  # (n, d)

            sampling_points = np.linspace(lb, ub, t, axis=0)  # (t, n, d)

        # if t is np.ndarray, use this points as the base for the calibrated
        # output distribution
        elif isinstance(t, np.ndarray):

            # distribute 1d/2d array
            if t.ndim == 1:
                sampling_points = np.reshape(t, (-1, 1, 1))  # (t, 1, 1)
                sampling_points = np.broadcast_to(sampling_points, (t.shape[0], n_samples, n_dims))  # (t, n, d)

            elif t.ndim == 2:
                sampling_points = np.expand_dims(t, axis=2)  # (t, n, 1)
                sampling_points = np.broadcast_to(sampling_points, (t.shape[0], n_samples, n_dims))  # (t, n, d)

            elif t.ndim == 3:
                sampling_points = t  # (t, n, d)

            else:
                raise RuntimeError("Invalid shape for parameter \'t\'.")

            # guarantee monotonically increasing sampling points (required for PDF diff)
            sampling_points = np.sort(sampling_points, axis=0)  # (t, n, d)
        else:
            raise AttributeError("Parameter \'t\' must be either of type int or np.ndarray.")

        t = sampling_points.shape[0]
        cdf, _ = cumulative(X, sampling_points)  # (t, n, d)

        # iterate over dimensions and perform recalibration of the cumulative
        for dim in range(n_dims):
            cdf[..., dim] = np.clip(
                self._iso[dim].transform(cdf[..., dim].flatten()),  # (t*n,)
                np.finfo(np.float32).eps,
                1.-np.finfo(np.float32).eps
            ).reshape((t, n_samples))  # (t, n)

        # get density function using the cumulative
        pdf = density_from_cumulative(sampling_points, cdf)

        return sampling_points, pdf, cdf

    def __repr__(self):
        """ Returns a string representation of the calibration method with the most important parameters. """

        if self._iso is None:
            return "IsotonicRegression(fitted=False)"
        else:
            return "IsotonicRegression(fitted=True)"
