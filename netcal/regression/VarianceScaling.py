# Copyright (C) 2021-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from netcal import AbstractCalibration, meanvar


class VarianceScaling(AbstractCalibration):
    """
    Variance recalibration using maximum likelihood estimation for multiple independent dimensions (optionally).
    Rescales the input standard deviation by a scalar parameter to achieve *variance calibration* [1]_, [2]_.
    This method uses the negative log likelihood as the loss function to optimize the scalar scaling parameter.
    The distributional mean is fixed during optimization.

    On the one hand, this method accepts as input X either a tuple X = (mean, stddev) using two NumPy arrays of
    shape N with N number of samples that express the estimated mean and standard deviation of a probabilistic
    forecaster. On the other hand, a NumPy array of shape (R, N) is also accepted where R denotes the number of
    probabilistic forecasts. For example, if probabilistic outputs are obtained by Monte-Carlo sampling using N samples
    and R stochastic forward passes, it is possible to pass all outputs to the calibration function in a single
    NumPy array.

    This method is capable of multiple independent data dimensions where separate calibration models are fitted for
    each data dimension. This method outputs the recalibrated standard deviation (stddev) for each dimension D.

    **Mathematical background:** In [1]_ and [2]_, regression calibration is defined in terms of *variance calibration*.
    A probabilistic forecaster :math:`h(X)` outputs for any input :math:`X \\in \\mathbb{R}` a mean :math:`\\mu_Y(X)`
    and a variance :math:`\\sigma_Y^2(X)` for the target domain :math:`Y \\in \\mathcal{Y} = \\mathbb{R}`.
    Using this notation, *variance calibration* [1]_, [2]_ is defined as

    .. math::
        \\mathbb{E}_{X,Y}\\Big[(\\mu_Y(X) - Y)^2 | \\sigma^2_Y(X) = \\sigma \\Big] = \\sigma^2,
        \\quad \\forall \\sigma \\in \\mathbb{R}_{>0},

    In other words, the estimated variance should match the observed variance given a certain variance level.
    For example, if a forecaster outputs 100 predictions with a variance of :math:`\\sigma^2=2`, we would also expect
    a variance (mean squared error) of 2.
    Further definitions for regression calibration are *quantile calibration* and *distribution calibration*.

    To achieve *variance calibration*, the Variance Scaling methods applies a temperature scaling on the
    input variance by a single scalar parameter :math:`\\theta`. The methods uses the negative log likelihood as the
    loss for the scalar. Since we are working with Gaussians, the loss is given by

    .. math::
        \\mathcal{L}(\\theta) &= -\\sum^N_{n=1} \\frac{1}{\\sqrt{2\\pi} \\theta \\cdot \\sigma_Y(x_n)}
        \\exp\\Bigg( \\frac{y_n - \\mu_Y(x_n)}{2 (\\theta \\cdot \\sigma_Y(x_n))^2} \\Bigg) \\\\
        &\\propto -N \\log(\\theta) - \\frac{1}{2\\theta^2} \\sum^N_{n=1} \\sigma_Y^{-2}(x_n) ,
        \\big(y_n - \\mu_Y(x_n)\\big)^2

    which is to be minimized.
    Thus, we seek to get the minimum of the optimization objective which can be analytically determined in this case,
    setting its first derivative to 0 by

    .. math::
        &\\frac{\\partial \\mathcal{L}(\\theta)}{\\partial \\theta} = 0\\\\
        \\leftrightarrow \\quad & -N \\theta^2 \\sum^N_{n=1} \\sigma_Y^{-2}(x_n) \\big(y_n - \\mu_Y(x_n) \\big)^2 \\\\
        \\leftrightarrow \\quad & \\theta \\pm \\sqrt{\\frac{1}{N} \\sum^N_{n=1} \\sigma_Y^{-2}(x_n) \\big(y_n - \\mu_Y(x_n) \\big)^2} .


    References
    ----------
    .. [1] Levi, Dan, et al.:
       "Evaluating and calibrating uncertainty prediction in regression tasks."
       arXiv preprint arXiv:1905.11659 (2019).
       `Get source online <https://arxiv.org/pdf/1905.11659.pdf>`__

    .. [2] Laves, Max-Heinrich, et al.:
       "Well-calibrated regression uncertainty in medical imaging with deep learning."
       Medical Imaging with Deep Learning. PMLR, 2020.
       `Get source online <http://proceedings.mlr.press/v121/laves20a/laves20a.pdf>`__
    """

    def __init__(self):
        """ Constructor. """

        super().__init__(detection=False, independent_probabilities=False)
        self._weight = None

    def clear(self):
        """ Clear model parameters. """

        self._weight = None

    def fit(
            self,
            X: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            tensorboard: Optional[SummaryWriter] = None
    ) -> 'VarianceScaling':
        """
        Fit a variance scaling calibration method to the provided data. If multiple dimensions are provided,
        multiple independent recalibration models are fitted for each dimension.

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
        VarianceScaling
            Instance of class :class:`netcal.regression.VarianceScaling`.
        """

        (mean, var), y, cov = meanvar(X, y)

        # covariance rescaling is currently not supported
        if cov:
            raise RuntimeError("VarianceScaling: covariance rescaling is currently not supported.")

        mean = np.expand_dims(mean, axis=1) if mean.ndim == 1 else mean  # (n, d)
        var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)
        y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

        # used closed-form solution to obtain the optimal rescaling parameter
        self._weight = np.sqrt(np.mean(np.square(y - mean) / var, axis=0, keepdims=True))  # (1, d)

        # log number of training samples and scalar weight to tensorboard
        if tensorboard is not None:
            for dim, weight in enumerate(self._weight[0]):
                tensorboard.add_scalar("nllvariance/train/scale/dim%02d" % dim, weight)

            tensorboard.add_scalar("nllvariance/train/n_samples", y.shape[0])

        return self

    def transform(self, X: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Transform uncalibrated distributional estimates (mean and stddev or stochastic samples) to calibrated ones
        by applying variance recalibration. Returns the calibrated standard deviation.

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

        Returns
        -------
        np.ndarray of shape (n, d)
            Recalibrated standard deviation for each sample in each dimension.
        """

        # check if method has already been trained
        if self._weight is None:
            raise RuntimeError("VarianceScaling: call \'fit()\' method first before transform.")

        # get mean and variance from input
        mean, var, cov = meanvar(X)

        # covariance rescaling is currently not supported
        if cov:
            raise RuntimeError("VarianceScaling: covariance rescaling is currently not supported.")

        # make variance at least 2d and perform rescaling of stddev
        var = np.expand_dims(var, axis=1) if var.ndim == 1 else var  # (n, d)
        calibrated_scale = np.sqrt(var) * self._weight  # (n, d)

        return calibrated_scale

    def __repr__(self):
        """ Returns a string representation of the calibration method with the most important parameters. """

        if self._weight is None:
            return "VarianceScaling(weight=None)"
        else:
            return "VarianceScaling(weight=%.4f)" % self._weight
