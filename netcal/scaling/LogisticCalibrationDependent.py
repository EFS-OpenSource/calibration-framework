# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.



import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from netcal import AbstractCalibration, accepts, dimensions


class LogisticCalibrationDependent(AbstractCalibration):
    """
    This calibration method uses multivariate normal distributions to obtain a
    calibration mapping by means of the confidence as well as additional features. This method is originally
    proposed by [1]_. This calibration scheme
    tries to model several dependencies in the variables given by the input ``X``.

    It is necessary to provide all data in input parameter ``X`` as an NumPy array of shape ``(n_samples, n_features)``,
    whereas the confidence must be the first feature given in the input array. The ground-truth samples ``y``
    must be an array of shape ``(n_samples,)`` consisting of binary labels :math:`y \\in \\{0, 1\\}`. Those
    labels indicate if the according sample has matched a ground truth box :math:`\\text{m}=1` or is a false
    prediction :math:`\\text{m}=0`.

    **Mathematical background:** For confidence calibration in classification tasks, a
    confidence mapping :math:`g` is applied on top of a miscalibrated scoring classifier :math:`\\hat{p} = h(x)` to
    deliver a calibrated confidence score :math:`\\hat{q} = g(h(x))`.

    For detection calibration, we can also use the additional box regression output which we denote as
    :math:`\\hat{r} \\in [0, 1]^J` with :math:`J` as the number of dimensions used for the box encoding (e.g.
    :math:`J=4` for x position, y position, width and height).
    Therefore, the calibration map is not only a function of the confidence score, but also of :math:`\\hat{r}`.
    To define a general calibration map for binary problems, we use the logistic function and the combined
    input :math:`s = (\\hat{p}, \\hat{r})` of size K by

    .. math::

       g(s) = \\frac{1}{1 + \\exp(-z(s))} ,

    According to [1]_, we can interpret the logit :math:`z` as the logarithm of the posterior odds

    .. math::

       z(s) = \\log \\frac{f(\\text{m}=1 | s)}{f(\\text{m}=0 | s)} \\approx
       \\log \\frac{f(s | \\text{m}=1)}{f(s | \\text{m}=1)} = \\ell r(s)

    Inserting multivariate normal density distributions into this framework with
    :math:`\\mu^+, \\mu^- \\in \\mathbb{R}^K` and :math:`\\Sigma^+, \\Sigma^- \\in \\mathbb{R}^{K \\times K}`
    as the mean vectors and covariance matrices for :math:`\\text{m}=1` and
    :math:`\\text{m}=0`, respectively, we get a likelihood ratio of

    .. math::

       \\ell r(s) = \\log \\frac{\\Sigma^-}{\\Sigma^+}
       + \\frac{1}{2} (s_-^T \\Sigma_-^{-1}s^-) - (s_+^T \\Sigma_+^{-1}s^+),

    with :math:`s^+ = s - \\mu^+` and :math:`s^- = s - \\mu^-`.

    To keep the restrictions to covariance matrices (symmetric and positive semidefinit), we optimize a decomposed
    matrix V as

    .. math::
       \\Sigma = V^T * V

    instead of estimating :math:`\\Sigma` directly. This guarantees both requirements.

    Parameters
    ----------
    detection : bool, default: True
        IMPORTANT: this parameter is only for compatibility reasons. It MUST be set to True.
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).

    References
    ----------
    .. [1] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    @accepts(bool)
    def __init__(self, detection: bool = True):
        """
        Constructor.

        Parameters
        ----------
        detection : bool, default: True
            IMPORTANT: this parameter is only for compatibility reasons. It MUST be set to True.
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        """

        assert detection, "Classification mode (detection=False) is not supported for class LogisticCalibrationDependent."
        super().__init__(detection=True, independent_probabilities=False)

        self._bias = None
        self._inverse_cov_pos = None
        self._inverse_cov_neg = None
        self._mean_pos = None
        self._mean_neg = None

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()

        self._bias = None
        self._inverse_cov_pos = None
        self._inverse_cov_neg = None
        self._mean_pos = None
        self._mean_neg = None

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticCalibrationDependent':
        """
        Build Logistic Calibration model for multivariate normal distributions.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        LogisticCalibrationDependent
            Instance of class :class:`LogisticCalibrationDependent`.
        """

        X, y = super().fit(X, y)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        solver = 'SLSQP'

        # build data input to compute logit
        data_input = self._build_data_input(X)
        num_features = X.shape[1]

        # scipy optimizer is very fast. SLSQP has shown similar performance compared to momentum optimizer
        theta_0 = np.random.rand(1 + np.power(num_features, 2) * 2 + num_features * 2)
        result = minimize(method=solver,
                          fun=self._loss_function, x0=theta_0,
                          args=(data_input, y, num_features))

        # get result of optimization and extract
        weights = result.x

        # get indices of weights and decompose
        index_1 = 1 + int(np.power(num_features, 2))
        index_2 = index_1 + int(np.power(num_features, 2))
        index_3 = index_2 + num_features

        # covariance matrices are not evaluated directly
        decomposed_inv_cov_pos = np.array(weights[1:index_1]).reshape((num_features, num_features))
        decomposed_inv_cov_neg = np.array(weights[index_1:index_2]).reshape((num_features, num_features))

        # calculate covariance matrices
        self._inverse_cov_pos = np.matmul(decomposed_inv_cov_pos, decomposed_inv_cov_pos.T)
        self._inverse_cov_neg = np.matmul(decomposed_inv_cov_neg, decomposed_inv_cov_neg.T)

        # get mean vectors and bias
        self._mean_pos = np.array(weights[index_2:index_3])
        self._mean_neg = np.array(weights[index_3:])

        self._bias = weights[0]

        return self

    @dimensions((1, 2))
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with calibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).
        """

        X = super().transform(X)

        data_input = self._build_data_input(X)
        logit = self._get_gaussian_ratio(data_input, self._bias, self._inverse_cov_pos, self._inverse_cov_neg,
                                         self._mean_pos, self._mean_neg)

        calibrated = self._sigmoid(logit)
        return calibrated

    @dimensions((1, 2))
    def _build_data_input(self, X: np.ndarray) -> np.ndarray:
        """
        Build data input for Matrix/Platt/Temperature scaling (even for detection mode).

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        np.ndarray
            Data input to calculate logits.
        """

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # if binary, use sigmoid instead of softmax
        if self.num_classes <= 2 or self.independent_probabilities or self.detection:
            if self.detection:
                data_input = np.concatenate((self._inverse_sigmoid(X[:, 0]).reshape(-1, 1), X[:, 1:]), axis=1)
            else:
                data_input = self._inverse_sigmoid(X)
        else:
            data_input = self._inverse_softmax(X)

        return data_input

    @dimensions(2, None, 2, 2, 1, 1)
    def _get_gaussian_ratio(self, data_input: np.ndarray, bias: float,
                            inverse_cov_pos: np.ndarray, inverse_cov_neg: np.ndarray,
                            mean_pos: np.ndarray, mean_neg: np.ndarray) -> np.ndarray:
        """
        Calculate ratio between two gaussian distributions given with parameters.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            Data input to calculate logits.
        bias : float
            Bias for the ratio.
        inverse_cov_pos : np.ndarray, shape=(n_features, n_features)
            Inverse covariance matrix of positive labels.
        inverse_cov_neg : np.ndarray, shape=(n_features, n_features)
            Inverse covariance matrix of negative labels.
        mean_pos : np.ndarray, shape=(n_features,)
            Means of positive labels.
        mean_neg : np.ndarray, shape=(n_features,)
            Means of negative labels.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            Ratio between both distributions.
        """

        # calculate data without means
        difference_pos = data_input - mean_pos
        difference_neg = data_input - mean_neg

        # add a new dimensions. This is necessary for NumPy to distribute dot product
        difference_pos = np.expand_dims(difference_pos, axis=-1)
        difference_neg = np.expand_dims(difference_neg, axis=-1)

        logit = 0.5 * (np.matmul(np.transpose(difference_neg, axes=[0, 2, 1]),
                                 np.matmul(inverse_cov_neg, difference_neg)) -
                       np.matmul(np.transpose(difference_pos, axes=[0, 2, 1]),
                                 np.matmul(inverse_cov_pos, difference_pos))
                       )

        # remove unnecessary dimensions
        logit = self.squeeze_generic(logit, axes_to_keep=0)

        # add log determinant ratio to logit
        logit = bias + logit
        return logit

    def _loss_function(self, weights: np.ndarray, data_input: np.ndarray,
                       y: np.ndarray, num_features: int) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray, shape=(3,) or (2*num_features + 1,)
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
        num_features : int
            Number of features for multivariate distribution.

        Returns
        -------
        float
            NLL-Loss
        """

        # get indices of weights
        index_1 = 1 + int(np.power(num_features, 2))
        index_2 = index_1 + int(np.power(num_features, 2))
        index_3 = index_2 + num_features

        # get weights of decomposed cov matrices V
        decomposed_inv_cov_pos = np.array(weights[1:index_1]).reshape((num_features, num_features))
        decomposed_inv_cov_neg = np.array(weights[index_1:index_2]).reshape((num_features, num_features))

        # calculate covariance matrices
        # COV^-1 = V^-1 * V^(-1,T)
        inv_cov_pos = np.matmul(decomposed_inv_cov_pos, decomposed_inv_cov_pos.T)
        inv_cov_neg = np.matmul(decomposed_inv_cov_neg, decomposed_inv_cov_neg.T)

        mean_pos = np.array(weights[index_2:index_3])
        mean_neg = np.array(weights[index_3:])

        bias = weights[0]

        # get logits by calculating gaussian ratio between both distributions
        logit = self._get_gaussian_ratio(data_input, bias, inv_cov_pos, inv_cov_neg, mean_pos, mean_neg)
        return self._nll_loss(logit=logit, ground_truth=y)
