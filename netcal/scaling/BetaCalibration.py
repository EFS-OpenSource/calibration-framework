# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.optimize import minimize
from netcal import AbstractCalibration, dimensions, accepts


class BetaCalibration(AbstractCalibration):
    """
    On classification, apply the beta calibration method to obtain a calibration mapping. The original method was
    proposed by [1]_.
    For the multiclass case, we extended this method to work with multinomial logistic regression instead of a
    one vs. all calibration mapping.
    On detection mode, this calibration method uses multiple independent Beta distributions to obtain a
    calibration mapping by means of the confidence as well as additional features [2]_. This calibration scheme
    assumes independence between all variables.

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

    If we assume independence of all variables given in :math:`s`, we can use multiple univariate probability
    density distributions to obtain a calibration mapping. Using multiple beta distributions (one for each variable
    and one for each case :math:`\\text{m}=1` and :math:`\\text{m}=0`), the log-likelihood ratio can be
    expressed as

    .. math::
       \\ell r(s) = \\sum^K_{k=1} \\log \\Bigg(\\frac{\\text{Beta}(s_k | \\alpha^+_k, \\beta^+_k)}
                                                     {\\text{Beta}(s_k | \\alpha^-_k, \\beta^-_k)}\\Bigg)

    with the beta distribution :math:`\\text{Beta}(s_k|\\alpha, \\beta)`. The shape parameters
    for :math:`\\text{m}=1` and :math:`\\text{m}=0` are denoted by :math:`\\alpha_k^+, \\beta_k^+` and
    :math:`\\alpha_k^-, \\beta_k^-`, respectively. We can reparametrize this expression by using

    .. math::
       \\ell r(s) = c + \\sum^K_{k=1} a_k \\log(s_k) - b_k \\log(1-s_k) ,

    where the distribution parameters are summarized by :math:`a_k=\\alpha_k^+-\\alpha_k^-`,
    :math:`b_k=\\beta_k^--\\beta_k^+` and
    :math:`c=\\sum_k \\log B(\\alpha_k^-, \\beta_k^-) - \\log B(\\alpha_k^+, \\beta_k^+)`.
    We utilize standard optimization methods to determine the calibration mapping :math:`g(s)`.

    Parameters
    ----------
    auto_select : bool, optional, default: False
        Auto selection of best combination on detection mode.
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    .. [1] Kull, Meelis, Telmo Silva Filho, and Peter Flach:
       "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
       Artificial Intelligence and Statistics, PMLR 54:623-631, 2017
       `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`_
    .. [2] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    @accepts(bool, bool, bool)
    def __init__(self, auto_select: bool = False, detection: bool = False, independent_probabilities: bool = False):
        """
        Constructor

        Parameters
        ----------
        auto_select : bool, optional, default: False
            Auto selection of best combination on detection mode.
        detection : bool, default: False
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """
        super().__init__(detection=detection, independent_probabilities=independent_probabilities)

        self.auto_select = auto_select
        self._weights = None

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        self._weights = None

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BetaCalibration':
        """
        Build Beta Calibration model.

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
        BetaCalibration
            Instance of class :class:`BetaCalibration`.
        """

        X, y = super().fit(X, y)
        solver = 'SLSQP'

        # number of given features is 1 on classification mode (confidence)
        num_features = 1
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # on detection mode, check if additional features are given
        if self.detection:
            num_features = X.shape[1]

        # prepare data for beta calibration
        data_input = self._get_data_input(X)

        # detection case: logistic regression with multiple features and bias
        if self.detection:
            num_weights = num_features * 2 + 1
            normalizer = 1. / float(num_features * 2)

        # binary classification case: logistic regression with two features and bias
        elif self._is_binary_classification():
            num_weights = 3
            normalizer = 0.5

        # multiclass classification case: multinomial logistic regression with two features and bias for each class
        else:
            num_weights = self.num_classes * 3
            normalizer = 0.5

            # convert ground truth to one hot if not binary
            y = self._get_one_hot_encoded_labels(y, self.num_classes)

        # initial parameter set
        theta_0 = np.ones(num_weights) * normalizer
        result = minimize(method=solver,
                          fun=self._loss_function, x0=theta_0,
                          args=(data_input, y))

        self._weights = result.x

        # get all weights masked whose values are negative and repeat optimization in that case
        masked_weights = self._mask_negative_weights(self._weights)
        if len(masked_weights):

            # rerun minimization routine
            theta_0 = np.ones(num_weights) * normalizer
            theta_0[masked_weights] = 0.0
            result = minimize(method=solver,
                              fun=self._loss_function, x0=theta_0,
                              args=(data_input, y, masked_weights))

            self._weights = result.x

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

        # prepare data for beta calibration
        data_input = self._get_data_input(X)
        logit = self._get_logit(self._weights, data_input)

        # compute logistic fit
        calibrated = self._sigmoid(logit)[:, 0] if self.detection or self._is_binary_classification() else self._softmax(logit)
        return self.squeeze_generic(calibrated, axes_to_keep=0)

    def _mask_negative_weights(self, weights: np.ndarray) -> list:
        """
        Seek for all relevant weights whose values are negative. Mask those values with optimization constraints
        in the interval [0, 0].

        Parameters
        ----------
        weights : np.ndarray
            Flattened weights array

        Returns
        -------
        list
            Indices of masked values.
        """

        num_weights = len(weights)

        # --------------------------------------------------------------------------------
        # check calculated weights
        # if either param a or param b < 0, the second distribution's parameter is fixed to zero
        # the logistic fit is repeated afterwards with remaining distribution

        # first step: extract weights without bias and get all weights below 0

        # on detection or binary classification, only face the confidence weights
        if self.detection or self._is_binary_classification():
            weights = weights[1:3]

            # on multiclass classification, face all weights (without biases)
        else:
            weights = weights[self.num_classes:]

        # now check if negative entries are present
        masked_weights = np.where((weights.flatten() < 0))[0]

        # weights below 0 found?
        # we need to make sure that not both weights are masked out for a single class
        if len(masked_weights):

            # on detection or binary classification, only keep the confidence dimension monotonically increasing
            # this is equivalent to the first dimension (weights 0 and 1)
            if self.detection or self._is_binary_classification():
                if len(masked_weights) == 2:
                    masked_weights = np.array([masked_weights[0]])

            # same on multiclass classification but for each class
            else:
                for cls in range(self.num_classes):
                    index_a = 2 * cls
                    index_b = 2 * cls + 1

                    # remove index of the second weight for this certain class from the masked weights array
                    if index_a in masked_weights and index_b in masked_weights:
                        masked_weights = np.delete(masked_weights, np.argwhere(masked_weights == index_b))

            # increase indices for optimization routine due to bias values
            masked_weights += 1 if self.detection or self._is_binary_classification() else self.num_classes

        return masked_weights

    def _get_data_input(self, X: np.ndarray) -> np.ndarray:
        """
        Build data input for beta calibration logistic regression.

        Note for multiclass classification:
        For NumPy's matmul function it is mandatory to provide the class dimension in the first
        axis in order to use its broadcasting mechanism that allows multiple dot-products in a single run.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).

        Returns
        -------
        np.ndarray, shape=(n_samples, 2*n_features) on detection or shape=([n_classes], n_samples, 2) on classification
            NumPy array with prepared input data for beta calibration.
        """

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # build 2-D array with [log(x), -log(1-x)]
        # convert confidences array and clip values to open interval (0, 1)

        # on detection, create array of shape (n_samples, 2*n_features)
        if self.detection:
            features = []
            for i in range(X.shape[1]):
                features.append(X[:, i])
                features.append(1. - X[:, i])

            data_input = np.stack(features, axis=1)
            data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)
            data_input = np.log(data_input)
            data_input[:, 1::2] *= -1

        # on classification, create an array
        # - binary classification: shape (n_samples, 2)
        # - multiclass classification: shape (n_classes, n_samples, 2)
        else:
            features = []
            for i in range(X.shape[1]):
                features.append(np.stack([X[:, i], 1. - X[:, i]], axis=1))

            data_input = features[0] if self._is_binary_classification() else np.stack(features, axis=0)
            data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)
            data_input = np.log(data_input)
            data_input[..., 1] *= -1

        return data_input

    def _get_logit(self, weights: np.ndarray, data_input: np.ndarray) -> np.ndarray:
        """
        Calculate logit depending on the provided weights and prepared data input for beta calibration.

        Parameters
        ----------
        weights : np.ndarray, shape=(3,) for binary classification or shape=(3*n_classes,) for multiclass
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=([n_classes], n_samples, 2)
            NumPy 2-D array with data input.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            Logit for beta calibration.
        """

        if self.detection or self._is_binary_classification():
            bias = weights[0]
            weights = np.array(weights[1:]).reshape(-1, 1)

            # compute logits
            logit = np.matmul(data_input, weights) + bias

        else:

            # get number of weights and biases according to number of classes
            bias = weights[:self.num_classes]
            weights = np.reshape(weights[self.num_classes:], (-1, 2, 1))

            # use broadcast mechanism of NumPy: if 3 dimensions are provided, treat first dimension
            # as a stack of matrices -> this speeds up the calculation
            result = np.matmul(data_input, weights)

            # as a result, we obtain an array of shape (n_classes, n_samples, 1)
            # remove last dim and swap axes
            logit = np.swapaxes(np.squeeze(result), 0, 1) + bias

        return logit

    def _loss_function(self, weights: np.ndarray, data_input: np.ndarray, y: np.ndarray, masked: list = None) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray, shape=(2*n_features+1) for detection, shape=(3,) for binary classification or
                 shape=(3*n_classes,) for multiclass
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=(n_samples, 2*n_features) for detection or shape=([n_classes], n_samples, 2)
                     for classification
            NumPy array with data input.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        float
            NLL-Loss
        """

        if masked is not None:
            weights[masked] = 0.

        logit = self._get_logit(weights, data_input)
        return self._nll_loss(logit, y)
