# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.optimize import minimize
from netcal import AbstractCalibration, accepts, dimensions


class LogisticCalibration(AbstractCalibration):
    """
    On classification, apply the logistic calibration method aka Platt scaling to obtain a
    calibration mapping. This method is originally proposed by [1]_.
    For the multiclass case, we use the Vector scaling proposed in [2]_.
    On detection mode, this calibration method uses multiple independent normal distributions to obtain a
    calibration mapping by means of the confidence as well as additional features [3]_. This calibration scheme
    assumes independence between all variables.

    On detection, it is necessary to provide all data in input parameter ``X`` as an NumPy array
    of shape ``(n_samples, n_features)``,
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
    density distributions with the same variance to obtain a calibration mapping. Using this formulation, we can
    simply extend the scaling factor (from classification logistic calibration) to a scaling
    vector :math:`w \\in \\mathbb{R}^K`.
    However, instead of using the uncalibrated confidence estimate :math:`\\hat{p}`, we use the logit of the
    network as part of :math:`s` to be conform with the original formulation in [1]_ and [2]_. Thus,
    the log-likelihood ratio can be expressed as

    .. math::
       \\ell r(s) = s^T w + c,

    with bias :math:`c \\in \\mathbb{R}`.
    We utilize standard optimization methods to determine the calibration mapping :math:`g(s)`.

    Parameters
    ----------
    temperature_only : bool, default: False
        If True, use Temperature Scaling instead of Platt/Vector Scaling.
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
    .. [1] Platt, John:
       "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods."
       Advances in large margin classifiers 10.3: 61-74, 1999
       `Get source online <https://www.researchgate.net/profile/John_Platt/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000.pdf>`_

    .. [2] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
       "On Calibration of Modern Neural Networks."
       Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
       `Get source online <https://arxiv.org/abs/1706.04599>`_

    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    @accepts(bool, bool, bool)
    def __init__(self, temperature_only: bool = False, detection: bool = False, independent_probabilities: bool = False):
        """
        Constructor

        Parameters
        ----------
        temperature_only : bool, default: False
            If True, use Temperature Scaling instead of Platt/Vector Scaling.
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

        self.temperature_only = temperature_only

        self._weights = None
        self._num_combination = 0

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()

        self._weights = None
        self._num_combination = 0

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticCalibration':
        """
        Build logitic calibration model.

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
        LogisticCalibration
            Instance of class :class:`LogisticCalibration`.
        """

        X, y = super().fit(X, y)
        data_input = self._build_data_input(X)

        # convert ground truth to one hot if not binary and not detection mode
        if self.num_classes > 2 and not self.detection:
            y = self._get_one_hot_encoded_labels(y, self.num_classes)

        # initialize weights
        # on temperature scaling, number of weights is single scalar
        if self.temperature_only:
            initial_weights = np.array(1.0)

        else:
            # detection mode: number of weights is number of features + bias
            # initialize weights equally weighted and bias slightly greater than 0
            if self.detection:
                num_weights = X.shape[1] + 1
                initial_weights = np.ones(num_weights) / float(num_weights-1)
                initial_weights[0] = self.epsilon

            # binary classification: use one weight and one bias
            elif self._is_binary_classification():
                num_weights = 2
                initial_weights = np.ones(num_weights)
                initial_weights[0] = self.epsilon

            # multiclass classification: use one weight and one bias for each class separately
            else:
                num_weights = 2*self.num_classes
                initial_weights = np.ones(num_weights)
                initial_weights[:self.num_classes] = self.epsilon

        # -----------------------------

        # invoke SciPy's optimization function
        result = minimize(fun=self._loss_function, x0=initial_weights, args=(data_input, y))

        # get weights after optimization
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

        # get data input and logits
        data_input = self._build_data_input(X)
        logit = self._calculate_logit(data_input, self._weights)

        if self.detection or self._is_binary_classification():
            calibrated = self._sigmoid(logit)
        else:
            calibrated = self._softmax(logit)

        return self.squeeze_generic(calibrated, axes_to_keep=0)

    def _build_data_input(self, X: np.ndarray) -> np.ndarray:
        """
        Build data input for Vector/Platt/Temperature scaling (even for detection mode).

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

        # on detection mode, convert confidence to sigmoid and append the remaining features
        if self.detection:
            data_input = np.concatenate((self._inverse_sigmoid(X[:, 0]).reshape(-1, 1), X[:, 1:]), axis=1)

        # on binary classification, simply convert the confidences to logits
        elif self._is_binary_classification():
            data_input = self._inverse_sigmoid(X)

        # on multiclass classification, use inverse softmax instead
        else:
            data_input = self._inverse_softmax(X)

        return data_input

    def _calculate_logit(self, data_input: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calculate logit by given data input and weights. The weights are decomposed automatically.

        Parameters
        ----------
        data_input : np.ndarray
            Data input to calculate logits.
        weights : np.ndarray
            Weights for scaling and shifting data input to calculate logits.

        Returns
        -------
        np.ndarray
            Scaled and shifted logits.
        """

        # only one weight is equal to temperature scaling - set bias to 0
        if len(weights) == 1:
            bias = 0.0
            logit = (data_input * weights) + bias

        # more than one weight is vector/platt scaling or detection calibration
        else:

            # on detection, perform a dot product operation
            # this is equivalent to (column-wise) multiplication on binary case
            if self.detection or self._is_binary_classification():
                bias = weights[0]
                weights = np.array(weights[1:]).reshape(-1, 1)
                logit = np.matmul(data_input, weights) + bias

            # on binary classification, extract bias and weights and perform (column-wise) multiplication
            # this is equivalent to a weight matrix restricted to a diagonal
            else:
                bias = weights[:self.num_classes]
                weights = np.array(weights[self.num_classes:])
                logit = np.multiply(data_input, weights) + bias

        return logit

    @dimensions(1, (1, 2), (1, 2))
    def _loss_function(self, weights: np.ndarray, data_input: np.ndarray, y: np.ndarray) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray, shape=(3,) or (2*num_features + 1,)
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=(n_samples, 2)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        float
            NLL-Loss
        """

        logit = self._calculate_logit(data_input, weights)
        return self._nll_loss(logit, y)
