# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy import optimize
from netcal import AbstractCalibration, dimensions, accepts


class LogisticCalibration(AbstractCalibration):
    """
    Perform Logistic Calibration scaling/Platt scaling to logits of NN.
    The calibrated probability :math:`\\hat{q}` is computed by

    .. math::

       \\hat{q} = \\sigma_{\\text{SM}} (Wz +b)

    with :math:`\\sigma_{\\text{SM}}` as the softmax operator (or the sigmoid alternatively),
    :math:`z` as the logits, :math:`W` as the weight matrix and :math:`b` as the bias estimated by logistic regression.
    This leds to calibrated confidence estimates. If 'temperature_only' is true, recover temperature scaling
    (no bias and one single parameter instead of a weight matrix used for all classes).

    Parameters
    ----------
    temperature_only: bool, default: False
        If True, use Temperature Scaling. If False, use standard Logistic Calibration/Platt Scaling.
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Platt, John:
    "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods."
    Advances in large margin classifiers, 10(3): 61–74, 1999.
    `Get source online <https://www.researchgate.net/profile/John_Platt/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000.pdf>`_


    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
    "On Calibration of Modern Neural Networks."
    arXiv (abs/1706.04599), 2017.
    `Get source online <https://arxiv.org/abs/1706.04599>`_
    """

    @accepts(bool, bool)
    def __init__(self, temperature_only: bool = False, independent_probabilities: bool = False):
        """
        Constructor

        Parameters
        ----------
        temperature_only: bool, default: False
            If True, use Temperature Scaling. If False, use standard Logistic Calibration/Platt Scaling.
        independent_probabilities : bool, default=False
            boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(independent_probabilities)
        self.temperature_only = temperature_only

        self._weights = None
        self._bias = None

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()

        self._weights = None
        self._bias = None

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticCalibration':
        """
        Build Logistic Calibration model.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        LogisticCalibration
            Instance of class :class:`LogisticCalibration`.
        """

        X, y = super().fit(X, y)

        # if binary, use sigmoid instead of softmax
        if self.num_classes <= 2 or self.independent_probabilities:
            logit = self._inverse_sigmoid(X)
        else:
            logit = self._inverse_softmax(X)

        # otherwise, use SciPy optimzation. Usually, this is much faster
        if self.num_classes > 2:
            # convert ground truth to one hot if not binary
            y = self._get_one_hot_encoded_labels(y, self.num_classes)

        # if temperature scaling, fit single parameter
        if self.temperature_only:
            theta_0 = np.array(1.0)

        # else fit bias and weights for each class (one parameter on binary)
        else:
            if self._is_binary_classification():
                theta_0 = np.array([0.0, 1.0])
            else:
                theta_0 = np.concatenate((np.zeros(self.num_classes), np.ones(self.num_classes)))

        # perform minimization of squared loss - invoke SciPy optimization suite
        result = optimize.minimize(fun=self._loss_function, x0=theta_0,
                                   args=(logit, y))

        # get results of optimization
        if self.temperature_only:
            self._bias = 0.0
            self._weights = np.array(result.x)
        else:
            if self._is_binary_classification():
                self._bias = result.x[0]
                self._weights = result.x[1]
            else:
                self._bias = np.array(result.x[:self.num_classes])
                self._weights = np.array(result.x[self.num_classes:])

        return self

    @dimensions((1, 2))
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with uncalibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with calibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).
        """

        X = super().transform(X)
        if self.num_classes <= 2 or self.independent_probabilities:

            logit = self._inverse_sigmoid(X)
            logit = np.multiply(logit, self._weights) + self._bias
            calibrated = self._sigmoid(logit)

        else:

            logit = self._inverse_softmax(X)
            logit = np.multiply(logit, self._weights) + self._bias
            calibrated = self._softmax(logit)

        return calibrated

    @dimensions(1, (1, 2), (1, 2))
    def _loss_function(self, weights: np.ndarray, logit: np.ndarray, y: np.ndarray) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray
            Scaling factor and bias terms for logits that gets optimized.
        logit : np.ndarray
            NumPy 2-D array with logits.
        y : np.ndarray
            NumPy array with one-hot encoded ground truth (1-D for binary classification, 2-D for multi class).

        Returns
        -------
        float
            NLL-Loss.
        """

        if self.temperature_only:
            bias = 0.0
        else:
            if self._is_binary_classification():
                bias = weights[0]
                weights = weights[1]
            else:
                bias = np.array(weights[:self.num_classes])
                weights = np.array(weights[self.num_classes:])

        logit = np.multiply(logit, weights) + bias
        return self._nll_loss(logit, y)
