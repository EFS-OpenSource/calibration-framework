# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# ---------------------------------------------------------
#
# Some methods are inspired by 'betacal' package. The MIT-license is included in the following:
#
# Copyright (c) 2017 betacal (https://pypi.org/project/betacal)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
from scipy.optimize import minimize
from calibration import AbstractCalibration, dimensions, accepts


class BetaCalibration(AbstractCalibration):
    """
    Beta Calibration method. This calibration method uses a Beta function to fit a function to
    ground truth data distribution. This is originally for binary classification. For multi class classification,
    this method is used in a 'one vs. all' fashion.

    Some methods are inspired by `betacal <https://pypi.org/project/betacal>`_ package
    - Copyright (c) 2017 betacal (MIT License)

    Parameters
    ----------
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Kull, Meelis, Telmo Silva Filho, and Peter Flach:
    "Beta calibration: a well-founded and easily implemented improvement on logistic
    calibration for binary classifiers."
    Artificial Intelligence and Statistics, PMLR 54:623-631, 2017.
    `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`_
    """

    @accepts(bool)
    def __init__(self, independent_probabilities: bool = False):
        """
        Constructor

        Parameters
        ----------
        independent_probabilities : bool, optional, default: False
            boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """
        super().__init__(independent_probabilities)

        self._weights = None
        self._bias = None

        # for multi class calibration with K classes, K binary calibration models are needed
        self._multiclass_instances = []

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()

        self._weights = None
        self._bias = None

        for instance in self._multiclass_instances:
            del instance

        self._multiclass_instances.clear()

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BetaCalibration':
        """
        Build Beta Calibration model.

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
        BetaCalibration
            Instance of class :class:`BetaCalibration`.
        """

        X, y = super().fit(X, y)
        solver = 'SLSQP'

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification():

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, BetaCalibration)
            return self

        # build 2-D array with [log(x), -log(1-x)]
        # convert confidences array and clip values to open interval (0, 1)
        data_input = np.reshape(X, (-1, 1))
        data_input = np.hstack((data_input, 1. - data_input))
        data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)

        data_input = np.log(data_input)
        data_input[:, 1] *= -1

        # --------------------------------------------------------------------------------
        # start initial solver

        theta_0 = np.ones(3) * 0.5
        result = minimize(method=solver,
                          fun=self._loss_function, x0=theta_0,
                          args=(data_input, y))

        self._bias = result.x[0]
        self._weights = np.array([result.x[1], result.x[2]]).reshape(-1, 1)

        # --------------------------------------------------------------------------------
        # check calculated weights
        # if either param a or param b < 0, the second distribution's parameter is fixed to zero
        # the logistic fit is repeated afterwards with remaining distribution
        if self._weights[0, 0] < 0:

            # only keep second distribution of parameter b
            data_input = data_input[:, 1].reshape(-1, 1)

            # invoke logistic fit with remaining distribution again
            theta_0 = np.ones(2)
            result = minimize(method=solver,
                              fun=self._loss_function, x0=theta_0,
                              args=(data_input, y))

            # set parameter b and new bias (parameter c afterwards)
            self._bias = result.x[0]
            self._weights = np.array([[0.0, result.x[1]]]).reshape(-1, 1)

        elif self._weights[1, 0] < 0:

            # only keep first distribution of parameter a
            data_input = data_input[:, 0].reshape(-1, 1)

            # invoke logistic fit with remaining distribution again
            theta_0 = np.ones(2)
            result = minimize(method=solver,
                              fun=self._loss_function, x0=theta_0,
                              args=(data_input, y))

            # set parameter a and new bias (parameter c afterwards)
            self._bias = result.x[0]
            self._weights = np.array([result.x[1], 0.0]).reshape(-1, 1)

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

        # store original shape (1-D or 2-D)
        orig_shape = X.shape

        # prepare return value vector
        calibrated = np.zeros(X.shape)

        # if multiclass classification problem, use binning models for each label separately
        if not self._is_binary_classification():

            # get all available labels and iterate over each one
            for (label, binning_model) in self._multiclass_instances:
                onevsall_confidence = self._get_one_vs_all_confidence(X, label)
                onevsall_calibrated = binning_model.transform(onevsall_confidence)

                # choose right position for submodel's calibration
                calibrated[:, label] = onevsall_calibrated

            if not self.independent_probabilities:
                normalizer = np.sum(calibrated, axis=1, keepdims=True)
                calibrated = np.divide(calibrated, normalizer)

        # on binary classification, it's much easier
        else:

            # build 2-D array with [log(x), -log(1-x)]
            # convert confidences array and clip values to open interval (0, 1)
            data_input = np.reshape(X, (-1, 1))
            data_input = np.hstack((data_input, 1. - data_input))
            data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)

            data_input = np.log(data_input)
            data_input[:, 1] *= -1

            # compute logistic fit
            calibrated = self._sigmoid(np.matmul(data_input, self._weights) + self._bias)[:, 0]
            calibrated = np.reshape(calibrated, orig_shape)

        return calibrated

    def _loss_function(self, weights: np.ndarray, data_input: np.ndarray, y: np.ndarray) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray, shape=(3,)
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=(n_samples, 2)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth (normal ground truth for binary or one-vs-all for multi class)

        Returns
        -------
        float
            NLL-Loss
        """

        bias = weights[0]
        weights = np.array(weights[1:]).reshape(-1, 1)

        # compute logits and NLL loss afterwards
        logit = np.matmul(data_input, weights) + bias
        return self._nll_loss(logit, y)
