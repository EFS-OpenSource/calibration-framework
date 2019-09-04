# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from calibration import AbstractCalibration, dimensions, accepts
from sklearn.isotonic import IsotonicRegression as sklearn_iso


class IsotonicRegression(AbstractCalibration):
    """
    Isotonic Regression method. This method is similar to :class:`HistogramBinning` but with dynamic bin sizes
    and boundaries. A piecewise constant function gets fit to ground truth labels sorted by
    given confidence estimates.

    Parameters
    ----------
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Zadrozny, Bianca and Elkan, Charles:
    "Transforming classifier scores into accurate multiclass probability estimates."
    In KDD, pp. 694–699, 2002.
    `Get source online <https://www.researchgate.net/profile/Charles_Elkan/publication/2571315_Transforming_Classifier_Scores_into_Accurate_Multiclass_Probability_Estimates/links/0fcfd509ae852a8bb9000000.pdf>`_
    """

    @accepts(bool)
    def __init__(self, independent_probabilities: bool = False):
        """
        Create an instance of `IsotonicRegression`.

        Parameters
        ----------
        independent_probabilities : bool, optional, default: False
            boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(independent_probabilities)
        self._multiclass_instances = []
        self._iso = None

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        # for multi class calibration with K classes, K binary calibration models are needed
        for instance in self._multiclass_instances:
            del instance

        self._multiclass_instances.clear()
        self._iso = None

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'IsotonicRegression':
        """
        Build Isotonic Regression model.

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
        IsotonicRegression
            Instance of class :class:`IsotonicRegression`.
        """

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification():

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, IsotonicRegression)
            return self

        # important: sort arrays by confidence
        X, y = self._sort_arrays(X, y)

        # use isotonic regression routine from sklearn
        # and store as member variable
        self._iso = sklearn_iso(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        self._iso.fit(X, y)

        return self

    @dimensions((1, 2))
    def transform(self, X: np.ndarray):
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
                # normalize to keep probability sum of 1
                normalizer = np.sum(calibrated, axis=1, keepdims=True)
                calibrated = np.divide(calibrated, normalizer)

        # on binary classification, it's much easier
        else:

            # use previously built isotonic regression model for prediction
            calibrated = self._iso.transform(X)

        return calibrated
