# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from netcal import AbstractCalibration, dimensions, accepts


class HistogramBinning(AbstractCalibration):
    """
    Simple Histogram Binning calibration method. Each prediction is sorted into a bin
    and assigned its calibrated confidence estimate. This method normally works for binary
    classification. For multiclass classification, this method is applied into a 1-vs-all manner.

    The bin boundaries are either chosen to be
    equal length intervals or to equalize the number of samples in each bin.

    Parameters
    ----------
    bins : int
        Number of bins used by the Histogram Binning.
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Zadrozny, Bianca and Elkan, Charles:
    "Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers."
    In ICML, pp. 609–616, 2001.
    `Get source online <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.3039&rep=rep1&type=pdf>`_

    Zadrozny, Bianca and Elkan, Charles:
    "Transforming classifier scores into accurate multiclass probability estimates."
    In KDD, pp. 694–699, 2002.
    `Get source online <https://www.researchgate.net/profile/Charles_Elkan/publication/2571315_Transforming_Classifier_Scores_into_Accurate_Multiclass_Probability_Estimates/links/0fcfd509ae852a8bb9000000.pdf>`_
    """

    @accepts(int, bool)
    def __init__(self, bins: int, independent_probabilities: bool = False):
        """
        Create an instance of `HistogramBinning`.

        Parameters
        ----------
        bins : int
            Number of bins used by the Histogram Binning.
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(independent_probabilities)

        self._bins = bins
        self._bin_boundaries = None

        # initialize bin calibrated confidences
        self._bin_confidence = np.array([0.5, ] * self._bins)

        # for multi class calibration with K classes, K binary calibration models are needed
        self._multiclass_instances = []

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        self._bin_boundaries = None

        # initialize bin calibrated confidences
        self._bin_confidence = np.array([0.5, ] * self._bins)

        # for multi class calibration with K classes, K binary calibration models are needed
        for instance in self._multiclass_instances:
            del instance

        self._multiclass_instances.clear()

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HistogramBinning':
        """
        Function call to build the calibration model.

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
        HistogramBinning
            Instance of class :class:`HistogramBinning`.
        """

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification():

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, HistogramBinning, self._bins)
            return self

        self._bin_boundaries = np.linspace(0.0, 1.0, self._bins + 1)

        # now calculate bin indices
        # this function gives the index for the upper bound of the according bin
        # for each sample. Thus, decrease by 1 to get the bin index
        current_indices = np.digitize(x=X, bins=self._bin_boundaries, right=True) - 1

        # if an index is out of bounds (e.g. 0), sort into first bin
        current_indices[current_indices == -1] = 0
        current_indices[current_indices == self._bins] = self._bins - 1

        # mean accuracy is new confidence in each bin
        for bin in range(self._bins):
            bin_gt = y[current_indices == bin]
            if bin_gt.size > 0:
                self._bin_confidence[bin] = np.mean(bin_gt)
            else:
                self._bin_confidence[bin] = (self._bin_boundaries[bin] + self._bin_boundaries[bin+1]) / 2.

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

            # binary classification problem but got two entries? (probability for 0 and 1 separately)?
            # we only need probability p for Y=1 (probability for 0 is (1-p) )
            if len(X.shape) == 2:
                X = np.array(X[:, 1])

            current_indices = np.digitize(x=X, bins=self._bin_boundaries, right=True) - 1

            # if an index is out of bounds (e.g. 0), sort into first bin
            current_indices[current_indices == -1] = 0
            current_indices[current_indices == self._bins] = self._bins - 1

            # iterate over each bin and assign bin confidence to each prediction
            for bin in range(self._bins):
                calibrated[current_indices == bin] = self._bin_confidence[bin]

        return calibrated

    def get_degrees_of_freedom(self) -> int:
        """
        Needed for BIC. Returns the degree of freedom. This simply returns the
        number of bins.

        Returns
        -------
        int
            Integer with degree of freedom.
        """

        return self._bins
