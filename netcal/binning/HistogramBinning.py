# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.stats import binned_statistic_dd
from typing import Union
from netcal import AbstractCalibration, dimensions, accepts


class HistogramBinning(AbstractCalibration):
    """
    Simple Histogram Binning calibration method. This method is originally proposed by [1]_. Each prediction is sorted into a bin
    and assigned its calibrated confidence estimate. This method normally works for binary
    classification. For multiclass classification, this method is applied into a 1-vs-all manner [2]_.

    The bin boundaries are either chosen to be
    equal length intervals or to equalize the number of samples in each bin.

    On object detection, use a multidimensional binning to include additional information of the box
    regression branch [3]_.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the Histogram Binning.
        On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    equal_intervals : bool, optional, default: True
        If True, the bins have the same width. If False, the bins are splitted to equalize
        the number of samples in each bin.
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
    .. [1] Zadrozny, Bianca and Elkan, Charles:
       "Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers."
       In ICML, pp. 609–616, 2001.
       `Get source online <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.3039&rep=rep1&type=pdf>`_

    .. [2] Zadrozny, Bianca and Elkan, Charles:
       "Transforming classifier scores into accurate multiclass probability estimates."
       In KDD, pp. 694–699, 2002.
       `Get source online <https://www.researchgate.net/profile/Charles_Elkan/publication/2571315_Transforming_Classifier_Scores_into_Accurate_Multiclass_Probability_Estimates/links/0fcfd509ae852a8bb9000000.pdf>`_

    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, in press.
    """

    @accepts((int, tuple, list), bool, bool, bool)
    def __init__(self, bins: Union[int, tuple, list] = 10, equal_intervals: bool = True,
                 detection: bool = False, independent_probabilities: bool = False):
        """
        Create an instance of `HistogramBinning`.

        Parameters
        ----------
        bins : int or iterable, default: 10
            Number of bins used by the Histogram Binning.
            On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
            If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
        equal_intervals : bool, optional, default: True
            If True, the bins have the same width. If False, the bins are splitted to equalize
            the number of samples in each bin.
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

        self.bins = bins
        self.equal_intervals = equal_intervals

        if not self.equal_intervals:
            raise ValueError("Parameter \'equal_intervals=False\' is currently not implemented.")

        # for multi class calibration with K classes, K binary calibration models are needed
        self._multiclass_instances = []

        # holds the multi-dimensional bin map with calibrated confidence estimates
        self._bin_map = None
        self._bin_bounds = None

        self._num_combination = 0

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        self._bin_map = None
        self._bin_bounds = None

        self._num_combination = 0

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
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        HistogramBinning
            Instance of class :class:`HistogramBinning`.
        """

        # ---------------------------------------
        # decide if case is binary classification, multiclass classification or detection

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification() and not self.detection:

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, HistogramBinning, self.bins)
            return self

        # ---------------------------------------
        # case: binary classification or detection

        # we need at least 2 dimensions for this algorithm - thus reshape X if only one dimension is present
        # assume binary classification
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))
            prediction = np.ones(X.shape[0])

        # got 2D array for X?
        elif len(X.shape) == 2:

            # on binary or detection mode, assume all predictions as 'matched'
            prediction = np.ones(X.shape[0])

        else:
            raise ValueError("More than 2 dimensions are not allowed. "
                             "This is a fatal error at this point. Check your implementation.")

        # calculate 'matched' (0 or 1)
        matched = prediction == y
        X = np.clip(X, self.epsilon, 1.-self.epsilon)

        # get number of features for detection mode calibration
        if self.detection:
            num_features = X.shape[1]

            # check bins parameter
            # is int? distribute to all dimensions
            if isinstance(self.bins, int):
                self.bins = [self.bins, ] * num_features

            # is iterable? check for compatibility with all properties found
            elif isinstance(self.bins, (tuple, list)):
                if len(self.bins) != num_features:
                    raise AttributeError("Length of \'bins\' parameter must match number of features.")
            else:
                raise AttributeError("Unknown type of parameter \'bins\'.")
        else:

            if not isinstance(self.bins, int):
                raise AttributeError("Parameter \'bins\' must be int for classification mode.")

            self.bins = [self.bins]

        # ---------------------------------------
        # get bin bounds
        self._bin_bounds = [np.linspace(0.0, 1.0, bin + 1) for bin in self.bins]

        # X must have the same shape as y
        acc_hist, _, _ = binned_statistic_dd(X, matched, statistic='mean', bins=self._bin_bounds)

        # identify all NaN indices
        nan_indices = np.nonzero(np.isnan(acc_hist))

        # first dimension is confidence dimension - use the binning in this dimension to
        # determine median as fill values for empty bins
        confidence_median = 0.5 * (self._bin_bounds[0][nan_indices[0]] + self._bin_bounds[0][nan_indices[0]+1])
        acc_hist[nan_indices] = confidence_median

        # assign histogram as bin mapping
        self._bin_map = acc_hist

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
        calibrated = np.zeros_like(X)

        # if multiclass classification problem, use binning models for each label separately
        if not self._is_binary_classification() and not self.detection:

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

        else:

            if len(X.shape) == 1:
                X = np.reshape(X, (-1, 1))

            # on detection, this is equivalent to the number of features
            # on binary classification, this is simply 1
            # on multiclass classification, we perform one vs. all binning - thus, it results in multiple binary cases
            num_features = X.shape[1]
            if self.equal_intervals:

                bin_indices = []

                # now calculate bin indices
                # this function gives the index for the upper bound of the according bin
                # for each sample. Thus, decrease by 1 to get the bin index
                for i in range(num_features):
                    indices = np.digitize(x=X[:, i], bins=self._bin_bounds[i], right=True) - 1

                    # if an index is out of bounds (e.g. 0), sort into first bin
                    indices[indices == -1] = 0
                    indices[indices == self.bins[i]] = self.bins[i] - 1
                    bin_indices.append(indices)
            else:
                # TODO: implement equal intervals
                raise ValueError("Parameter \'equal_intervals=False\' is currently not implemented.")

            calibrated = self._bin_map[tuple(bin_indices)]

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

        return int(np.prod(self.bins))
