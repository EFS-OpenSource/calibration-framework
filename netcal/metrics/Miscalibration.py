# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.stats import binned_statistic_dd
from typing import Union
from netcal import accepts, dimensions


class _Miscalibration(object):
    """
    Generic base class to calculate Average/Expected/Maximum Calibration Error.
    ACE [1]_, ECE [2]_ and MCE [2]_ are used for measuring miscalibration on classification.
    The according variants D-ACE/D-ECE/D-MCE are used for object detection [3]_.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the Histogram Binning.
        On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    sample_threshold : int, optional, default: 1
        Bins with an amount of samples below this threshold are not included into the miscalibration metrics.

    References
    ----------
    .. [1] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht:
       "Obtaining well calibrated probabilities using bayesian binning."
       Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
       `Get source online <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_
    .. [2] Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi:
       "Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection."
       Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.
       `Get source online <https://openreview.net/pdf?id=S1lG7aTnqQ>`_
    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    epsilon = np.finfo(np.float).eps

    @accepts((int, tuple, list), bool, int)
    def __init__(self, bins: Union[int, tuple, list] = 10, detection: bool = False, sample_threshold: int = 1):
        """
        Constructor.

        Parameters
        ----------
        bins : int or iterable, default: 10
            Number of bins used by the Histogram Binning.
            On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
            If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
        detection : bool, default: False
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        sample_threshold : int, optional, default: 1
            Bins with an amount of samples below this threshold are not included into the miscalibration metrics.
        """
        self.bins = bins
        self.detection = detection
        self.sample_threshold = sample_threshold

    @classmethod
    def squeeze_generic(cls, a: np.ndarray, axes_to_keep: Union[int, list, tuple]) -> np.ndarray:
        """
        Squeeze input array a but keep axes defined by parameter 'axes_to_keep' even if the dimension is
        of size 1.

        Parameters
        ----------
        a : np.ndarray
            NumPy array that should be squeezed.
        axes_to_keep : int or iterable
            Axes that should be kept even if they have a size of 1.

        Returns
        -------
        np.ndarray
            Squeezed array.

        """

        # if type is int, convert to iterable
        if type(axes_to_keep) == int:
            axes_to_keep = (axes_to_keep,)

        # iterate over all axes in a and check if dimension is in 'axes_to_keep' or of size 1
        out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
        return a.reshape(out_s)

    @dimensions((1, 2), (1, 2), None, None, None)
    def _measure(self, X: np.ndarray, y: np.ndarray, metric: str, return_map: bool = False,
                 return_num_samples: bool = False) -> Union[float, tuple]:
        """
        Measure calibration by given predictions with confidence and the according ground truth.
        Assume binary predictions with y=1.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
        metric : str
            Determine metric to measure. Must be one of 'ACE', 'ECE' or 'MCE'.
        return_map: bool, optional, default: False
            If True, return map with miscalibration metric separated into all remaining dimension bins.
        return_num_samples : bool, optional, default: False
            If True, also return the number of samples in each bin.

        Returns
        -------
        float or tuple of (float, np.ndarray) or tuple of (float, np.ndarray, np.ndarray)
            Always returns miscalibration metric.
            If 'return_map' is False, return miscalibration metric only.
            If 'return_map' is True, return tuple with miscalibration metric and map over all bins.
            If 'return_num_samples' is False, return map with miscalibration metric.
            If 'return_num_samples' is True, return tuple with miscalibration metric map and number of samples
            in each bin.
        """

        # ---------------------------------------
        # perform parameter checks

        # remove single-dimensional entries if present
        X = self.squeeze_generic(X, axes_to_keep=0)
        y = self.squeeze_generic(y, axes_to_keep=0)

        # check if metric is correct set
        if not isinstance(metric, str):
            raise AttributeError('Parameter \'metric\' must be string \'ACE\', \'ECE\' or \'MCE\'.')
        if not metric.lower() in ['ace', 'ece', 'mce']:
            raise AttributeError('Parameter \'metric\' must be string \'ACE\', \'ECE\' or \'MCE\'.')
        else:
            metric = metric.lower()

        if y.shape[0] != X.shape[0]:
            raise AttributeError('Number of samples given by \'X\' and \'y\' is not equal.')

        # check number of given samples
        if y.size <= 0:
            raise ValueError("No samples provided.")

        elif len(y.shape) == 2:

            # still assume y as binary with ground truth labels present in y=1 entry
            if y.shape[1] <= 2:
                y = y[:, -1]

            # assume y as one-hot encoded
            else:
                y = np.argmax(y, axis=1)

        # we need at least 2 dimensions for this algorithm - thus reshape X if only one dimension is present
        # assume binary classification
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))
            prediction = np.ones(X.shape[0])

        # got 2D array for X?
        elif len(X.shape) == 2:

            # on detection mode, assume all predictions as 'matched'
            if self.detection:
                prediction = np.ones(X.shape[0])

            # on classification, if less than 2 entries for 2nd dimension are present, assume binary classification
            # (independent sigmoids are currently not supported)
            elif X.shape[1] == 1:
                prediction = np.ones(X.shape[0])

            # classification and more than 1 entry for 2nd dimension? assume multiclass classification
            else:
                prediction = np.argmax(X, axis=1)
                X = np.reshape(np.max(X, axis=1), (-1, 1))
        else:
            prediction = np.ones_like(X)

        # clip to (0, 1) in order to get all samples into binning scheme
        X = np.clip(X, self.epsilon, 1.-self.epsilon)

        # calculate 'matched' (0 or 1)
        matched = prediction == y
        total_samples = X.shape[0]

        # get number of features for detection mode calibration
        if self.detection:
            num_features = X.shape[1]

            # check bins parameter
            # is int? distribute to all dimensions
            if isinstance(self.bins, int):
                bins = [self.bins, ] * num_features

            # is iterable? check for compatibility with all properties found
            elif isinstance(self.bins, (tuple, list)):
                if len(self.bins) != num_features:
                    raise AttributeError("Length of \'bins\' parameter must match number of features.")
                bins = self.bins
            else:
                raise AttributeError("Unknown type of parameter \'bins\'.")
        else:

            if not isinstance(self.bins, int):
                raise AttributeError("Parameter \'bins\' must be int for classification mode.")

            bins = [self.bins]

        # ---------------------------------------
        # get bin bounds
        bin_bounds = [np.linspace(0.0, 1.0, bins + 1) for bins in bins]

        # X must have the same shape as y
        acc_hist, _, _ = binned_statistic_dd(X, matched, statistic='mean', bins=bin_bounds)
        conf_hist, _, _ = binned_statistic_dd(X, X[:, 0], statistic='mean', bins=bin_bounds)
        num_samples_hist, _ = np.histogramdd(X, bins=bin_bounds)

        # convert NaN entries to 0
        acc_hist, conf_hist, num_samples_hist = [np.nan_to_num(x, nan=0.0) for x in (acc_hist, conf_hist, num_samples_hist)]

        # in order to determine miscalibration w.r.t. additional features (excluding confidence dimension),
        # reduce the first (confidence) dimension and determine the amount of samples in the remaining bins
        samples_hist_reduced_conf = np.sum(num_samples_hist, axis=0)

        # first, get deviation map
        deviation_map = np.abs(acc_hist - conf_hist)

        # second, determine metric scheme
        if metric == 'ace':

            # ace is the average miscalibration weighted by the amount of non-empty bins
            # for the bin map, reduce confidence dimension
            reduced_deviation_map = np.sum(deviation_map, axis=0)
            non_empty_bins = np.count_nonzero(num_samples_hist, axis=0)

            # divide by leaving out empty bins (those are initialized to 0)
            bin_map = np.divide(reduced_deviation_map, non_empty_bins,
                                out=np.zeros_like(reduced_deviation_map), where=non_empty_bins != 0)

            miscalibration = np.sum(bin_map / np.count_nonzero(np.sum(num_samples_hist, axis=0)))

        elif metric == 'ece':

            # relative number of samples in each bin (including confidence dimension)
            rel_samples_hist = num_samples_hist / total_samples
            miscalibration = np.sum(deviation_map * rel_samples_hist)

            # The following computation is a little bit confusing but necessary because:
            # We are interested in the D-ECE score for each feature bin (excluding the confidence dimension) separately.
            # Thus, we need to know the total amount of samples over all confidence bins for each bin
            # combination in the remaining dimensions separately.
            # This amount of samples for each bin combination is then treated as the total amount of samples in order
            # to compute the D-ECE in the current bin combination properly.

            # extend the reduced histogram again
            extended_hist = np.repeat(
                np.expand_dims(samples_hist_reduced_conf, axis=0),
                num_samples_hist.shape[0],
                axis=0
            )

            # get the relative amount of samples according to a certain bin combination over all confidence bins
            # leave out empty bin combinations
            rel_samples_hist_reduced_conf = np.divide(num_samples_hist,
                                                      extended_hist,
                                                      out=np.zeros_like(num_samples_hist),
                                                      where=extended_hist != 0)

            # sum weighted deviation along confidence dimension
            bin_map = np.sum(deviation_map * rel_samples_hist_reduced_conf, axis=0)

        elif metric == 'mce':

            # get maximum deviation
            miscalibration = np.max(deviation_map)
            bin_map = np.max(deviation_map,  axis=0)

        else:
            raise ValueError("Unknown miscalibration metric. This exception is fatal at this point. Fix your implementation.")

        if return_map or return_num_samples:
            return_value = (float(miscalibration),)

            if return_map:
                return_value = return_value + (bin_map,)
            if return_num_samples:
                return_value = return_value + (samples_hist_reduced_conf,)

            return return_value
        else:
            return float(miscalibration)
