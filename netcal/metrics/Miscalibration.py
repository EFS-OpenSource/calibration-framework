# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union, Iterable, Tuple, List

import numpy as np
from scipy.stats import binned_statistic_dd
from netcal import accepts, hpdi


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
    equal_intervals : bool, optional, default: True
        If True, the bins have the same width. If False, the bins are splitted to equalize
        the number of samples in each bin.
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
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.
       `Get source online <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf>`_
    """

    epsilon = np.finfo(np.float).eps

    @accepts((int, tuple, list), bool, bool, int)
    def __init__(self, bins: Union[int, Iterable[int]] = 10, equal_intervals: bool = True,
                 detection: bool = False, sample_threshold: int = 1):
        """ Constructor. For parameter doc see class doc. """

        self.bins = bins
        self.detection = detection
        self.sample_threshold = sample_threshold
        self.equal_intervals = equal_intervals

    @classmethod
    def squeeze_generic(cls, a: np.ndarray, axes_to_keep: Union[Iterable[int], int]) -> np.ndarray:
        """ Squeeze input array a but keep axes defined by parameter
        'axes_to_keep' even if the dimension is of size 1. """

        # if type is int, convert to iterable
        if type(axes_to_keep) == int:
            axes_to_keep = (axes_to_keep,)

        # iterate over all axes in a and check if dimension is in 'axes_to_keep' or of size 1
        out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
        return a.reshape(out_s)

    def reduce(self, histogram: np.ndarray, distribution: np.ndarray, axis: int, reduce_result: Tuple = None):
        """
        Calculate the weighted mean on a given histogram based on a dedicated data distribution.
        If 'reduce_result' is given, reuse the data distribution of the previous result instead of the distribution
        given by 'distribution' parameter.
        """

        if reduce_result is None:
            # in order to determine miscalibration w.r.t. additional features (excluding confidence dimension),
            # reduce the first (confidence) dimension and determine the amount of samples in the remaining bins
            samples_map = np.sum(distribution, axis=axis)

            # The following computation is a little bit confusing but necessary because:
            # We are interested in the miscalibration score (here mainly D-ECE) as well as the confidence, accuracy and
            # uncertainty for each feature bin (excluding the confidence dimension) separately.
            # Thus, we need to know the total amount of samples over all confidence bins for each bin combination in the
            # remaining dimensions separately. This amount of samples for each bin combination is then treated as the total
            # amount of samples in order to compute the D-ECE in the current bin combination properly.

            # extend the reduced histogram again
            extended_hist = np.repeat(
                np.expand_dims(samples_map, axis=axis),
                distribution.shape[axis],
                axis=axis
            )

            # get the relative amount of samples according to a certain bin combination over all confidence bins
            # leave out empty bin combinations
            rel_samples_hist_reduced_conf = np.divide(distribution,
                                                      extended_hist,
                                                      out=np.zeros_like(distribution),
                                                      where=extended_hist != 0)
        else:
            # reuse reduced data distribution from a previous call
            rel_samples_hist_reduced_conf = reduce_result[1]

        # now reduce confidence dimension of accuracy, confidence and uncertainty histograms
        weighted_mean = np.sum(histogram * rel_samples_hist_reduced_conf, axis=axis)

        return weighted_mean, rel_samples_hist_reduced_conf

    def prepare(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray],
                batched: bool = False, uncertainty: str = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, int]:
        """ Check input data. For detailed documentation of the input parameters, check "_measure" method. """

        # batched: interpret X and y as multiple predictions
        if not batched:
            assert isinstance(X, np.ndarray), 'Parameter \'X\' must be Numpy array if not on batched mode.'
            assert isinstance(y, np.ndarray), 'Parameter \'y\' must be Numpy array if not on batched mode.'
            X, y = [X], [y]

        # if we're in batched mode, create new lists for X and y to prevent overriding
        else:
            assert isinstance(X, (list, tuple)), 'Parameter \'X\' must be type list on batched mode.'
            assert isinstance(y, (list, tuple)), 'Parameter \'y\' must be type list on batched mode.'
            X, y = [x for x in X], [y_ for y_ in y]

        # if input X is of type "np.ndarray", convert first axis to list
        # this is necessary for the following operations
        if isinstance(X, np.ndarray):
            X = [x for x in X]

        if isinstance(y, np.ndarray):
            y = [y0 for y0 in y]

        # empty list to collect uncertainty estimates for each sample provided in each batch
        matched, sample_uncertainty = [], []

        num_features = -1
        for i, (batch_X, batch_y) in enumerate(zip(X, y)):

            # we need at least 2 dimensions (for classification as well as for detection)
            if batch_X.ndim == 1:
                X[i] = batch_X = np.reshape(batch_X, (-1, 1))

            # -------------------------------------------------
            # process uncertainty mode first
            batch_X, batch_y, batch_uncertainty = self._prepare_uncertainty(batch_X, batch_y, uncertainty)
            X[i], y[i] = batch_X, batch_y

            # uncertainty (std deviation) of X values of current batch
            sample_uncertainty.append(batch_uncertainty)

            # -------------------------------------------------
            # check and prepare input data
            batch_X, batch_y, batch_matched = self._prepare_input(batch_X, batch_y)
            X[i], y[i] = batch_X, batch_y
            matched.append(batch_matched)

            # -------------------------------------------------
            # check if number of features is consistent along all batches
            batch_num_features = batch_X.shape[1] if self.detection and batch_X.ndim > 1 else 1

            # get number of additional dimensions (if not initialized)
            if num_features == -1:
                num_features = batch_num_features
            else:
                # if number of features is not equal over all instances, raise exception
                assert num_features == batch_num_features, "Unequal number of classes/features given in batched mode."

        # -----------------------------------------------------
        # prepare bin amount with the current amount of features

        bin_bounds = self._prepare_bins(X, num_features)

        return X, matched, sample_uncertainty, bin_bounds, num_features

    def binning(self, bin_bounds: List, samples: np.ndarray, *values: Iterable, nan: float = 0.0) -> Tuple:
        """
        Perform binning on value (and all additional values passed) based on samples.

        Parameters
        ----------
        bin_bounds : list, length=samples.shape[1]
            Binning boundaries used for each dimension given in 'samples' parameter.
        samples : np.ndarray of shape (n_samples, n_features)
            Array used to group all samples into bins.
        *values : instances np.ndarray of shape (n_samples, 1)
            Arrays whose values are binned.
        nan : float, optional default: 0.0
            If a bin has no samples or less than defined sample_threshold, the according bin is marked as
            NaN. Specify fill float to insert instead of NaN.

        Returns
        -------
        tuple of length equal to the amount of passed value arrays with binning schemes and an additional histogram
        with number of samples in each bin as well as an index tuple containing the bin indices.
        """

        # determine number of samples in histogram bins
        num_samples_hist, _ = np.histogramdd(samples, bins=bin_bounds)
        binning_schemes = []
        binning_result = None

        # iterate over passed value arrays
        for val in values:
            binning_result = binned_statistic_dd(samples, val, statistic='mean', bins=bin_bounds, binned_statistic_result=binning_result)
            hist, _, _ = binning_result

            # blank out each bin that has less samples than a certain sample threshold in order
            # to improve robustness of the miscalibration scores
            # convert NaN entries to float
            hist[num_samples_hist < self.sample_threshold] = np.nan
            hist = np.nan_to_num(hist, nan=nan)

            binning_schemes.append(hist)

        binning_schemes.append(num_samples_hist)
        _, _, idx = binning_result

        # first step: expand bin numbers
        # correct bin number afterwards as this variable has offset of 1
        idx = np.asarray(np.unravel_index(idx, [len(bounds)+1 for bounds in bin_bounds]))
        idx -= 1

        # convert to tuple as this can be used for array indexing
        idx = tuple([dim for dim in idx])
        binning_schemes.append(idx)

        return tuple(binning_schemes)

    def process(self,
                metric: str,
                acc_hist: np.ndarray,
                conf_hist: np.ndarray,
                variance_hist: np.ndarray,
                num_samples_hist: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine miscalibration based on passed histograms.

        Parameters
        ----------
        metric : str
            Identifier to specify the used metric. Must be one of 'ace', 'ece' or 'mce'.
        acc_hist : np.ndarray of shape (n_bins, [n_bins, [n_bins, [...]]])
            Histogram with average accuracy in each bin.
        conf_hist : np.ndarray of shape (n_bins, [n_bins, [n_bins, [...]]])
            Histogram with average confidence in each bin.
        variance_hist : np.ndarray of shape (n_bins, [n_bins, [n_bins, [...]]])
            Histogram with average variance in each bin. This array is currently not used but
            might be utilized in the future.
        num_samples_hist : np.ndarray of shape (n_bins, [n_bins, [n_bins, [...]]])
            Histogram with number of samples in each bin.

        Returns
        -------
        tuple of length 6 (miscalibration score, miscalibration map, accuracy map, confidence map, variance map, num samples map)
        All maps without confidence dimension.
        """

        # in order to determine miscalibration w.r.t. additional features (excluding confidence dimension),
        # reduce the first (confidence) dimension and determine the amount of samples in the remaining bins
        samples_map = np.sum(num_samples_hist, axis=0)
        total_samples = np.sum(samples_map)

        # first, get deviation map
        deviation_map = np.abs(acc_hist - conf_hist)

        reduce_result = self.reduce(acc_hist, num_samples_hist, axis=0)
        acc_hist = reduce_result[0]

        conf_hist, _ = self.reduce(conf_hist, num_samples_hist, axis=0, reduce_result=reduce_result)
        variance_hist, _ = self.reduce(variance_hist, num_samples_hist, axis=0, reduce_result=reduce_result)

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

            # sum weighted deviation along confidence dimension
            bin_map, _ = self.reduce(deviation_map, num_samples_hist, axis=0, reduce_result=reduce_result)

        elif metric == 'mce':

            # get maximum deviation
            miscalibration = np.max(deviation_map)
            bin_map = np.max(deviation_map, axis=0)

        else:
            raise ValueError("Unknown miscalibration metric. This exception is fatal at this point. Fix your implementation.")

        return miscalibration, bin_map, acc_hist, conf_hist, variance_hist, samples_map
    
    def _prepare_bins(self, X: List[np.ndarray], num_features: int) -> List[List[np.ndarray]]:
        """ Prepare number of bins for binning scheme. """

        # check bins parameter
        # is int? distribute to all dimensions
        if isinstance(self.bins, int):
            bins = [self.bins, ] * num_features

        # is iterable? check for compatibility with all properties found
        elif isinstance(self.bins, (tuple, list)):
            if len(self.bins) != num_features:
                raise AttributeError("Length of \'bins\' parameter must match number of features.")
            else:
                bins = self.bins
        else:
            raise AttributeError("Unknown type of parameter \'bins\'.")

        # create an own set of bin boundaries for each batch in X
        bin_bounds = [[np.linspace(0.0, 1.0, bins + 1) for bins in bins] for _ in X]

        # on equal_intervals=True, simply use linspace
        # if the goal is to equalize the amount of samples in each bin, use np.quantile
        if not self.equal_intervals:
            for i, (batch_X, bounds) in enumerate(zip(X, bin_bounds)):
                for dim, b in enumerate(bounds):
                    quantile = np.quantile(batch_X[:, dim], q=b, axis=0)

                    # set lower and upper bounds to confidence limits
                    quantile[0] = 0.
                    quantile[-1] = 1.
                    bin_bounds[i][dim] = quantile

        return bin_bounds
    
    def _prepare_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Prepare structure of input data (number of dimensions, etc.) """

        # remove unnecessary dims if given
        y = self.squeeze_generic(y, axes_to_keep=0)

        # after processing uncertainty, we expect batch_X to be only 2-D afterwards (but probably with more samples)
        # if we had no uncertainty, we expect that anyway
        assert X.ndim <= 2, "Fatal error: invalid number of dimensions."
        assert y.size > 0, "No samples provided."
        assert X.shape[0] == y.shape[0], "Unequal number of samples given in X and y."

        # on detection mode, we only have binary samples
        if (y.ndim > 1 or (np.unique(y) > 1).any()) and self.detection:
            raise ValueError("On detection, only binary values for y are valid.")

        # on detection mode, leave y array untouched
        elif len(y.shape) == 2 and not self.detection:
            # still assume y as binary with ground truth labels present in y=1 entry
            if y.shape[1] <= 2:
                y = y[:, -1]

            # assume y as one-hot encoded
            else:
                y = np.argmax(y, axis=1)

        # clip to (0, 1) in order to get all samples into binning scheme
        X = np.clip(X, self.epsilon, 1. - self.epsilon)

        # -------------------------------------------------
        # now evaluate the accuracy/precision
        # on detection mode or binary classification, the accuracy/precision is already given in y
        if self.detection or len(np.unique(y)) <= 2:
            matched = np.array(y)

        # on multiclass classification, we need to evaluate the accuracy by the predictions in X
        else:
            matched = np.argmax(X, axis=1) == y
            X = np.max(X, axis=1, keepdims=True)

        return X, y, matched

    def _prepare_uncertainty(self, X: np.ndarray, y: np.ndarray, uncertainty: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Prepare input data for uncertainty handling. """

        # -------------------------------------------------
        # process uncertainty mode first
        if uncertainty is None:
            if X.ndim == 3:
                print("Input data is 3D but uncertainty type not specified. Using \'mean\'.")

                # set uncertainty type and according list within this loop since this will be executed
                # only once (if ever)
                uncertainty = 'mean'
            else:
                return X, y, np.zeros_like(X)

        # on uncertainty mode, there might be two reasons why there are only 2 dimensions:
        # first case: no additional uncertainty support, only observation and feature/multiclass dimension given
        # second case: no additional features/multiclass probs, only realization and obervation dimensions are given
        if X.ndim == 2:

            # identify axis that holds the observation dimension
            obs_dim = [shape == y.shape[-1] for shape in X.shape].index(True)

            # first case: no probability/realization axis - prepend dimension
            # this is equivalent to no uncertainty
            if obs_dim == 0:
                X = np.expand_dims(X, axis=0)

            # second case: no feature/multiclass prob axis - append axis
            elif obs_dim == 1:
                X = np.expand_dims(X, axis=2)

            else:
                raise ValueError("Input data is incosistent for uncertainty mode.")

        # process the different types of uncertainty
        # first one: MC integration with additional uncertainty per sample
        if uncertainty in ['mean', 'median', 'mode']:

            # first condition: check for invalid detection mode
            # second condition: check for invalid binary classification mode
            # third condition: check for invalid multiclass classification mode
            if (y.ndim == 2 and self.detection) or \
                    (y.ndim == 2 and X.ndim == 2 and not self.detection) or \
                    (y.ndim == 3 and X.ndim == 3 and not self.detection):
                raise ValueError("Separate ground-truth information is provided for each probability forward pass "
                                 "but uncertainty type \'mean\', \'median\' or \'mode\' is specified.")

            if uncertainty == 'mean':
                X, X_uncertainty = self._mean(X)
            elif uncertainty == 'median':
                X, X_uncertainty = self._median(X)
            elif uncertainty == 'mode':
                X, X_uncertainty = self._mode(X)
            else:
                raise AttributeError("Fatal implementation error.")

        # second one: treat each parameter set separately
        # however, we can not assess the uncertainty of a single sample in this case
        elif uncertainty == 'flatten':
            X, y = self._flatten(X, y)
            X_uncertainty = np.zeros_like(X)
        else:
            raise NotImplementedError("Uncertainty type \'%s\' is not implemented." % uncertainty)

        return X, y, X_uncertainty

    def _flatten(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ repeat features to flattened confidence estimates """

        # multiclass classification
        if X.ndim == 3 and not self.detection:

            n_classes = X.shape[1]

            # if y is 3-D on multiclass classification, we also have separate ground-truth information available
            # then simply flatten
            if y.ndim == 3:
                y = np.reshape(y, (-1, y.shape[2]))
            else:
                y = np.tile(y, X.shape[0])

            # use NumPy's reshape function to flatten array along first axis
            X = np.reshape(X, (-1, n_classes))

        # binary classification
        else:
            n_features = X.shape[2]

            # if y is 2-D on binary classification or detection, we also
            # have separate ground-truth information available
            # then simply flatten
            if y.ndim == 2:
                y = y.flatten()
            else:
                y = np.tile(y, X.shape[0])

            # use NumPy's reshape function to flatten array along first axis
            X = np.reshape(X, (-1, n_features))

        return X, y

    def _mean(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ return mean of input data along first axis """
        return np.mean(X, axis=0), np.var(X, axis=0)

    def _median(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ return median of input data along first axis """
        return np.median(X, axis=0), np.var(X, axis=0)

    def _mode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ return mode of input data along first axis """

        ret = []
        # credible interval bounds on confidence only
        for feature in range(X.shape[-1]):
            bounds = hpdi(X[..., feature], 0.05)
            mode = np.sum(bounds, axis=1) / 2.
            ret.append(mode)

        return np.stack(ret, axis=1), np.var(X, axis=0)

    def _measure(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray],
                 metric: str, batched: bool = False, uncertainty: str = None,
                 return_map: bool = False,
                 return_num_samples: bool = False,
                 return_uncertainty_map: bool = False) -> Union[float, Tuple]:
        """
        Measure calibration by given predictions with confidence and the according ground truth.
        Assume binary predictions with y=1.

        Parameters
        ----------
        X : iterable of np.ndarray, or np.ndarray of shape=([n_bayes], n_samples, [n_classes/n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            If 3-D, interpret first dimension as samples from an Bayesian estimator with mulitple data points
            for a single sample (e.g. variational inference or MC dropout samples).
            If this is an iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : iterable of np.ndarray with same length as X or np.ndarray of shape=([n_bayes], n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            If 3-D, interpret first dimension as samples from an Bayesian estimator with mulitple data points
            for a single sample (e.g. variational inference or MC dropout samples).
            If iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.
        uncertainty : str, optional, default: False
            Define uncertainty handling if input X has been sampled e.g. by Monte-Carlo dropout or similar methods
            that output an ensemble of predictions per sample. Choose one of the following options:
            - flatten:  treat everything as a separate prediction - this option will yield into a slightly better
                        calibration performance but without the visualization of a prediction interval.
            - mean:     compute Monte-Carlo integration to obtain a simple confidence estimate for a sample
                        (mean) with a standard deviation that is visualized.
        metric : str
            Determine metric to measure. Must be one of 'ACE', 'ECE' or 'MCE'.
        return_map: bool, optional, default: False
            If True, return map with miscalibration metric separated into all remaining dimension bins.
        return_num_samples : bool, optional, default: False
            If True, also return the number of samples in each bin.
        return_uncertainty_map : bool, optional, default: False
            If True, also return the average deviation of the confidence within each bin.

        Returns
        -------
        float or tuple of (float, np.ndarray, [np.ndarray, [np.ndarray]])
            Always returns miscalibration metric.
            If 'return_map' is True, return tuple and append miscalibration map over all bins.
            If 'return_num_samples' is True, return tuple and append the number of samples in each bin (excluding confidence dimension).
            If 'return_uncertainty' is True, return tuple and append the average standard deviation of confidence within each bin (excluding confidence dimension).
        """

        # check if metric is correct set
        if not isinstance(metric, str):
            raise AttributeError('Parameter \'metric\' must be string \'ACE\', \'ECE\' or \'MCE\'.')
        if not metric.lower() in ['ace', 'ece', 'mce']:
            raise AttributeError('Parameter \'metric\' must be string \'ACE\', \'ECE\' or \'MCE\'.')
        else:
            metric = metric.lower()

        # prepare input data
        X, matched, sample_uncertainty, bin_bounds, _ = self.prepare(X, y, batched, uncertainty)

        # iterate over all batches of X and matched and calculate average miscalibration
        results = []
        for batch_X, batch_matched, batch_uncertainty, bounds in zip(X, matched, sample_uncertainty, bin_bounds):

            # perform binning on input arrays and drop last outcome (idx bin indices are not needed here)
            histograms = self.binning(bounds, batch_X, batch_matched, batch_X[:, 0], batch_uncertainty[:, 0])
            histograms = histograms[:-1]

            result = self.process(metric, *histograms)
            results.append(result)

        # finally, average over all batches
        miscalibration = np.mean([result[0] for result in results], axis=0)
        bin_map = np.mean([result[1] for result in results], axis=0)
        samples_map = np.mean([result[-1] for result in results], axis=0)
        uncertainty_map = np.sqrt(np.mean([result[-2] for result in results], axis=0))

        # build output structure w.r.t. user input
        if return_map or return_num_samples or return_uncertainty_map:
            return_value = (float(miscalibration),)

            if return_map:
                return_value = return_value + (bin_map,)
            if return_num_samples:
                return_value = return_value + (samples_map,)
            if return_uncertainty_map:
                return_value = return_value + (uncertainty_map,)

            return return_value
        else:
            return float(miscalibration)
