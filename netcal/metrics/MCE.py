# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union, Iterable, Tuple
import numpy as np
from .Miscalibration import _Miscalibration


class MCE(_Miscalibration):
    """
    Maximum Calibration Error (MCE).
    This metric is used on classification [1]_ or as Detection Maximum calibration error (D-MCE) on
    object detection [2]_. This metrics measures the maximum difference between accuracy and confidence over all bins
    by grouping all samples into :math:`K` bins and calculating

    .. math::

       MCE = \max_{i \\in \\{1, ..., K\\}} |\\text{acc}_i - \\text{conf}_i| ,

    where :math:`\\text{acc}_i` and :math:`\\text{conf}_i` denote the accuracy and average confidence in the i-th bin.

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
    .. [2] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.
       `Get source online <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf>`_
    """

    def measure(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray],
                batched: bool = False, uncertainty: str = None,
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
        return_map: bool, optional, default: False
            If True, return map with miscalibration metric separated into all remaining dimension bins.
        return_num_samples : bool, optional, default: False
            If True, also return the number of samples in each bin.
        return_uncertainty_map : bool, optional, default: False
            If True, also return the average deviation of the confidence within each bin.

        Returns
        -------
        float or tuple of (float, np.ndarray, [np.ndarray, [np.ndarray]])
            Always returns Maximum Calibration Error.
            If 'return_map' is True, return tuple and append miscalibration map over all bins.
            If 'return_num_samples' is True, return tuple and append the number of samples in each bin (excluding confidence dimension).
            If 'return_uncertainty' is True, return tuple and append the average standard deviation of confidence within each bin (excluding confidence dimension).
        """

        return self._measure(X=X, y=y, metric='mce',
                             batched=batched, uncertainty=uncertainty,
                             return_map=return_map,
                             return_num_samples=return_num_samples,
                             return_uncertainty_map=return_uncertainty_map)
