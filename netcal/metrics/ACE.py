# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from typing import Union
from .Miscalibration import _Miscalibration
from netcal import dimensions


class ACE(_Miscalibration):
    """
    Average Calibration Error (ACE).
    This metric is used on classification [1]_ or as Detection Average Calibration Error (D-ACE)
    [2]_ on object detection tasks. This metrics measures the average difference between accuracy and confidence by
    grouping all samples into :math:`K` bins and calculating

    .. math::

       ACE = \\frac{1}{K} \\sum_{i=1}^K |\\text{acc}_i - \\text{conf}_i| ,

    where :math:`\\text{acc}_i` and :math:`\\text{conf}_i` denote the accuracy and average confidence in the i-th bin.
    The main difference to :class:`ECE` is that each bin is weighted equally.

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

    References
    ----------
    .. [1] Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi:
       "Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection."
       Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.
       `Get source online <https://openreview.net/pdf?id=S1lG7aTnqQ>`_
    .. [2] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    @dimensions((1, 2), (1, 2), None, None)
    def measure(self, X: np.ndarray, y: np.ndarray,
                return_map: bool = False, return_num_samples: bool = False) -> Union[np.ndarray, tuple]:
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
        return_map: bool, optional, default: False
            If True, return map with miscalibration metric separated into all remaining dimension bins.
        return_num_samples : bool, optional, default: False
            If True, also return the number of samples in each bin.

        Returns
        -------
        float or tuple of (float, np.ndarray) or tuple of (float, np.ndarray, np.ndarray)
            Always returns miscalibration metric.
            If 'return_map' is False, return ACE only (or num_samples map).
            If 'return_map' is True, return tuple with ACE and map over all bins.
            If 'return_num_samples' is False, ACE only (or ACE map).
            If 'return_num_samples' is True, return tuple with ACE and number of samples in each bin.
        """

        return self._measure(X=X, y=y, metric='ace', return_map=return_map, return_num_samples=return_num_samples)
