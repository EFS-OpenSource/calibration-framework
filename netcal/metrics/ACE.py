# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from netcal import accepts, dimensions


class ACE(object):
    """
    Metric for Average Calibration Error (ACE). This metrics measures the
    average difference between accuracy and confidence by grouping all samples into :math:`K` bins
    and calculating

    .. math::

       ACE = \\frac{1}{K} \\sum_{i=1}^K |\\text{acc}_i - \\text{conf}_i| ,

    where :math:`\\text{acc}_i` and :math:`\\text{conf}_i` denote the accuracy and average confidence in the i-th bin.
    The main difference to :class:`ECE` is that each bin is weighted equally.

    Parameters
    ----------
    bins : int
        Number of bins. The output space is partitioned into M equally sized bins.

    References
    ----------
    Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi:
    "Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection."
    Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.
    `Get source online <https://openreview.net/pdf?id=S1lG7aTnqQ>`_
    """

    @accepts(int)
    def __init__(self, bins: int):
        """
        Constructor.

        Parameters
        ----------
        bins : int
            Number of bins. The output space is partitioned into M equally sized bins.
        """
        self.bins = bins

    @dimensions((1, 2), (1, 2))
    def measure(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Measure calibration by given predictions with confidence and the according ground truth.
        Assume binary predictions with y=1.

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
        float
            Average Calibration Error (ACE).
        """

        # remove single-dimensional entries if present
        X = np.squeeze(X)
        y = np.squeeze(y)

        if y.size <= 0:
            raise ValueError("No samples provided.")
        elif len(y.shape) == 2:
            if y.shape[1] <= 2:
                y = y[:, -1]

        if len(X.shape) == 2:
            if X.shape[1] <= 2:
                prediction = np.ones(X.shape[0])
                X = X[:, -1]
            else:
                prediction = np.argmax(X, axis=1)
                X = np.max(X, axis=1)
        else:
            prediction = np.ones_like(X)

        # compute where prediction matches ground truth
        matched = np.array(prediction == y)

        # create bin bounds
        bin_boundaries = np.linspace(0.0, 1.0, self.bins + 1)

        # now calculate bin indices
        # this function gives the index for the upper bound of the according bin
        # for each sample. Thus, decrease by 1 to get the bin index
        current_indices = np.digitize(x=X, bins=bin_boundaries, right=True) - 1

        # if an index is out of bounds (e.g. 0), sort into first bin
        current_indices[current_indices == -1] = 0
        current_indices[current_indices == self.bins] = self.bins - 1

        ace = 0.0
        num_bins_not_empty = 0

        # mean accuracy is new confidence in each bin
        for bin in range(self.bins):
            bin_confidence = X[current_indices == bin]
            bin_matched = matched[current_indices == bin]

            if bin_confidence.size > 0:
                ace += np.abs(np.mean(bin_matched) - np.mean(bin_confidence))
                num_bins_not_empty += 1

        ace /= float(num_bins_not_empty)
        return ace
