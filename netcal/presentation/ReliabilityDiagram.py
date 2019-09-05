# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
# 
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import matplotlib.pyplot as plt
import numpy as np
import logging

from netcal import dimensions, accepts


class ReliabilityDiagram(object):
    """
    Plot Confidence Histogram and Reliability Diagram to visualize miscalibration.

    Parameters
    ----------
    bins : int
        Number of bins. The output space is partitioned into M equally sized bins.

    References
    ----------
    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
    "On Calibration of Modern Neural Networks."
    arXiv (abs/1706.04599), 2017.
    `Get source online <https://arxiv.org/abs/1706.04599>`_

    A. Niculescu-Mizil and R. Caruana:
    “Predicting good probabilities with supervised learning.”
    Proceedings of the 22nd International Conference on Machine Learning, 2005, pp. 625–632.
    `Get source online <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>`_
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
        self.logger = logging.getLogger('calibration')

    @dimensions((1, 2), 1, None)
    def plot(self, X: np.ndarray, y: np.ndarray, title_suffix: str = None):
        """
        Plot confidence histogram and reliability diagram to visualize miscalibration.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        """

        plt.figure(figsize=(7, 8))

        bins = self.bins
        num_samples = y.size

        title = "Reliability Diagram"

        # -----------------------------------------
        # get prediction labels and sort arrays

        # binary classification problem but got two entries? (probability for 0 and 1 separately)?
        # we only need probability p for Y=1 (probability for 0 is (1-p) )
        if len(X.shape) == 1:

            # first, get right binary predictions (y=0 or y=1)
            predictions = np.where(X > 0.5, 1, 0)

            # calculate average accuracy and average confidence
            total_accuracy = np.equal(predictions, y).sum() / num_samples

            prediction_confidence = np.where(X > 0.5, X, 1. - X)
            total_confidence = np.sum(prediction_confidence) / num_samples

            # plot confidence estimates only for y=1
            # thus, set predictions to 1 for each sample
            predictions = np.ones_like(X)

            title += " for y=1"

        else:

            predictions = np.argmax(X, axis=1)
            X = np.max(X, axis=1)

            # calculate average accuracy and average confidence
            total_accuracy = np.equal(predictions, y).sum() / num_samples
            total_confidence = np.sum(X) / num_samples

        # -----------------------------------------

        # prepare visualization metrics
        bin_confidence = np.zeros(bins)
        bin_accuracy = np.zeros(bins)
        bin_gap = np.zeros(bins)
        bin_samples = np.zeros(bins)
        bin_color = ['blue', ] * bins

        # iterate over bins, get avg accuracy and confidence of each bin and add to ECE
        for i in range(1, bins+1):

            # get lower and upper boundaries
            low = (i - 1) / float(bins)
            high = i / float(bins)
            condition = (X > low) & (X <= high)

            num_samples_bin = condition.sum()
            if num_samples_bin <= 0:
                bin_confidence[i-1] = bin_accuracy[i-1] = (high + low) / 2.0
                bin_color[i-1] = "yellow"
                continue

            # calc avg confidence and accuracy
            right_predictions_bin = np.equal(predictions[condition], y[condition]).sum()
            avg_accuracy = right_predictions_bin / float(num_samples_bin)

            bin_confidence[i-1] = (high + low) / 2.0
            bin_accuracy[i-1] = avg_accuracy
            bin_gap[i-1] = bin_confidence[i-1] - avg_accuracy
            bin_samples[i-1] = num_samples_bin

        bin_samples /= num_samples

        self.logger.info("Average accuracy: %.4f - average confidence: %.4f" % (total_accuracy, total_confidence))

        # -----------------------------------------
        # plot stuff
        ax = plt.subplot(211)

        if title_suffix is not None:
            ax.set_title('Confidence Histogram - ' + title_suffix)
        else:
            ax.set_title('Confidence Histogram')

        plt.bar(bin_confidence, height=bin_samples, width=1. / bins, align='center', edgecolor='black')
        plt.plot([total_accuracy, total_accuracy], [0.0, 1.0], color='black', linestyle='--')
        plt.plot([total_confidence, total_confidence], [0.0, 1.0], color='gray', linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        plt.xlabel('Confidence')
        plt.ylabel('% of Samples')

        plt.legend(['Avg. Accuracy', 'Avg. Confidence', 'Relative Amount of Samples'])

        ax = plt.subplot(212)

        if title_suffix is not None:
            ax.set_title(title + " - " + title_suffix)
        else:
            ax.set_title(title)

        plt.bar(bin_confidence, height=bin_accuracy, width=1./bins, align='center', edgecolor='black')
        plt.bar(bin_confidence, height=bin_gap, bottom=bin_accuracy, width=1./bins, align='center',
                edgecolor='black', color='red', alpha=0.6)

        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        plt.legend(['Perfect Calibration', 'Output', 'Gap'])
        plt.tight_layout()

        plt.show()
