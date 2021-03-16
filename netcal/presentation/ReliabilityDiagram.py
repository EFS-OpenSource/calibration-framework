# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d, griddata
from typing import Union
from copy import deepcopy
import numpy as np
import logging

from netcal.metrics import _Miscalibration
from netcal import accepts


class ReliabilityDiagram(object):
    """
    Plot Confidence Histogram and Reliability Diagram to visualize miscalibration.
    On classification, plot the gaps between average confidence and observed accuracy bin-wise over the confidence
    space [1]_, [2]_.
    On detection, plot the miscalibration w.r.t. the additional regression information provided (1-D or 2-D) [3]_.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the ACE/ECE/MCE.
        On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    fmin : float, optional, default: None
        Minimum value for scale color.
    fmax : float, optional, default: None
        Maximum value for scale color.
    metric : str, default: 'ECE'
        Metric to measure miscalibration. Might be either 'ECE', 'ACE' or 'MCE'.
    title_suffix : str, optional, default: None
        Suffix for plot title.

    References
    ----------
    .. [1] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
       "On Calibration of Modern Neural Networks."
       Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
       `Get source online <https://arxiv.org/abs/1706.04599>`_
    .. [2] A. Niculescu-Mizil and R. Caruana:
       “Predicting good probabilities with supervised learning.”
       Proceedings of the 22nd International Conference on Machine Learning, 2005, pp. 625–632.
       `Get source online <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>`_
    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
    """

    @accepts((int, tuple, list), bool, (float, None), int, (list, None), (float, None), (str, None), (str, None))
    def __init__(self, bins: Union[int, tuple, list] = 10,
                 detection: bool = False, sample_threshold: int = 1,
                 feature_names: list = None,
                 fmin: float = None, fmax: float = None,
                 metric: str = 'ECE', title_suffix: str = None):
        """
        Constructor.

        Parameters
        ----------
        bins : int or iterable, default: 10
            Number of bins used by the ACE/ECE/MCE.
            On detection mode: if int, use same amount of bins for each dimension (nx1 = nx2 = ... = bins).
            If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
        detection : bool, default: False
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        sample_threshold : int, optional, default: 1
            If the number of samples in a bin is less than this threshold, do not use this bin for miscalibration
            computation.
        feature_names : list, length: [n_box_features-1] optional, default: None
            Names of the additional features (excluding confidence). These names are added to the according axes.
        fmin : float, optional, default: None
            Minimum value for scale color.
        fmax : float, optional, default: None
            Maximum value for scale color.
        metric : str, default: 'ECE'
            Metric to measure miscalibration. Might be either 'ECE', 'ACE' or 'MCE'.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        """

        self.bins = bins
        self.detection = detection
        self.sample_threshold = sample_threshold
        self.feature_names = feature_names
        self.fmin = fmin
        self.fmax = fmax
        self.metric = metric
        self.title_suffix = title_suffix

    def plot(self, X: Union[tuple, list, np.ndarray], y: Union[tuple, list, np.ndarray], batched: bool = False,
             filename: str = None, **save_args):
        """
        Reliability diagram to visualize miscalibration. This could be either in classical way for confidences only
        or w.r.t. additional properties (like x/y-coordinates of detection boxes, width, height, etc.). The additional
        properties get binned. Afterwards, the ECE will be calculated for each bin. This is visualized as (multiple)
        2-D plots.

        Parameters
        ----------
        X : iterable of np.ndarray, or np.ndarray of shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            If this is an iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : iterable of np.ndarray with same length as X or np.ndarray of shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            If iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function.

        Returns
        -------
        matplotlib.pyplot.Figure

        Raises
        ------
        AttributeError
            - If parameter metric is not string or string is not 'ACE', 'ECE' or 'MCE'
            - If parameter 'feature_names' is set but length does not fit to second dim of X
            - If no ground truth samples are provided
            - If length of bins parameter does not match the number of features given by X
            - If more than 3 feature dimensions (including confidence) are provided
        """

        # copy X and y for visualization as those variables might get modified
        X, y = deepcopy(X), deepcopy(y)

        # check if metric is correct
        if not isinstance(self.metric, str):
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')

        if self.metric.lower() not in ['ece', 'ace', 'mce']:
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')
        else:
            self.metric = self.metric.lower()

        # batched: interpret X and y as multiple predictions. Display average miscalibration map
        # if batch mode is not enabled, use
        if not batched:
            X, y = [X], [y]

        num_features = -1
        for i, (batch_X, batch_y) in enumerate(zip(X, y)):

            # we need at least 2 dimensions (for classification as well as for detection)
            if len(batch_X.shape) == 1:
                X[i] = np.reshape(batch_X, (-1, 1))

            # remove unnecessary dims if given
            y[i] = np.squeeze(batch_y)

            # check number of given samples
            if batch_y.size <= 0:
                raise ValueError("No samples provided.")

            # on detection mode, leave y array untouched
            elif len(batch_y.shape) == 2 and not self.detection:
                # still assume y as binary with ground truth labels present in y=1 entry
                if batch_y.shape[1] <= 2:
                    y[i] = batch_y[:, -1]

                # assume y as one-hot encoded
                else:
                    y[i] = np.argmax(batch_y, axis=1)

            batch_num_features = batch_X.shape[1] if self.detection and batch_X.ndim > 1 else 1
            # get number of additional dimensions (if not initialized)
            if num_features == -1:
                num_features = batch_num_features
            else:
                # if number of features is not equal over all instances, raise exception
                if num_features != batch_num_features:
                    raise ValueError("Unequal number of classes/features given in batched mode.")

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

        if self.feature_names is not None:
            if len(self.feature_names) != num_features-1:
                raise AttributeError("If attribute \'feature_names\' is set, the length must equal the number of additional properties given.")

        # no additional dimensions? compute standard reliability diagram
        if num_features == 1:
            fig = self.__plot_confidence_histogram(X, y)

        # one additional feature? compute 1D-plot
        elif num_features == 2:
            fig = self.__plot_1d(X, y)

        # two additional features? compute 2D plot
        elif num_features == 3:
            fig = self.__plot_2d(X, y)

        # number of dimensions exceeds 3? quit
        else:
            raise AttributeError("Diagram is not defined for more than 3 additional feature dimensions.")

        if filename is not None:
            fig.savefig(filename, **save_args)

        return fig

    @classmethod
    def __interpolate_grid(cls, metric_map: np.ndarray):
        """
        Interpolate missing values in a 2D-grid.

        Parameters
        ----------
        metric_map: np.ndarray
            Metric map computed by :class`_Miscalibration` class.

        Returns
        -------
        np.ndarray
            Interpolated 2D metric map
        """

        # get all NaNs
        nans = np.isnan(metric_map)
        x = lambda z: z.nonzero()

        # get mean of the remaining values and interpolate missing by the mean
        mean = float(np.mean(metric_map[~nans]))
        metric_map[nans] = griddata(x(~nans), metric_map[~nans], x(nans), method='cubic', fill_value=mean)
        return metric_map

    def __plot_confidence_histogram(self, X: list, y: list) -> plt.Figure:
        """
        Plot confidence histogram and reliability diagram to visualize miscalibration for condidences only.

        Parameters
        ----------
        X : list of np.ndarray, each with shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            List of NumPy arrays with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : list of np.ndarray, shape=(n_samples, [n_classes])
            List of NumPy arrays with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        matplotlib.pyplot.Figure
        """

        acc, conf, num_samples, mean_acc, mean_conf = [], [], [], [], []
        bounds = np.linspace(0., 1., self.bins[0] + 1)
        median_confidence = (bounds[1:] + bounds[:-1]) * 0.5

        for batch_X, batch_y in zip(X, y):
            if len(batch_X.shape) == 1:
                batch_X = np.reshape(batch_X, (-1, 1))
                prediction = np.ones(batch_X.shape[0])

            # got 2D array for X?
            elif len(batch_X.shape) == 2:

                # on detection mode, assume all predictions as 'matched'
                if self.detection:
                    prediction = np.ones(batch_X.shape[0])

                # on classification, if less than 2 entries for 2nd dimension are present, assume binary classification
                # (independent sigmoids are currently not supported)
                elif batch_X.shape[1] == 1:
                    prediction = np.ones(batch_X.shape[0])

                # classification and more than 1 entry for 2nd dimension? assume multiclass classification
                else:
                    prediction = np.argmax(batch_X, axis=1)
                    batch_X = np.reshape(np.max(batch_X, axis=1), (-1, 1))
            else:
                prediction = np.ones_like(batch_X)

            matched = prediction == batch_y

            # get binned statistic of average accuracy and confidence
            # as well as number of samples in each bin
            batch_acc, _, _ = binned_statistic(batch_X[:, 0], values=matched, statistic='mean',
                                               bins=self.bins[0], range=[[0.0, 1.0]])
            batch_conf, _, _ = binned_statistic(batch_X[:, 0], values=batch_X[:, 0], statistic='mean',
                                                bins=self.bins[0], range=[[0.0, 1.0]])
            batch_num_samples, _ = np.histogram(batch_X[:, 0], bins=self.bins[0], range=(0.0, 1.0))

            # identify all NaN indices
            nan_indices = np.nonzero(np.isnan(batch_acc))[0]

            # first dimension is confidence dimension - use the binning in this dimension to
            # determine median as fill values for empty bins
            batch_acc[nan_indices] = median_confidence[nan_indices]
            batch_conf[nan_indices] = median_confidence[nan_indices]

            # convert to relative amount of samples
            batch_num_samples = batch_num_samples / np.sum(batch_num_samples)

            # calculate overall mean accuracy and confidence
            batch_mean_acc = np.mean(batch_y)
            batch_mean_conf = np.mean(batch_X)

            # collect batched values
            acc.append(batch_acc)
            conf.append(batch_conf)
            num_samples.append(batch_num_samples)
            mean_acc.append(batch_mean_acc)
            mean_conf.append(batch_mean_conf)

        # calculate mean over batched values
        acc = np.mean(np.array(acc), axis=0)
        conf = np.mean(np.array(conf), axis=0)
        num_samples = np.mean(np.array(num_samples), axis=0)
        mean_acc = np.mean(np.array(mean_acc))
        mean_conf = np.mean(np.array(mean_conf))

        # calculate deviation
        deviation = conf - acc

        logger = logging.getLogger(__name__)
        logger.info("Average accuracy: %.4f - average confidence: %.4f" % (mean_acc, mean_conf))

        # -----------------------------------------
        # plot stuff
        fig = plt.figure()
        ax = plt.subplot(211)

        # set title suffix is given
        if self.title_suffix is not None:
            ax.set_title('Confidence Histogram - ' + self.title_suffix)
        else:
            ax.set_title('Confidence Histogram')

        # create bar chart with relative amount of samples in each bin
        # as well as average confidence and accuracy
        plt.bar(median_confidence, height=num_samples, width=1. / self.bins[0], align='center', edgecolor='black')
        plt.plot([mean_acc, mean_acc], [0.0, 1.0], color='black', linestyle='--')
        plt.plot([mean_conf, mean_conf], [0.0, 1.0], color='gray', linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        # labels and legend
        plt.xlabel('Confidence')
        plt.ylabel('% of Samples')
        plt.legend(['Avg. Accuracy', 'Avg. Confidence', 'Relative Amount of Samples'])

        # second plot: reliability histogram
        ax = plt.subplot(212)

        # set title suffix if given
        if self.title_suffix is not None:
            ax.set_title('Reliability Diagram' + " - " + self.title_suffix)
        else:
            ax.set_title('Reliability Diagram')

        # create two overlaying bar charts with bin accuracy and the gap of each bin to the perfect calibration
        plt.bar(median_confidence, height=acc, width=1./self.bins[0], align='center', edgecolor='black')
        plt.bar(median_confidence, height=deviation, bottom=acc, width=1./self.bins[0], align='center',
                edgecolor='black', color='red', alpha=0.6)

        # draw diagonal as perfect calibration line
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        # labels and legend of second plot
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.legend(['Perfect Calibration', 'Actual', 'Gap'])

        plt.tight_layout()
        return fig

    def __plot_1d(self, X: list, y: list) -> plt.Figure:
        """
        Plot 1-D miscalibration w.r.t. one additional feature.

        Parameters
        ----------
        X : list of np.ndarray, each with shape=(n_samples, [n_box_features])
            List of NumPy arrays with confidence values for each prediction and and additional box features in last dim.
        y : list of np.ndarray, each with shape=(n_samples, [n_classes])
            List of NumPy arrays with ground truth labels as label vector (1-D).

        Returns
        -------
        matplotlib.pyplot.Figure
        """

        # iterate over all given models and build mean confidence and accuracy
        acc, edge_acc, conf, edge_conf, miscalibration_map = [], [], [], [], []

        # miscalibration object is used to get metric map
        miscalibration = _Miscalibration(bins=self.bins, detection=self.detection,
                                         sample_threshold=self.sample_threshold)

        for batch_X, batch_y in zip(X, y):
            # get miscalibration w.r.t. to given feature
            # get binned statistic of average accuracy and confidence w.r.t. binning by additional feature
            batch_acc, batch_edge_acc, _ = binned_statistic(batch_X[:, -1], values=batch_y,
                                                            statistic='mean', bins=self.bins[-1],
                                                            range=[[0.0, 1.0]])
            batch_conf, batch_edge_conf, _ = binned_statistic(batch_X[:, -1], values=batch_X[:, 0],
                                                              statistic='mean', bins=self.bins[-1],
                                                              range=[[0.0, 1.0]])
            _, batch_miscal = miscalibration._measure(batch_X, batch_y, metric=self.metric,
                                                      return_map=True, return_num_samples=False)
            miscalibration_map.append(batch_miscal)

            # set empty bins to 0
            # TODO: mark those ranges with a gray box
            batch_acc[np.isnan(batch_acc)] = 0.0
            batch_conf[np.isnan(batch_conf)] = 0.0

            # correct binning indices
            batch_edge_acc = (batch_edge_acc[:-1] + batch_edge_acc[1:]) * 0.5
            batch_edge_conf = (batch_edge_conf[:-1] + batch_edge_conf[1:]) * 0.5

            # append to global variables
            acc.append(batch_acc)
            edge_acc.append(batch_edge_acc)
            conf.append(batch_conf)
            edge_conf.append(batch_edge_conf)

        # calculate mean over all given instances
        acc = np.mean(np.array(acc), axis=0)
        edge_acc = np.mean(np.array(edge_acc), axis=0)
        conf = np.mean(np.array(conf), axis=0)
        edge_conf = np.mean(np.array(edge_conf), axis=0)
        miscalibration_map = np.mean(np.array(miscalibration_map), axis=0)

        # interpolate missing values
        x = np.linspace(0.0, 1.0, 1000)
        acc = interp1d(edge_acc, acc, kind='cubic', fill_value='extrapolate')(x)
        conf = interp1d(edge_conf, conf, kind='cubic', fill_value='extrapolate')(x)
        miscalibration_map = interp1d(edge_conf, miscalibration_map, kind='cubic', fill_value='extrapolate')(x)

        # draw routines
        fig, ax1 = plt.subplots()
        color = 'tab:blue'

        # set name of the additional feature
        if self.feature_names is not None:
            ax1.set_xlabel(self.feature_names[0])

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.set_ylabel('accuracy/confidence', color=color)

        # draw confidence and accuracy on the same (left) axis
        line1, = ax1.plot(x, acc, '-.', color='black')
        line2, = ax1.plot(x, conf, '--', color=color)
        ax1.tick_params('y', labelcolor=color)

        # create second axis for ECE
        ax11 = ax1.twinx()
        color = 'tab:red'
        line3, = ax11.plot(x, miscalibration_map, '-', color=color)

        ax11.set_ylabel('Expected Calibration Error (ECE)', color=color)
        ax11.tick_params('y', labelcolor=color)

        # set ECE limits if given
        if self.fmin is not None and self.fmax is not None:
            ax11.set_ylim([self.fmin, self.fmax])

        ax1.legend((line1, line2, line3), ('accuracy', 'confidence', 'ECE'), loc='lower left')

        if self.title_suffix is not None:
            ax1.set_title('Accuracy, confidence and ECE depending on cx coordinate\n- %s -' % self.title_suffix)
        else:
            ax1.set_title('Accuracy, confidence and ECE depending on cx coordinate')

        ax1.grid(True)

        fig.tight_layout()
        return fig

    def __plot_2d(self, X: list, y: list) -> plt.Figure:
        """
        Plot 2D miscalibration reliability diagram heatmap.

        Parameters
        ----------
        X : list of np.ndarray, each with shape=(n_samples, [n_box_features])
            List of NumPy arrays with confidence values for each prediction and and additional box features in last dim.
        y : list of np.ndarray, each with shape=(n_samples, [n_classes])
            List of NumPy array with ground truth labels as label vector (1-D).

        Returns
        -------
        matplotlib.pyplot.Figure
        """

        # miscalibration object is used to get metric map
        miscalibration = _Miscalibration(bins=self.bins, detection=self.detection,
                                         sample_threshold=self.sample_threshold)

        metric_map = []
        for batch_X, batch_y in zip(X, y):
            batch_miscal, batch_metric_map, batch_num_samples_map = miscalibration._measure(batch_X, batch_y,
                                                                                            metric=self.metric,
                                                                                            return_map=True,
                                                                                            return_num_samples=True)

            # on 2D (3 dimensions including confidence), use grid interpolation
            # set missing entries to NaN and interpolate
            batch_metric_map[batch_num_samples_map == 0.0] = np.nan
            batch_metric_map = self.__interpolate_grid(batch_metric_map)

            metric_map.append(batch_metric_map)

        # calculate mean miscalibration along all metric maps
        metric_map = np.mean(np.array(metric_map), axis=0)

        # transpose is necessary. Miscalibration is calculated in the order given by the features
        # however, imshow expects arrays in format [rows, columns] or [height, width]
        # e.g., miscalibration with additional x/y (in this order) will be drawn [y, x] otherwise
        metric_map = metric_map.T

        # draw routines
        fig, _ = plt.subplots()
        plt.imshow(metric_map, origin='lower', interpolation="gaussian", cmap='jet', aspect=1, vmin=self.fmin, vmax=self.fmax)

        # set correct x- and y-ticks
        plt.xticks(np.linspace(0., self.bins[1] - 1, 5), np.linspace(0., 1., 5))
        plt.yticks(np.linspace(0., self.bins[2] - 1, 5), np.linspace(0., 1., 5))
        plt.xlim([0.0, self.bins[1] - 1])
        plt.ylim([0.0, self.bins[2] - 1])

        # draw feature names on axes if given
        if self.feature_names is not None:
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])

        plt.colorbar()

        # draw title if given
        if self.title_suffix is not None:
            plt.title("ECE depending on cx/cy coordinates\n- %s -" % self.title_suffix)
        else:
            plt.title("ECE depending on cx/cy coordinates")

        return fig
