# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union, Iterable, List
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d, griddata

import matplotlib.pyplot as plt
import tikzplotlib

from netcal.metrics import _Miscalibration


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
    equal_intervals : bool, optional, default: True
        If True, the bins have the same width. If False, the bins are splitted to equalize
        the number of samples in each bin.
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
       The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.
       `Get source online <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf>`_
    """

    def __init__(self, bins: Union[int, Iterable[int]] = 10, equal_intervals: bool = True,
                 detection: bool = False, sample_threshold: int = 1,
                 fmin: float = None, fmax: float = None,
                 metric: str = 'ECE', **kwargs):
        """ Constructor. For detailed parameter documentation view classdocs. """

        self.bins = bins
        self.detection = detection
        self.sample_threshold = sample_threshold
        self.fmin = fmin
        self.fmax = fmax
        self.metric = metric

        if 'feature_names' in kwargs:
            self.feature_names = kwargs['feature_names']

        if 'title_suffix' in kwargs:
            self.title_suffix = kwargs['title_suffix']

        self._miscalibration = _Miscalibration(bins=bins, equal_intervals=equal_intervals,
                                               detection=detection, sample_threshold=sample_threshold)

    def plot(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray],
             batched: bool = False, uncertainty: str = None, filename: str = None, tikz: bool = False,
             title_suffix: str = None, feature_names: List[str] = None, **save_args) -> Union[plt.Figure, str]:
        """
        Reliability diagram to visualize miscalibration. This could be either in classical way for confidences only
        or w.r.t. additional properties (like x/y-coordinates of detection boxes, width, height, etc.). The additional
        properties get binned. Afterwards, the miscalibration will be calculated for each bin. This is
        visualized as a 2-D plots.

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
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        feature_names : list, optional, default: None
            Names of the additional features that are attached to the axes of a reliability diagram.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.

        Raises
        ------
        AttributeError
            - If parameter metric is not string or string is not 'ACE', 'ECE' or 'MCE'
            - If parameter 'feature_names' is set but length does not fit to second dim of X
            - If no ground truth samples are provided
            - If length of bins parameter does not match the number of features given by X
            - If more than 3 feature dimensions (including confidence) are provided
        """

        # assign deprecated constructor parameter to title_suffix and feature_names
        if hasattr(self, 'title_suffix') and title_suffix is None:
            title_suffix = self.title_suffix

        if hasattr(self, 'feature_names') and feature_names is None:
            feature_names = self.feature_names

        # check if metric is correct
        if not isinstance(self.metric, str):
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')

        # check metrics parameter
        if self.metric.lower() not in ['ece', 'ace', 'mce']:
            raise AttributeError('Parameter \'metric\' must be string with either \'ece\', \'ace\' or \'mce\'.')
        else:
            self.metric = self.metric.lower()

        # perform checks and prepare input data
        X, matched, sample_uncertainty, bin_bounds, num_features = self._miscalibration.prepare(X, y, batched, uncertainty)
        if num_features > 3:
            raise AttributeError("Diagram is not defined for more than 2 additional feature dimensions.")

        histograms = []
        for batch_X, batch_matched, batch_uncertainty, bounds in zip(X, matched, sample_uncertainty, bin_bounds):
            batch_histograms = self._miscalibration.binning(bounds, batch_X, batch_matched, batch_X[:, 0], batch_uncertainty[:, 0])
            histograms.append(batch_histograms[:-1])

        # no additional dimensions? compute standard reliability diagram
        if num_features == 1:
            fig = self.__plot_confidence_histogram(X, matched, histograms, bin_bounds, title_suffix)

        # one additional feature? compute 1D-plot
        elif num_features == 2:
            fig = self.__plot_1d(histograms, bin_bounds, title_suffix, feature_names)

        # two additional features? compute 2D plot
        elif num_features == 3:
            fig = self.__plot_2d(histograms, bin_bounds, title_suffix, feature_names)

        # number of dimensions exceeds 3? quit
        else:
            raise AttributeError("Diagram is not defined for more than 2 additional feature dimensions.")

        # if tikz is true, create tikz code from matplotlib figure
        if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            plt.close(fig)
            fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig

    @classmethod
    def __interpolate_grid(cls, metric_map: np.ndarray) -> np.ndarray:
        """ Interpolate missing values in a 2D-grid using the mean of the data. The interpolation is done inplace. """

        # get all NaNs
        nans = np.isnan(metric_map)
        x = lambda z: z.nonzero()

        # get mean of the remaining values and interpolate missing by the mean
        mean = float(np.mean(metric_map[~nans]))
        metric_map[nans] = griddata(x(~nans), metric_map[~nans], x(nans), method='cubic', fill_value=mean)
        return metric_map

    def __plot_confidence_histogram(self, X: List[np.ndarray], matched: List[np.ndarray], histograms: List[np.ndarray],
                                    bin_bounds: List, title_suffix: str = None) -> plt.Figure:
        """ Plot confidence histogram and reliability diagram to visualize miscalibration for condidences only. """

        # get number of bins (self.bins has not been processed yet)
        n_bins = len(bin_bounds[0][0])-1

        median_confidence = [(bounds[0][1:] + bounds[0][:-1]) * 0.5 for bounds in bin_bounds]
        mean_acc, mean_conf = [], []
        for batch_X, batch_matched, batch_hist, batch_median in zip(X, matched, histograms, median_confidence):
            acc_hist, conf_hist, _, num_samples_hist = batch_hist
            empty_bins, = np.nonzero(num_samples_hist == 0)

            # calculate overall mean accuracy and confidence
            mean_acc.append(np.mean(batch_matched))
            mean_conf.append(np.mean(batch_X))

            # set empty bins to median bin value
            acc_hist[empty_bins] = batch_median[empty_bins]
            conf_hist[empty_bins] = batch_median[empty_bins]

            # convert num_samples to relative afterwards (inplace denoted by [:])
            num_samples_hist[:] = num_samples_hist / np.sum(num_samples_hist)

        # get mean histograms and values over all batches
        acc = np.mean([hist[0] for hist in histograms], axis=0)
        conf = np.mean([hist[1] for hist in histograms], axis=0)
        uncertainty = np.sqrt(np.mean([hist[2] for hist in histograms], axis=0))
        num_samples = np.mean([hist[3] for hist in histograms], axis=0)
        mean_acc = np.mean(mean_acc)
        mean_conf = np.mean(mean_conf)
        median_confidence = np.mean(median_confidence, axis=0)
        bar_width = np.mean([np.diff(bounds[0]) for bounds in bin_bounds], axis=0)

        # compute credible interval of uncertainty
        p = 0.05
        z_score = norm.ppf(1. - (p / 2))
        uncertainty = z_score * uncertainty

        # if no uncertainty is given, set variable uncertainty to None in order to prevent drawing error bars
        if np.count_nonzero(uncertainty) == 0:
            uncertainty = None

        # calculate deviation
        deviation = conf - acc

        # -----------------------------------------
        # plot data distribution histogram first
        fig, axes = plt.subplots(2, squeeze=True, figsize=(7, 6))
        ax = axes[0]

        # set title suffix is given
        if title_suffix is not None:
            ax.set_title('Confidence Histogram - ' + title_suffix)
        else:
            ax.set_title('Confidence Histogram')

        # create bar chart with relative amount of samples in each bin
        # as well as average confidence and accuracy
        ax.bar(median_confidence, height=num_samples, width=bar_width, align='center', edgecolor='black')
        ax.plot([mean_acc, mean_acc], [0.0, 1.0], color='black', linestyle='--')
        ax.plot([mean_conf, mean_conf], [0.0, 1.0], color='gray', linestyle='--')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

        # labels and legend
        ax.set_xlabel('Confidence')
        ax.set_ylabel('% of Samples')
        ax.legend(['Avg. Accuracy', 'Avg. Confidence', 'Relative Amount of Samples'])

        # second plot: reliability histogram
        ax = axes[1]

        # set title suffix if given
        if title_suffix is not None:
            ax.set_title('Reliability Diagram' + " - " + title_suffix)
        else:
            ax.set_title('Reliability Diagram')

        # create two overlaying bar charts with bin accuracy and the gap of each bin to the perfect calibration
        ax.bar(median_confidence, height=acc, width=bar_width, align='center',
               edgecolor='black', yerr=uncertainty, capsize=4)
        ax.bar(median_confidence, height=deviation, bottom=acc, width=bar_width, align='center',
               edgecolor='black', color='red', alpha=0.6)

        # draw diagonal as perfect calibration line
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

        # labels and legend of second plot
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend(['Perfect Calibration', 'Output', 'Gap'])

        plt.tight_layout()
        return fig

    def __plot_1d(self, histograms: List[np.ndarray], bin_bounds: List,
                  title_suffix: str = None, feature_names: List[str] = None) -> plt.Figure:
        """ Plot 1-D miscalibration w.r.t. one additional feature. """

        # z score for credible interval (if uncertainty is given)
        p = 0.05
        z_score = norm.ppf(1. - (p / 2))

        results = []
        for batch_hist, bounds in zip(histograms, bin_bounds):
            result = self._miscalibration.process(self.metric, *batch_hist)
            bin_median = (bounds[-1][:-1] + bounds[-1][1:]) * 0.5

            # interpolate missing values
            x = np.linspace(0.0, 1.0, 1000)
            miscalibration = interp1d(bin_median, result[1], kind='cubic', fill_value='extrapolate')(x)
            acc = interp1d(bin_median, result[2], kind='cubic', fill_value='extrapolate')(x)
            conf = interp1d(bin_median, result[3], kind='cubic', fill_value='extrapolate')(x)
            uncertainty = interp1d(bin_median, result[4], kind='cubic', fill_value='extrapolate')(x)

            results.append((miscalibration, acc, conf, uncertainty))

        # get mean over all batches and convert mean variance to a std deviation afterwards
        miscalibration = np.mean([result[0] for result in results], axis=0)
        acc = np.mean([result[1] for result in results], axis=0)
        conf = np.mean([result[2] for result in results], axis=0)
        uncertainty = np.sqrt(np.mean([result[3] for result in results], axis=0))

        # draw routines
        fig, ax1 = plt.subplots()
        conf_color = 'tab:blue'

        # set name of the additional feature
        if feature_names is not None:
            ax1.set_xlabel(feature_names[0])

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.set_ylabel('accuracy/confidence', color=conf_color)

        # draw confidence and accuracy on the same (left) axis
        x = np.linspace(0.0, 1.0, 1000)
        line1, = ax1.plot(x, acc, '-.', color='black')
        line2, = ax1.plot(x, conf, '--', color=conf_color)
        ax1.tick_params('y', labelcolor=conf_color)

        # if uncertainty is given, compute average of variances over all bins and get std deviation by sqrt
        # compute credible interval afterwards
        # define lower and upper bound
        uncertainty = z_score * uncertainty
        lb = conf - uncertainty
        ub = conf + uncertainty

        # create second axis for miscalibration
        ax11 = ax1.twinx()
        miscal_color = 'tab:red'
        line3, = ax11.plot(x, miscalibration, '-', color=miscal_color)

        if self.metric == 'ace':
            ax11.set_ylabel('Average Calibration Error (ACE)', color=miscal_color)
        elif self.metric == 'ece':
            ax11.set_ylabel('Expected Calibration Error (ECE)', color=miscal_color)
        elif self.metric == 'mce':
            ax11.set_ylabel('Maximum Calibration Error (MCE)', color=miscal_color)

        ax11.tick_params('y', labelcolor=miscal_color)

        # set miscalibration limits if given
        if self.fmin is not None and self.fmax is not None:
            ax11.set_ylim([self.fmin, self.fmax])

        ax1.legend((line1, line2, line3),
                   ('accuracy', 'confidence', '%s' % self.metric.upper()),
                   loc='best')

        if title_suffix is not None:
            ax1.set_title('Accuracy, confidence and %s\n- %s -' % (self.metric.upper(), title_suffix))
        else:
            ax1.set_title('Accuracy, confidence and %s' % self.metric.upper())

        ax1.grid(True)

        fig.tight_layout()
        return fig

    def __plot_2d(self, histograms: List[np.ndarray], bin_bounds: List[np.ndarray],
                  title_suffix: str = None, feature_names: List[str] = None) -> plt.Figure:
        """ Plot 2D miscalibration reliability diagram heatmap. """

        results = []
        for batch_hist in histograms:
            result = self._miscalibration.process(self.metric, *batch_hist)

            # interpolate 2D data inplace to avoid "empty" bins
            batch_samples = result[-1]
            for map in result[1:-1]:
                map[batch_samples == 0.0] = 0.0
                # TODO: check what to do here
                # map[batch_samples == 0.0] = np.nan
                # self.__interpolate_grid(map)

            # on interpolation, it is sometimes possible that empty bins have negative values
            # however, this is invalid for variance
            result[4][result[4] < 0] = 0.0
            results.append(result)

        # calculate mean over all batches and transpose
        # transpose is necessary. Miscalibration is calculated in the order given by the features
        # however, imshow expects arrays in format [rows, columns] or [height, width]
        # e.g., miscalibration with additional x/y (in this order) will be drawn [y, x] otherwise
        miscalibration = np.mean([result[1] for result in results], axis=0).T
        acc = np.mean([result[2] for result in results], axis=0).T
        conf = np.mean([result[3] for result in results], axis=0).T
        mean = np.mean([result[4] for result in results], axis=0).T
        uncertainty = np.sqrt(mean)

        # -----------------------------------------------------------------------------------------
        # draw routines

        def set_axis(ax, map, vmin=None, vmax=None):
            """ Generic function to set all subplots equally """
            # TODO: set proper fmin, fmax values
            img = ax.imshow(map, origin='lower', interpolation="gaussian", cmap='jet', aspect=1, vmin=vmin, vmax=vmax)

            # set correct x- and y-ticks
            ax.set_xticks(np.linspace(0., len(bin_bounds[0][1])-2, 5))
            ax.set_xticklabels(np.linspace(0., 1., 5))
            ax.set_yticks(np.linspace(0., len(bin_bounds[0][2])-2, 5))
            ax.set_yticklabels(np.linspace(0., 1., 5))
            ax.set_xlim([0.0, len(bin_bounds[0][1])-2])
            ax.set_ylim([0.0, len(bin_bounds[0][2])-2])

            # draw feature names on axes if given
            if feature_names is not None:
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])

            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

            return ax, img

        # -----------------------------------

        # create only two subplots if no additional uncertainty is given
        if np.count_nonzero(uncertainty) == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # process additional uncertainty if given
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, squeeze=True, figsize=(10, 10))
            ax4, img4 = set_axis(ax4, uncertainty)

            if title_suffix is not None:
                ax4.set_title("Confidence std deviation\n- %s -" % title_suffix)
            else:
                ax4.set_title("Confidence std deviation")

        ax1, img1 = set_axis(ax1, acc, vmin=0, vmax=1)
        ax2, img2 = set_axis(ax2, conf, vmin=0, vmax=1)
        ax3, img3 = set_axis(ax3, miscalibration, vmin=self.fmin, vmax=self.fmax)

        # draw title if given
        if title_suffix is not None:
            ax1.set_title("Average accuracy\n- %s -" % title_suffix)
            ax2.set_title("Average confidence\n- %s -" % title_suffix)
            ax3.set_title("%s\n- %s -" % (self.metric.upper(), title_suffix))
        else:
            ax1.set_title("Average accuracy")
            ax2.set_title("Average confidence")
            ax3.set_title("%s" % self.metric.upper())

        # -----------------------------------------------------------------------------------------

        return fig
