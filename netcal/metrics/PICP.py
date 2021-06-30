# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, NamedTuple, Tuple, List
from collections import namedtuple
import numpy as np
from scipy.stats import norm
from typing import Union

from netcal import hpdi
from netcal.metrics import _Miscalibration


class PICP(_Miscalibration):
    """
    Compute Prediction Interval Coverage Probability (PICP) [1]_,[2]_ and Mean Prediction Interval Width (MPIW) [2]_.
    This metric is used for Bayesian models to determine the quality of the uncertainty estimates.
    In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability, that
    the true (observed) accuracy falls into the p% prediction interval. The uncertainty is well-calibrated, if
    the PICP is equal to p%. Simultaneously, the MPIW measures the mean width of all prediction intervals to evaluate
    the sharpness of the uncertainty estimates.

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
        Bins with an amount of samples below this threshold are not included into the process metrics.

    References
    ----------
    .. [1] Kuleshov, V.; Fenner, N. & Ermon, S.:
       "Accurate Uncertainties for Deep Learning Using Calibrated Regression."
       International Conference on Machine Learning (ICML), 2018
       `Get source online <http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf>`_

    .. [2] Jiayu  Yao,  Weiwei  Pan,  Soumya  Ghosh,  and  Finale  Doshi-Velez:
       "Quality of Uncertainty Quantification for Bayesian Neural Network Inference."
       Workshop on Uncertainty and Robustness in Deep Learning, ICML, 2019
       `Get source online <https://arxiv.org/pdf/1906.09686.pdf>`_
    """

    def accuracy(self, X: Union[Iterable[np.ndarray], np.ndarray],
                 y: Union[Iterable[np.ndarray], np.ndarray],
                 batched: bool = False,
                 uncertainty: str = 'mean') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List, int]:
        """ Measure the accuracy of each point by binning. """

        # prepare data - first, use "mean" to obtain the mean prediction of the posterior predictive
        # use flattened confidence estimates in order to get a better evaluation of the accuracy within each bin
        X_mean, _, sample_uncertainty, bin_bounds, num_features = self.prepare(X, y, batched, uncertainty=uncertainty)

        # convert mean variance to mean std deviation
        uncertainty = [np.sqrt(var) for var in sample_uncertainty]
        sample_accuracy, num_samples = [], []

        # for batch in zip(X_mean, X_flatten, matched_flatten, bin_bounds):
        for batch_X_mean, bounds in zip(X_mean, bin_bounds):

            # perform binning
            # acc_hist, _, _ = self.binning(bounds, batch_X_flatten, batch_matched_flatten)
            acc_hist, num_samples_hist, idx_mean = self.binning(bounds, batch_X_mean, batch_X_mean[:, 0], nan=np.nan)

            # use accuracy histogram from flattened estimates
            # and assign to "mean" values
            sample_accuracy.append(acc_hist[idx_mean])
            num_samples.append(num_samples_hist)

        return X_mean, sample_accuracy, uncertainty, num_samples, bin_bounds, num_features

    def measure(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray], p: float = 0.05,
                use_hpd: bool = True, batched: bool = False, uncertainty: str = 'mean',
                return_map: bool = False) -> NamedTuple:
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
        p : float, optional, default: 0.05
            Confidence level.
        use_hpd : bool, optional, default: True
            If True, use highest posterior density (HPD) interval to determine the prediction interval width.
            Use variance with Gaussian assumption otherwise.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.
        uncertainty : str, optional, default: "mean"
            Mode to measure mean estimate and uncertainty of the samples in Bayesian mode. Must be one
            of "mean" (mean of all samples), "mode" (mode of all samples), "median" (median of all samples) or
            "flatten" (no uncertainty will be computed, all samples are seen as independent predictions).
        return_map: bool, optional, default: False
            If True, return map with PICP and MPIW metric separated into all remaining dimension bins.

        Returns
        -------
        Namedtuple PICPResult with fields "picp" and "mpiw", where each field either holds the PICP/MPIW score
        or a tuple of (float, np.ndarray)
            Always returns a named tuple with PICP (prediction interval coverage probability) and MPIW
            (mean prediction interval width).
            If 'return_map' is True, each field holds a tuple for the metric itself and the PICP/MPIW distribution
            over all bins.
        """

        # PICP returns a namedtuple - init
        picpresult = namedtuple('PICPResult', ['picp', 'mpiw'])

        # 'prediction interval coverage probability' and 'mean prediction interval width'
        picp, picp_map = [], []
        mpiw, mpiw_map = [], []

        result = self.accuracy(X=X, y=y, batched=batched, uncertainty=uncertainty)

        # TODO: check different cases
        if not batched and isinstance(X, np.ndarray):
            X = [X]

        it = (X,) + tuple(result[:-1])

        # iterate over batches and get mean estimates
        for batch_X, batch_X_mean, batch_acc, batch_uncertainty, batch_num_samples, bounds, in zip(*it):

            batch_uncertainty = batch_uncertainty[:, 0]

            # if uncertainty is 0 everywhere (that is the case if no Bayesian method is evaluated)
            # return NaN
            if np.count_nonzero(batch_uncertainty) == 0:
                return picpresult(np.nan, np.nan)

            # TODO: do that more dynamically
            elif len(batch_X.shape) == 2:
                batch_X = np.expand_dims(batch_X, axis=-1)

            # remove all entries that are NaN
            nans = np.isnan(batch_acc)
            batch_X_mean, batch_acc, batch_uncertainty = batch_X_mean[~nans], batch_acc[~nans], batch_uncertainty[~nans]

            # since the output distributions might be skewed, using highed posterior density for interval
            # calculation should be preferred
            if use_hpd:

                # use confidence only for HPD
                interval_bounds = hpdi(batch_X[..., 0], 1.-p)
                lb_ci = interval_bounds[0, :][~nans]
                ub_ci = interval_bounds[1, :][~nans]

            else:

                # calculate prediction interval assuming a normal distribution
                # calculate credible interval
                z_score = norm.ppf(1. - (p / 2))
                batch_uncertainty = z_score * batch_uncertainty
                lb_ci = batch_X_mean[:, 0] - batch_uncertainty
                ub_ci = batch_X_mean[:, 0] + batch_uncertainty

            # use accuracy histogram from flattened estimates
            # and assign to "mean" values
            within_interval = np.where((batch_acc >= lb_ci) & (batch_acc <= ub_ci),
                                       np.ones_like(batch_acc), np.zeros_like(batch_acc))
            width = ub_ci - lb_ci

            # mean prediction interval width
            mpiw.append(np.mean(width))
            picp.append(np.mean(within_interval))

            # if a map along the "feature" bins is requested, use the binning routines of _Miscalibration class
            if return_map:
                # TODO: not that pretty
                threshold = int(self.sample_threshold)
                self.sample_threshold = 1
                batch_picp_map, batch_mpiw_map, _ = self.binning(bounds, batch_X_mean, within_interval, width)
                self.sample_threshold = threshold

                # after binning, reduce first dimension
                result = self.reduce(batch_picp_map, batch_num_samples, axis=0)
                batch_mpiw_map, _ = self.reduce(batch_mpiw_map, batch_num_samples, axis=0, reduce_result=result)

                mpiw_map.append(batch_mpiw_map)
                picp_map.append(result[0])

        picp = np.mean(picp)
        mpiw = np.mean(mpiw)

        if return_map:
            picp_map = np.mean(picp_map, axis=0)
            mpiw_map = np.mean(mpiw_map, axis=0)
            result = picpresult((picp, picp_map), (mpiw, mpiw_map))
        else:
            result = picpresult(picp, mpiw)

        return result
