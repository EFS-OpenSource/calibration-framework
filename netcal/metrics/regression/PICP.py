# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, NamedTuple
from collections import namedtuple
import numpy as np
from typing import Union

from netcal import is_in_quantile
from netcal.metrics.Miscalibration import _Miscalibration


class PICP(_Miscalibration):
    """
    Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    These metrics have been proposed by [1]_, [2]_.
    This metric is used for Bayesian models to determine the quality of the uncertainty estimates.
    In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability, that
    the true (observed) accuracy falls into the p% prediction interval. The uncertainty is well-calibrated, if
    the PICP is equal to p%. Simultaneously, the MPIW measures the mean width of all prediction intervals to evaluate
    the sharpness of the uncertainty estimates.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the PICP.
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
       `Get source online <http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf>`__

    .. [2] Jiayu  Yao,  Weiwei  Pan,  Soumya  Ghosh,  and  Finale  Doshi-Velez:
       "Quality of Uncertainty Quantification for Bayesian Neural Network Inference."
       Workshop on Uncertainty and Robustness in Deep Learning, ICML, 2019
       `Get source online <https://arxiv.org/pdf/1906.09686.pdf>`__
    """

    def measure(
            self,
            X: Union[Iterable[np.ndarray], np.ndarray],
            y: Union[Iterable[np.ndarray], np.ndarray],
            q: Union[float, Iterable[float], np.ndarray],
            *,
            kind: str = 'meanstd',
            reduction: str = 'batchmean',
    ) -> NamedTuple:
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
        q : np.ndarray of shape (q,)
            Quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                       for each sample. In this case, the NLL is calculated for multivariate distributions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
            - confidence: asserts X as a single NumPy array of shape (t, n, [1]) with t stochastic forward passes with
                          scores in [0, 1] that represent confidence scores obtained e.g. by Monte-Carlo sampling.
                          Furthermore, this mode asserts the ground-truth labels 'y' in the {0, 1} set and converts
                          them to continuous [0, 1] scores by binning. Thus, it is possible to evaluate the
                          confidence uncertainty with binned labels.
        reduction : str, one of 'none', 'mean' or 'batchmean', default: 'batchmean'
            Specifies the reduction to apply to the output:
            - none : no reduction is performed. Return QCE for each sample and for each dim separately.
            - mean : calculate mean over all quantiles and all dimensions.
            - batchmean : calculate mean over all quantiles but for each dim separately.
                          If input has covariance matrices, 'batchmean' is the same as 'mean'.

        Returns
        -------
        Namedtuple PICPResult with fields "picp" and "mpiw", where each field either holds the PICP/MPIW score
        or a tuple of (float, np.ndarray)
            Always returns a named tuple with PICP (prediction interval coverage probability) and MPIW
            (mean prediction interval width).
        """

        # PICP returns a namedtuple - init
        picpresult = namedtuple('PICPResult', ['picp', 'mpiw'])

        # kind 'confidence' induces a different treatment - the target scores given by 'y' are assumed
        # to be in the {0, 1} set and are converted to continuous [0, 1] scores by binning the samples.
        # Thus, it is possible to evaluate the confidence uncertainty
        if kind == "confidence":
            result = self.frequency(X=X, y=y, batched=False, uncertainty="mean")

            # frequency is 2nd entry in return tuple and within first batch
            X = X[..., :1]  # (t, n, 1)
            y = np.reshape(result[1][0], (-1, 1))  # (n, 1)

            # kind 'meanstd' will result in a HPDI computation if sample distribution is given
            kind = "meanstd"

        in_quantile, _, (qbounds_min, qbounds_max), _, _ = is_in_quantile(X, y, q, kind)  # (q, n, [d]), (q, n, d), (q, n, d),

        picp = np.mean(in_quantile, axis=1)  # (q, d)
        mpiw = np.mean(np.abs(qbounds_max - qbounds_min), axis=1)  # (q, d)

        # no reduction is applied
        if reduction is None or reduction == 'none':
            pass

        # 'mean' is mean over all quantiles and all dimensions
        elif reduction == "mean":
            picp = np.mean(picp)
            mpiw = np.mean(mpiw)

        # 'batchmean' is mean over all quantiles but for each dim separately.
        # If input has covariance matrices, 'batchmean' is the same as 'mean'.
        elif reduction == "batchmean":
            picp = np.mean(picp, axis=0)  # (d,)
            mpiw = np.mean(mpiw, axis=0)  # (d,)

        # unknown reduction method
        else:
            raise RuntimeError("Unknown reduction: \'%s\'" % reduction)

        # pack to namedtuple result
        result = picpresult(picp, mpiw)

        return result
