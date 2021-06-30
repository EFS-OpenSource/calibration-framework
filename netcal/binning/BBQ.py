# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import itertools
from tqdm import tqdm

from netcal import AbstractCalibration, dimensions, accepts
from .HistogramBinning import HistogramBinning


class BBQ(AbstractCalibration):
    """
    Bayesian Binning into Quantiles (BBQ) [1]_. This method utilizes multiple :class:`HistogramBinning`
    instances with different amounts of bins and computes a weighted sum of all methods to obtain a
    well-calibrated confidence estimate. The scoring function "BDeu", which is proposed in the original paper,
    is currently not supported.

    Let :math:`\\mathcal{D} = \\{(x_0, y_0), (x_1, y_1), ... \\}` denote
    a data set with input data :math:`x` and ground truth labels :math:`y \\in \\{0, 1\\}` of length :math:`N`.
    Let :math:`M` denote a model with estimated parameters :math:`\\hat{\\theta}` of length :math:`K`
    and let :math:`\\hat{p}` denote the confidence estimates
    on :math:`\\mathcal{D}` of model :math:`M` by parameters :math:`\\hat{\\theta}`.
    The score function might either be the *Aikaike Information
    Criterion (AIC)* given by

    .. math::

       AIC = -2 L ( \\hat{\\theta} ) + 2K

    or the *Bayesian Information Criterion* given by

    .. math::

       BIC = -2 L ( \\hat{\\theta} ) + \log(N)K

    with :math:`L (\\hat{ \\theta })` as the log-likelihood given by

    .. math::

       L (\\hat{ \\theta }) = \\sum_{i=1}^N y^{(i)}  \\log(\\hat{p}^{(i)}_\\hat{\\theta}) +
       (1-y^{(i)}) \\log(1 - \\hat{p}^{(i)}_\\hat{\\theta}) .

    These scores can be used to calculate a model posterior given by

    .. math::

       p(M | \\mathcal{D}) \\propto p( \\mathcal{D} | M )p(M) \\approx \\exp( -BIC/2 )p(M) .

    Using the elbow method to sort out models with a low relative score, the weights for each model can be obtained
    by normalizing over all model posterior scores.

    Parameters
    ----------
    score_function: str, default='BIC'
        define score functions:
        - 'BIC': Bayesian-Information-Criterion
        - 'AIC': Akaike-Information-Criterion
    equal_intervals : bool, optional, default: True
        If True, the bins have the same width. If False, the bins are splitted to equalize
        the number of samples in each bin.
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    .. [1] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht:
       "Obtaining well calibrated probabilities using bayesian binning."
       Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
       `Get source online <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_

    """

    @accepts(str, bool, bool, bool)
    def __init__(self, score_function: str = 'BIC', equal_intervals: bool = True,
                 detection: bool = False, independent_probabilities: bool = False):
        """
        Constructor.

        Parameters
        ----------
        score_function: str, default='BIC'
            define score functions:
            - 'BIC': Bayesian-Information-Criterion
            - 'AIC': Akaike-Information-Criterion
        detection : bool, default: False
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(detection=detection, independent_probabilities=independent_probabilities)

        # for multi class calibration with K classes, K binary calibration models are needed
        self._multiclass_instances = []

        # list of all binning models with [<HistogramBinning>, ...]
        self._binning_models = []
        self._model_scores = []

        if type(score_function) != str:
            raise AttributeError("Score function must be string.")
        if score_function.lower() not in ['aic', 'bic']:
            raise AttributeError("Unknown score function \'%s\'" % score_function)

        self.score_function = score_function.lower()
        self.equal_intervals = equal_intervals

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()

        # for multi class calibration with K classes, K binary calibration models are needed
        for instance in self._multiclass_instances:
            del instance

        self._multiclass_instances.clear()

        # list of all binning models with [<HistogramBinning>, ...]
        for model in self._binning_models:
            del model

        self._binning_models.clear()
        self._model_scores = None

    @accepts(bool)
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        # get all params of current instance and save as dict
        params = super().get_params(deep=deep)

        if deep:

            # save binning models as well - this is not captured by super class method
            params['_binning_models'] = []

            for model in self._binning_models:
                params['_binning_models'].append(model.get_params(deep=deep))

        return params

    def set_params(self, **params) -> 'BBQ':
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """

        if '_binning_models' in params:
            self._binning_models = []

            for model in params['_binning_models']:

                instance = HistogramBinning()
                instance.set_params(**model)
                self._binning_models.append(instance)

            # remove key and value from dict to prevent override in super method
            del params['_binning_models']

        # invoke super method
        super().set_params(**params)

        return self

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BBQ':
        """
        Build BBQ calibration model.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        BBQ
            Instance of class :class:`BBQ`.
        """

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification() and not self.detection:

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, BBQ, self.score_function, self.equal_intervals)
            return self

        num_features = 1
        if self.detection and len(X.shape) == 2:
            num_features = X.shape[1]
        
        num_samples = y.size

        constant = 10. / num_features
        sqrt_num_samples = np.power(num_samples, 1. / (num_features + 2))

        # bin range as proposed in the paper of the authors
        # guarantee, that at least one bin model is present and least a 5 bin model
        min_bins = int(max(1, np.floor(sqrt_num_samples / constant)))
        max_bins = int(min(np.ceil(num_samples / 5), np.ceil(sqrt_num_samples * constant)))

        bin_range = range(min_bins, max_bins+1)
        all_ranges = [bin_range,] * num_features

        # iterate over all different binnings and fit Histogram Binning methods
        model_list = []

        # iterate over all bin combinations
        for bins in tqdm(itertools.product(*all_ranges), total=np.power(len(bin_range), num_features)):

            bins = bins[0] if not self.detection else bins
            histogram = HistogramBinning(bins=bins, equal_intervals=self.equal_intervals, detection=self.detection)
            histogram.fit(X, y)

            model_list.append(histogram)

        # get model scores and binning models by elbow method
        self._model_scores, self._binning_models = self._elbow(X, y, model_list, self.score_function, alpha=0.001)
        return self

    @dimensions((1, 2))
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with uncalibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with calibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).
        """

        X = super().transform(X)

        # prepare return value vector
        calibrated = np.zeros(X.shape)

        # if multiclass classification problem, use binning models for each label separately
        if not self._is_binary_classification() and not self.detection:

            # get all available labels and iterate over each one
            for (label, binning_model) in self._multiclass_instances:
                onevsall_confidence = self._get_one_vs_all_confidence(X, label)
                onevsall_calibrated = binning_model.transform(onevsall_confidence)

                # choose right position for submodel's calibration
                calibrated[:, label] = onevsall_calibrated

            if not self.independent_probabilities:
                # normalize to keep probability sum of 1
                normalizer = np.sum(calibrated, axis=1, keepdims=True)
                calibrated = np.divide(calibrated, normalizer)

        # on binary classification, it's much easier
        else:

            # get calibrated confidence estimates of each model and calculate scores
            calibrated = np.array([x.transform(X) for x in self._binning_models])
            calibrated = calibrated * np.expand_dims(self._model_scores, axis=1)
            calibrated = np.sum(calibrated, axis=0)

        return calibrated
