# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from calibration import AbstractCalibration, dimensions, accepts
from .HistogramBinning import HistogramBinning


class BBQ(AbstractCalibration):
    """
    Bayesian Binning into Quantiles (BBQ). This method utilizes multiple :class:`HistogramBinning`
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

       p(M | \\mathcal{D}) \\propto p( \\mathcal{D} | M )p(M) \\approx \exp( -BIC/2 )p(M) .

    Using the elbow method to sort out models with a low relative score, the weights for each model can be obtained
    by normalizing over all model posterior scores.

    Parameters
    ----------
    score_function: str, default='BIC'
        define score functions:
        - 'BIC': Bayesian-Information-Criterion
        - 'AIC': Akaike-Information-Criterion
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht:
    "Obtaining well calibrated probabilities using bayesian binning."
    Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
    `Get source online <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_

    """

    @accepts(str, bool)
    def __init__(self, score_function: str = 'BIC', independent_probabilities: bool = False):
        """
        Constructor.

        Parameters
        ----------
        score_function: str, default='BIC'
            define score functions:
            - 'BIC': Bayesian-Information-Criterion
            - 'AIC': Akaike-Information-Criterion
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(independent_probabilities)

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

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BBQ':
        """
        Build BBQ calibration model.

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
        BBQ
            Instance of class :class:`BBQ`.
        """

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification():

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, BBQ, self.score_function)
            return self

        num_samples = y.size
        sqrt3_num_samples = np.power(num_samples, 1. / 3.)

        # bin range as proposed in the paper of the authors
        # guarantee, that at least one bin model is present and least a 5 bin model
        min_bins = int(max(1, np.floor(sqrt3_num_samples / 10.)))
        max_bins = int(min(np.ceil(num_samples / 5), np.ceil(sqrt3_num_samples * 10.)))

        num_binning_models = max_bins - min_bins + 1

        # iterate over all different binnings and fit Histogram Binning methods
        model_list = []
        for num_model in range(num_binning_models):

            histogram = HistogramBinning(bins=min_bins+num_model)
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
        if not self._is_binary_classification():

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
