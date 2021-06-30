# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from netcal import AbstractCalibration, dimensions, accepts
from .NearIsotonicRegression import NearIsotonicRegression


class ENIR(AbstractCalibration):
    """
    Ensemble of Near Isotonic Regression (ENIR) models [1]_. These models allow - in contrast to standard
    :class:`IsotonicRegression` method - a violation of the monotony restrictions. Using the *modified
    Pool-Adjacent-Violators Algorithm (mPAVA)*, this method build multiple Near Isotonic Regression models
    and weights them by a certain score function.

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
    quick_init : bool, default=True
        Allow quick initialization of NIR (equal consecutive values are grouped directly).
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
    .. [1] Naeini, Mahdi Pakdaman, and Gregory F. Cooper:
       "Binary classifier calibration using an ensemble of near isotonic regression models."
       2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.
       `Get source online <https://ieeexplore.ieee.org/iel7/7837023/7837813/07837860.pdf>`_
    """

    @accepts(str, bool, bool, bool)
    def __init__(self, score_function: str = 'BIC', quick_init: bool = True,
                 detection: bool = False, independent_probabilities: bool = False):
        """
        Constructor.

        Parameters
        ----------
        score_function: str, default='BIC'
            define score functions:
            - 'BIC': Bayesian-Information-Criterion
            - 'AIC': Akaike-Information-Criterion
        quick_init : bool, default=True
            Allow quick initialization of NIR (equal consecutive values are grouped directly).
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

        # list of all binning models with [<NearIsotonicRegression>, ...]
        self._binning_models = []
        self._model_scores = []

        if type(score_function) != str:
            raise AttributeError("Score function must be string.")
        if score_function.lower() not in ['aic', 'bic']:
            raise AttributeError("Unknown score function \'%s\'" % score_function)

        self.score_function = score_function.lower()
        self.quick_init = quick_init

    def clear(self):
        """
        Clear model parameters.
        """
        super().clear()

        # for multi class calibration with K classes, K binary calibration models are needed
        for instance in self._multiclass_instances:
            del instance

        self._multiclass_instances.clear()

        # list of all binning models with [<NearIsotonicRegression>, ...]
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

    def set_params(self, **params) -> 'ENIR':
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
                instance = NearIsotonicRegression()
                instance.set_params(**model)
                self._binning_models.append(instance)

            # remove key and value from dict to prevent override in super method
            del params['_binning_models']

        # invoke super method
        super().set_params(**params)

        return self

    @dimensions((1, 2), (1, 2))
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ENIR':
        """
        Build ENIR calibration model.

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
        ENIR
            Instance of class :class:`ENIR`.
        """

        # detection mode is not supported natively
        if self.detection:
            print("WARNING: Detection mode is not supported natively by ENIR method. "
                  "This will discard all additional box information and only keep confidence scores.")

            # if 2d, keep only confidence scores and preserve 2d structure
            if len(X.shape) == 2:
                X = np.expand_dims(X[:, 0], axis=1)

        X, y = super().fit(X, y)

        # multiclass case: create K sub models for each label occurrence
        if not self._is_binary_classification():

            # create multiple one vs all models
            self._multiclass_instances = self._create_one_vs_all_models(X, y, ENIR, self.score_function,
                                                                        self.quick_init)
            return self

        # binary classification problem but got two entries? (probability for 0 and 1 separately)?
        # we only need probability p for Y=1 (probability for 0 is (1-p) )
        if len(X.shape) == 2:
            X = np.array(X[:, 1])
        else:
            X = np.array(X)

        X, y = self._sort_arrays(X, y)

        # log action
        print("Get path of all Near Isotonic Regression models with mPAVA ...")
        iso = NearIsotonicRegression(quick_init=self.quick_init,
                                     independent_probabilities=self.independent_probabilities)
        iso.fit(X, y)
        model_list = [iso]

        while iso is not None:
            iso = iso.get_next_model()
            model_list.append(iso)

        # first element is perfect fit to training data - discard due to overfitting
        model_list.pop(0)

        # last element is always None - indicator of mPAVA termination
        model_list.pop()

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

        # detection mode is not supported natively
        if self.detection:

            # if 2d, keep only confidence scores and preserve 2d structure
            if len(X.shape) == 2:
                X = np.expand_dims(X[:, 0], axis=1)

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
                normalizer = np.clip(np.sum(calibrated, axis=1, keepdims=True), self.epsilon, None)
                calibrated = np.divide(calibrated, normalizer)

        # on binary classification, it's much easier
        else:

            # get calibrated confidence estimates of each model and calculate scores
            calibrated = np.array([x.transform(X) for x in self._binning_models])
            calibrated = calibrated * np.expand_dims(self._model_scores, axis=1)
            calibrated = np.sum(calibrated, axis=0)

        return calibrated
