# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import abc, os
import numpy as np
from typing import Union, Tuple, Iterable, List
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn.functional as F

from netcal import __version__ as version
from .Decorator import accepts, dimensions


try:
    import cPickle as pickle
except ImportError:
    import pickle


class AbstractCalibration(BaseEstimator, TransformerMixin):
    """
    Abstract base class for all calibration methods.
    Inherits functions from sklearn's BaseEstimator.

    Parameters
    ----------
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    Attributes
    ----------
    epsilon : float
        Lowest possible digit that can be computed. Needed for several operations like divisions or log to guarantee
        values inequal to 0 or 1.
    """

    # epsilon to prevent division by zero
    epsilon = np.finfo(np.float).eps

    @accepts(bool, bool)
    def __init__(self, detection: bool = False, independent_probabilities: bool = False):
        """
        Create an instance of `AbstractCalibration`.

        Parameters
        ----------
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

        super().__init__()

        self.detection = detection
        self.num_classes = None
        self.independent_probabilities = independent_probabilities

        # this one is for 'clear' method to restore default
        self._default_independent_probabilities = independent_probabilities

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract function call to build the calibration model.
        This function performs several checks and returns the improved X and y.

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
        tuple
            Checked and converted X and y.

        Raises
        ------
        AttributeError
            - If 'X' has more than 2 dimensions
            - If 'y' has more than 2 dimensions
            - If 'y' as one-hot encoded ground truth does not match to detected number of classes
        """

        # invoke clear method on each fit
        self.clear()

        # check number of given samples
        if X.size <= 0:
            raise ValueError("No X samples provided.")

        # check number of given samples
        if y.size <= 0:
            raise ValueError("No y samples provided.")

        if y.shape[0] != X.shape[0]:
            raise AttributeError('Number of samples given by \'X\' and \'y\' is not equal.')

        # -----------------------------------------------------------------
        # preprocessing of confidence values given with X

        # remove single-dimensional entries if present
        X = self.squeeze_generic(X, axes_to_keep=0)

        # check shape of input array X and determine number of classes
        # first case: confidence array is 1-D: binary classification problem
        # this is either detection or classification mode (not important)
        if len(X.shape) == 1:
            self.num_classes = 2
            self.independent_probabilities = False

            # on detection mode, two dimensions are mandatory
            if self.detection:
                X = np.reshape(X, (-1, 1))

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

            # on detection, we face a binary classification problem
            if self.detection:
                self.num_classes = 2

            # classification
            else:

                # number of classes is length of second dimension (softmax) but at least 2 classes (binary case)
                self.num_classes = max(2, X.shape[1])

                # if second dimensions is less or equal than 2 and inputs are independent (multiple sigmoid outputs)
                # treat as binary classification and extract confidence estimates for y=1
                if X.shape[1] <= 2 and not self.independent_probabilities:
                    X = X[:, -1]
        else:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'X\' with num dimensions of %d."
                                 % len(X.shape))

        # check if array's values are in interval [0, 1]. This is not allowed for classification
        # and detection as well
        if not ((X >= 0).all() and (X <= 1).all()):
            raise ValueError("Some values of \'X\' are not in range [0, 1].")

        # -----------------------------------------------------------------
        # preprocessing of ground truth values given with y

        # remove single-dimensional entries if present
        y = self.squeeze_generic(y, axes_to_keep=0)

        # check shape of ground truth array y
        # array is 2-D: assume one-hot encoded
        if len(y.shape) == 2:

            # only binary problem but 2-dimensional and probabilities independent?
            # get labels only for y=1
            if y.shape[1] <= 2 and self.num_classes == 2 and not self.independent_probabilities:
                y = y[:, -1]

            elif y.shape[1] != self.num_classes:
                raise AttributeError("Assuming \'y\' as 2-dimensional one-hot encoded does not match to number of "
                                     "classes found in confidence array \'X\'.")

            else:
                # convert y from one-hot encoded to class labels (1-D)
                y = np.argmax(y, axis=1)

        elif len(y.shape) > 2:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'y\' with num dimensions of %d."
                                 % len(y.shape))

        # count all class labels and warn if not all labels are present
        unique = np.unique(y)

        # on detection, only binary labels are allowed
        if self.detection and len(unique) != 2:
            raise RuntimeError("On detection mode, it is mandatory to provide binary labels y in [0,1].")

        if len(unique) != self.num_classes:
            print("WARNING: Not all class labels are present in ground truth array \'y\'. "
                  "This could led to errors in some models.")

        return X, y

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract function. After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.
        This function performs several checks and returns the improved X.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with uncalibrated but checked confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).

        Raises
        ------
        RuntimeError
            If method is called before 'fit'.
        AttributeError
            - If model was built for multi class but 2nd dimension of 'X' does not match to number of classes.
            - If 'X' has more than 2 dimensions
        """

        # transform method invoked but model not built? Raise error
        if self.num_classes is None:
            raise RuntimeError('Could not invoke \'transform\' method before fit.')

        # -----------------------------------------------------------------
        # preprocessing of confidence values given with X

        # remove single-dimensional entries if present
        X = self.squeeze_generic(X, axes_to_keep=0)

        # got only 1-D array but model was fit for more than 2 classes?
        if len(X.shape) == 1:

            if self.num_classes > 2 and not self.detection:
                raise AttributeError("Model was build for %d classes but 1-D confidence array was provided."
                                     % self.num_classes)

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

            if X.shape[1] != self.num_classes and not self.detection:
                raise AttributeError("Model was build for %d classes but 2-D confidence  array  with %d classes"
                                     " was provided."
                                     % (self.num_classes, X.shape[1]))

            # if second dimensions is less or equal than 2 and inputs are independent (multiple sigmoid outputs)
            # treat as binary classification and extract confidence estimates for y=1
            if X.shape[1] == self.num_classes == 2 and not self.independent_probabilities and not self.detection:
                X = X[:, -1]

        else:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'X\' with num dimensions of %d."
                                 % len(X.shape))

        return X

    @abc.abstractmethod
    def clear(self):
        """
        Clear model parameters.
        """

        self.num_classes = 2 if self.detection else None
        self.independent_probabilities = self._default_independent_probabilities

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
        params = dict(vars(self))

        if deep:

            # needed for all binary methods that are distributed to multi class by one-vs-all
            if hasattr(self, '_multiclass_instances'):
                params['_multiclass_instances'] = []

                for i, instance in self._multiclass_instances:
                    params['_multiclass_instances'].append((i, instance.get_params(deep=True)))

        return params

    def set_params(self, **params) -> 'AbstractCalibration':
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

        if '_multiclass_instances' in params:
            self._multiclass_instances = []

            for label, subparams in params['_multiclass_instances']:
                instance = self.__class__()
                instance.set_params(**subparams)

                self._multiclass_instances.append((label, instance))

            # remove key and value from dict to prevent override in super method
            del params['_multiclass_instances']

        for key, value in params.items():
            setattr(self, key, value)

        return self

    @accepts(str)
    def save_model(self, filename: str):
        """
        Save model instance as with torch's save function as this is safer for torch tensors.

        Parameters
        ----------
        filename : str
            String with filename.
        """

        if filename.rfind("/") > 0:
            dir_path = filename[:filename.rfind("/")]
            os.makedirs(dir_path, exist_ok=True)

        with open(filename, 'wb') as write_object:
            torch.save(self.get_params(deep=True), write_object, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        """
        Load model from saved torch dump.

        Parameters
        ----------
        filename : str
            String with filename.

        Returns
        -------
        AbstractCalibration
            Instance of a child class of `AbstractCalibration`.
        """

        try:
            with open(filename, 'rb') as read_object:
                params = torch.load(read_object)
        except RuntimeError:
            raise IOError("Stored models of version <1.1 are not compatible with current version %s" % version)

        self.set_params(**params)
        return self

    @classmethod
    def squeeze_generic(cls, a: np.ndarray, axes_to_keep: Union[int, Iterable[int]]) -> np.ndarray:
        """
        Squeeze input array a but keep axes defined by parameter 'axes_to_keep' even if the dimension is
        of size 1.

        Parameters
        ----------
        a : np.ndarray
            NumPy array that should be squeezed.
        axes_to_keep : int or iterable
            Axes that should be kept even if they have a size of 1.

        Returns
        -------
        np.ndarray
            Squeezed array.

        """

        # if type is int, convert to iterable
        if type(axes_to_keep) == int:
            axes_to_keep = (axes_to_keep, )

        # iterate over all axes in a and check if dimension is in 'axes_to_keep' or of size 1
        out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
        return a.reshape(out_s)

    @accepts(np.ndarray, np.ndarray, list, str)
    def _calc_model_scores(self, confidences: np.ndarray, ground_truth: np.ndarray,
                           model_list: List['AbstractCalibration'], score_function: str = 'BIC') -> np.ndarray:
        """
        Calculates the Bayesian Scores for each Histogram Binning model and discards each model which
        gets a score of 0 to speed up predictions later on.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        model_list : list
            List with models to compute the scores for.
        score_function : str, default: 'BIC'
            Score function which to use. Must be either 'BIC' or 'AIC'.

        Returns
        -------
        np.ndarray, shape=(n_models,)
            NumPy 1-D array with scores for each model provided in list.
        """

        num_samples = ground_truth.size

        # calculate log likelihood of confidences
        log_likelihood = np.zeros(len(model_list))
        for i, model in enumerate(model_list):
            estimate = model.transform(confidences)

            # use torch routines to get log likelihood
            if self._is_binary_classification():
                loss = F.binary_cross_entropy(torch.from_numpy(estimate).type(torch.FloatTensor),
                                              torch.from_numpy(ground_truth).type(torch.FloatTensor),
                                              reduction='sum')

            # on multiclass, use NLLLoss - this function expects log-probabilities
            else:
                loss = F.nll_loss(torch.from_numpy(np.log(estimate)).type(torch.FloatTensor),
                                  torch.from_numpy(ground_truth).type(torch.FloatTensor),
                                  reduction='sum')

            log_likelihood[i] = -loss.item()

        # get degrees of freedom of each model. This is equivalent to number of groups
        degrees_of_freedom = np.array([x.get_degrees_of_freedom() for x in model_list], dtype=np.int)

        # choose scoring function 'AIC' oder 'BIC'
        if score_function == 'aic':
            score = 2. * degrees_of_freedom - 2. * log_likelihood
        elif score_function == 'bic':
            score = -2. * log_likelihood + degrees_of_freedom * np.log(num_samples)
        else:
            raise ValueError("Unknown score function \'%s\'. Fix your implementation")

        # calculate relative likelihood of each model and normalize scores
        model_scores = np.exp((np.min(score) - score) / 2.)

        return model_scores

    @accepts(np.ndarray, np.ndarray, list, str, float)
    def _elbow(self, confidences: np.ndarray, ground_truth: np.ndarray, model_list: List['AbstractCalibration'],
               score_function: str = 'BIC', alpha: float = 0.001) -> Tuple[List[float], List['AbstractCalibration']]:
        """
        Select models by Bayesian score and discard models below a certain threshold with elbow method.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        model_list : list
            List with models to compute the scores for.
        score_function : str, default: 'BIC'
            Score function which to use. Must be either 'BIC' or 'AIC'.
        alpha : float, default: 0.001
            Threshold of model scores.

        Returns
        -------
        tuple, shape=(2,)
            Tuple with two lists of (normed) model scores and list of models.
        """

        model_scores = self._calc_model_scores(confidences, ground_truth, model_list, score_function)

        num_models = len(model_scores)
        score_variance = np.var(model_scores)

        # get reversed sorted indices
        sorted_indices = np.argsort(model_scores)[::-1]
        sorted_model_scores = model_scores[sorted_indices]

        # k is number of models kept
        k = 0

        # first part: seek for firt models with equal score
        while k < (num_models-1) and sorted_model_scores[k] == sorted_model_scores[k+1]:
            k += 1

        # second part: seek for models with score difference of two consecutive models scaled by
        # score variance below threshold
        while k < (num_models-1) and ((sorted_model_scores[k] - sorted_model_scores[k+1]) / score_variance) > alpha:
            k += 1

        # k denotes amount of kept models - range(k) must also include last model
        # thus, increase by 1
        k += 1

        kept_models = [model_list[sorted_indices[i]] for i in range(k)]

        # norm scores to sum up to 1
        kept_scores = [model_scores[sorted_indices[i]] for i in range(k)]
        kept_scores /= np.sum(kept_scores)

        return kept_scores, kept_models

    def _create_one_vs_all_models(self, confidences: np.ndarray, ground_truth: np.ndarray,
                                  model_class: 'AbstractCalibration', *constructor_args) -> List['AbstractCalibration']:
        """
        Create for K classes K one vs all calibration models.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, n_classes)
            NumPy array with confidence values for each prediction.
            This must be 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        model_class : child of AbstractCalibration
            instance of child of `AbstractCalibration` class.
        *constructor_args
            several args passed to class constructor.

        Returns
        -------
        list, shape=(n_classes,)
            List with calibration models.
        """

        multiclass_instances = []
        for label in range(self.num_classes):

            if np.where(ground_truth == label)[0].size == 0:
                print("WARNING: no training data for label %d present" % label)
                continue

            # get 1 vs all vector depending on current label
            onevsall_ground_truth = self._get_one_vs_all_label(ground_truth, label)
            onevsall_confidence = self._get_one_vs_all_confidence(confidences, label)

            # build an own Histogram Binning model for each label in a 1 vs all manner
            # now it's a k-fold binary classification task
            model = model_class(*constructor_args, independent_probabilities=self.independent_probabilities)
            model.fit(onevsall_confidence, onevsall_ground_truth)

            # add instances to internal list for calibrating new confidence estimates
            multiclass_instances.append(tuple((label, model)))

        return multiclass_instances

    @classmethod
    def _sort_arrays(cls, array1: np.ndarray, *args) -> Tuple:
        """
        Sort multiple NumPy arrays by values given with array1.

        Parameters
        ----------
        array1 : np.ndarray
            NumPy 1-D array with values to sort.
        *args : np.ndarray
            NumPy 1-D arrays that get sorted by array1.

        Returns
        -------
        tuple, shape=(n_args,)
            Tuple with all given arrays sorted.

        Raises
        ------
        AttributeError
            If types of parameters are not np.ndarray, if shapes of arrays is not 1-D or
            if shapes of arrays do not match to each other.
        """

        if type(array1) != np.ndarray:
            raise AttributeError("Type of arrays must be numpy.ndarray")

        if len(array1.shape) != 1:
            raise AttributeError("Shape of arrays must be 1-D")

        # check dimensions
        for array in args:
            if type(array) != np.ndarray:
                raise AttributeError("Type of arrays must be numpy.ndarray")

            if array1.shape != array.shape:
                raise AttributeError("Shape mismatch of arrays")

        # sort arrays by confidence
        p = array1.argsort()
        array1 = array1[p]
        return_tuple = [array1, ]

        for array in args:
            array = array[p]
            return_tuple.append(array)

        return tuple(return_tuple)

    def _is_binary_classification(self) -> bool:
        """
        Determine if a given label vector is from a binary classification problem.
        This is important in order to process Histogram Binning.

        Returns
        -------
        bool
            True if binary classification problem (only two different labels occur), else False.
        """

        if self.num_classes == 2 and not self.independent_probabilities:
            return True
        else:
            return False

    @dimensions(1, None)
    def _get_one_vs_all_label(self, label_vec: np.ndarray, label: int) -> np.ndarray:
        """
        Return the all vs one vector for a given label vector. If an entry matches 'label', this entry
        is converted to 1, the rest to 0. The confidence for entries which are not 'label' are inverted (1 - conf).

        Parameters
        ----------
        label_vec : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with label_vec (prediction or ground_truth).
        label : int
            Integer with label which will be replaced by 1 (rest by 0).

        Returns
        -------
        np.ndarray
            NumPy 1-D array with replaced labels ('label' by 1, rest by 0).
        """

        onevsall_label = np.array(label_vec, dtype=np.int)
        onevsall_label[label_vec == label] = 1
        onevsall_label[label_vec != label] = 0

        return onevsall_label

    @dimensions(2, None)
    def _get_one_vs_all_confidence(self, confidences: np.ndarray, label: int) -> np.ndarray:
        """
        Return the all vs one vector for confidence. Each confidence entry will be calculated as (1 - conf)
        for each entry in 'onevsall_label' array which is 0.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, n_classes)
            NumPy array with confidence values for each prediction.
            This must be 2-D for multi class (softmax).
        label : int
            Integer with label which will be replaced by 1 (rest by 0).

        Returns
        -------
        np.ndarray
            NumPy 1-D array with confidences for given label.
        """

        onevsall_conf = confidences[:, label]
        return onevsall_conf
