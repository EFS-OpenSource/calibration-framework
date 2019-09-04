# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import abc, os, logging
import numpy as np
from scipy.special import expit as safe_sigmoid
from scipy.special import logit as safe_logit
from scipy.special import softmax as safe_softmax
from sklearn.base import BaseEstimator, TransformerMixin
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
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    Attributes
    ----------
    epsilon : float
        Lowest possible digit that can be computed. Needed for several operations like divisions or log to guarantee
        values inequal to 0 or 1.

    logger : RootLogger
        Logger for printing debug/info/warning/error messages.

    """

    # epsilon to prevent division by zero
    epsilon = np.finfo(np.float).eps

    @accepts(bool)
    def __init__(self, independent_probabilities: bool = False):
        """
        Create an instance of `AbstractCalibration`.

        Parameters
        ----------
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__()
        self.logger = logging.getLogger('calibration')
        self.num_classes = None
        self.independent_probabilities = independent_probabilities

        # this one is for 'clear' method to restore default
        self.__default_independent_probabilities = independent_probabilities

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Abstract function call to build the calibration model.
        This function performs several checks and returns the improved X and y.

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

        # -----------------------------------------------------------------
        # preprocessing of confidence values given with X

        # remove single-dimensional entries if present
        X = np.squeeze(X)

        # check shape of input array X and determine number of classes
        # first case: confidence array is 1-D: binary classification problem
        if len(X.shape) == 1:
            self.num_classes = 2
            self.independent_probabilities = False

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

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

        # -----------------------------------------------------------------
        # preprocessing of ground truth values given with y

        # remove single-dimensional entries if present
        y = np.squeeze(y)

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
        if len(unique) != self.num_classes:
            self.logger.warning("Not all class labels are present in ground truth array \'y\'. This could led to "
                                "errors in some models.")

        return X, y

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract function. After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.
        This function performs several checks and returns the improved X.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with uncalibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).

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
        X = np.squeeze(X)

        # got only 1-D array but model was fit for more than 2 classes?
        if len(X.shape) == 1:

            if self.num_classes > 2:
                raise AttributeError("Model was build for %d classes but 1-D confidence array was provided."
                                     % self.num_classes)

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

            if X.shape[1] != self.num_classes:
                raise AttributeError("Model was build for %d classes but 2-D confidence  array  with %d classes"
                                     " was provided."
                                     % (self.num_classes, X.shape[1]))

            # if second dimensions is less or equal than 2 and inputs are independent (multiple sigmoid outputs)
            # treat as binary classification and extract confidence estimates for y=1
            if X.shape[1] == self.num_classes == 2 and not self.independent_probabilities:
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

        self.num_classes = None
        self.independent_probabilities = self.__default_independent_probabilities

    @accepts(str)
    def save_model(self, filename: str):
        """
        Save model instance as Pickle Object.

        Parameters
        ----------
        filename : str
            String with filename.
        """

        if filename.rfind("/") > 0:
            dir_path = filename[:filename.rfind("/")]
            os.makedirs(dir_path, exist_ok=True)

        with open(filename, 'wb') as write_object:
            pickle.dump(self, write_object, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, filename):
        """
        Load model from saved Pickle instance.

        Parameters
        ----------
        filename : str
            String with filename.

        Returns
        -------
        AbstractCalibration
            Instance of a child class of `AbstractCalibration`.
        """

        with open(filename, 'rb') as read_object:
            new_instance = pickle.load(read_object)

        return new_instance

    @accepts(np.ndarray, np.ndarray, list, str)
    def _calc_model_scores(self, confidences: np.ndarray, ground_truth: np.ndarray, model_list: list,
                           score_function: str = 'BIC') -> np.ndarray:
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
            log_likelihood[i] = self._log_likelihood(estimate, ground_truth)

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
    def _elbow(self, confidences: np.ndarray, ground_truth: np.ndarray, model_list: list, score_function: str = 'BIC',
               alpha: float = 0.001) -> tuple:
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
                                  model_class: 'AbstractCalibration', *constructor_args) -> list:
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
                self.logger.warning("Warning: no training data for label %d present" % label)
                continue

            # get 1 vs all vector depending on current label
            onevsall_ground_truth = self._get_one_vs_all_label(ground_truth, label)
            onevsall_confidence = self._get_one_vs_all_confidence(confidences, label)

            # build an own Histogram Binning model for each label in a 1 vs all manner
            # now it's a k-fold binary classification task
            binning_model = model_class(*constructor_args, independent_probabilities=self.independent_probabilities)
            binning_model.fit(onevsall_confidence, onevsall_ground_truth)

            # add instances to internal list for calibrating new confidence estimates
            multiclass_instances.append(tuple((label, binning_model)))

        return multiclass_instances

    @classmethod
    def _sort_arrays(cls, array1: np.ndarray, *args) -> tuple:
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

    @dimensions(1, None)
    def _get_one_hot_encoded_labels(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Compute one hot encoded label array by given label vector.

        Parameters
        ----------
        labels : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with labels.
        num_classes : int
            Total amount of present classes.

        Returns
        -------
        np.ndarray
            NumPy 2-D array with one hot encoded labels.
        """

        # one hot encoded label vector on multi class calibration
        return np.eye(num_classes)[labels]

    @dimensions((1, 2), (1, 2))
    def _nll_loss(self, logit: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute negative log-likelihood.

        Parameters
        ----------
        logit : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with logits for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels.
            1-D for binary classification, 2-D for multi class (one-hot encoded).

        Returns
        -------
        float
            float with NLL-Loss.
        """

        num_samples = ground_truth.shape[0]

        # scale logits by optional parameter 'scale_logits' (cf. Temperature Scaling)
        # afterwards, compute softmax or sigmoid (depends on binary/multi class)
        if self.num_classes <= 2 or self.independent_probabilities:

            # if array is two dimensional, 2 cases can occur:
            # - 2nd dimension is length 1 - thus, the dimension can be squeezed
            # - 2nd dimension is length 2 - logits for y=0 and y=1 - take estimates for y=1 then
            if len(logit.shape) == 2 and not self.independent_probabilities:
                logit = logit[:, -1]

            confidences = self._sigmoid(logit)
        else:
            # convert ground truth to one hot encoded if necessary
            if len(ground_truth.shape) == 1:
                ground_truth = self._get_one_hot_encoded_labels(ground_truth, self.num_classes)

            confidences = self._softmax(logit)

        # clip to epsilon and 1.-epsilon
        confidences = np.clip(confidences, self.epsilon, 1. - self.epsilon)

        log_likelihood = self._log_likelihood(confidences, ground_truth)
        return -log_likelihood / num_samples

    @dimensions((1, 2), (1, 2))
    def _log_likelihood(self, confidences: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute log-likelihood of predictions given in confidences with according
        ground truth information.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels.
            1-D for binary classification, 2-D for multi class (one-hot encoded).

        Returns
        -------
        float
            Log likelihood.

        Raises:
            ValueError
                If number of classes is 2 but 1-D NumPy array provided.
        """

        if self.num_classes > 2:
            if len(confidences.shape) < 2:
                raise ValueError("Need 2-D array for multiclass log-likelihood")

            # convert ground truth to one hot encoded if necessary
            if len(ground_truth.shape) == 1 and len(confidences.shape) == 2:
                ground_truth = self._get_one_hot_encoded_labels(ground_truth, self.num_classes)

            # clip confidences for log
            confidences = np.clip(confidences, self.epsilon, 1. - self.epsilon)

            # first, create log of softmax and multiply by according ground truth (0 or 1)
            # second, sum each class and calculate mean over all samples
            cross_entropy = np.multiply(ground_truth, np.log(confidences))
            log_likelihood = np.sum(cross_entropy)

        else:

            # binary classification problem but got two entries? (probability for 0 and 1 separately)?
            # we only need probability p for Y=1 (probability for 0 is (1-p) )
            if len(confidences.shape) == 2:
                confidences = np.array(confidences[:, 1])

            # clip confidences for log - extra clip for negative values necessary due to
            # numerical stability
            negative_confidences = np.clip(1. - confidences, self.epsilon, 1. - self.epsilon)
            confidences = np.clip(confidences, self.epsilon, 1. - self.epsilon)

            # first, create log of sigmoid and multiply by according ground truth (0 or 1)
            # second, calculate mean over all samples
            cross_entropy = np.multiply(ground_truth, np.log(confidences)) + \
                            np.multiply(1. - ground_truth, np.log(negative_confidences))

            log_likelihood = np.sum(cross_entropy)

        return float(log_likelihood)

    @dimensions((1, 2))
    def _sigmoid(self, logit: np.ndarray) -> np.ndarray:
        """
        Calculate Sigmoid of Logit

        Parameters
        ----------
        logit : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with logit of Neural Network.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with sigmoid output.
        """

        return safe_sigmoid(logit)

    @dimensions((1, 2))
    def _inverse_sigmoid(self, confidence: np.ndarray) -> np.ndarray:
        """
        Calculate inverse of Sigmoid to get Logit.

        Parameters
        ----------
        confidence : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with Sigmoid output.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with logit.
        """

        # return - np.log( (1./(confidence + self.epsilon)) - 1)
        clipped = np.clip(confidence, self.epsilon, 1. - self.epsilon)
        return safe_logit(clipped)

    @dimensions(2)
    def _softmax(self, logit: np.ndarray) -> np.ndarray:
        """
        Calculate Softmax of multi class logit.

        Parameters
        ----------
        logit : np.ndarray, shape=(n_samples, n_classes)
            NumPy 2-D array with logits.

        Returns
        -------
        np.ndarray, shape=(n_samples, n_classes)
            NumPy 2-D array with softmax output.
        """

        return safe_softmax(logit, axis=1)

    @dimensions(2)
    def _inverse_softmax(self, confidences: np.ndarray) -> np.ndarray:
        """
        Calculate inverse of multi class softmax.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, n_classes)
            NumPy 2-D array with softmaxes.

        Returns
        -------
        np.ndarray, shape=(n_samples, n_classes)
            NumPy 2-D array with logits.
        """

        clipped = np.clip(confidences, self.epsilon, 1. - self.epsilon)
        return np.log(clipped)
