# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from typing import Union
from netcal import AbstractCalibration, dimensions, accepts


class NearIsotonicRegression(AbstractCalibration):
    """
    Near Isotonic Regression Calibration method [1]_ (commonly used by :class:`ENIR` [2]_).

    Parameters
    ----------
    quick_init : bool, optional, default: True
        Allow quick initialization of NIR (equal consecutive values are grouped directly).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    .. [1] Ryan J Tibshirani, Holger Hoefling, and Robert Tibshirani:
       "Nearly-isotonic regression."
       Technometrics, 53(1):54–61, 2011.
       `Get source online <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.7054&rep=rep1&type=pdf>`

    .. [2] Naeini, Mahdi Pakdaman, and Gregory F. Cooper:
       "Binary classifier calibration using an ensemble of near isotonic regression models."
       2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.
       `Get source online <https://ieeexplore.ieee.org/iel7/7837023/7837813/07837860.pdf>`
    """

    @accepts(bool, bool)
    def __init__(self, quick_init: bool = True, independent_probabilities: bool = False):
        """
        Create an instance of `NearIsotonicRegression`.

        Parameters
        ----------
        quick_init : bool, optional, default: True
            Allow quick initialization of NIR (equal consecutive values are grouped directly).
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """
        super().__init__(detection=False, independent_probabilities=independent_probabilities)

        self._lambda = 0.0

        # group values: betas/new confidence in each group
        # group items: ground truth labels of each sample in a group
        # group bounds: lower and upper boundaries of each group
        self._num_groups = None
        self._group_items = None
        self._group_bounds = None
        self._group_values = None

        self.quick_init = quick_init

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        self._num_groups = None
        self._group_items = None
        self._group_bounds = None
        self._group_values = None

        self._lambda = 0.0

    def fit(self, X: np.ndarray = None, y: np.ndarray = None,
            last_model: 'NearIsotonicRegression' = None) -> Union['NearIsotonicRegression', None]:
        """
        Build NIR model either as initial model given by parameters 'ground_truth' and 'confidences' or as
        intermediate model based on 'last_model' which is an instance of 'NearIsotonicRegression'.

        Parameters
        ----------
        X : np.ndarray, optional, default: None, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, optional, default: None, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
        last_model : NearIsotonicRegression, optional, default: None
            Instance of NearIsotonicRegression (required, if 'X' and 'y' is empty).

        Returns
        -------
        NearIsotonicRegression or None
            Instance of class :class:`NearIsotonicRegression` or None
            if next lambda is less than current lambda (end of mPAVA).
        """

        if last_model is None:
            if X is None or y is None:
                raise AttributeError("Could not initialize mPAVA algorithm without "
                                     "an array of ground truth and confidence values")

            X, y = super().fit(X, y)
            if self.quick_init:
                self.__initial_model_quick(X, y)
            else:
                self.__initial_model_standard(X, y)

            return self

        # get attributes of previous NIR model
        self._lambda = last_model._lambda
        self._num_groups = last_model._num_groups
        self._group_items = last_model._group_items
        self._group_values = last_model._group_values
        self._group_bounds = last_model._group_bounds

        self.quick_init = last_model.quick_init
        self.num_classes = last_model.num_classes
        self.independent_probabilities = last_model.independent_probabilities

        # get slopes and collision times (where consecutive bins might be merged)
        slopes = self.__get_slopes()
        t_values = self.__get_collision_times(slopes)

        # calculate next lambda (monotony violation weight)
        # this is denoted as lambda ast
        next_lambda = np.min(t_values)

        # if next lambda is less than current lambda, terminate mPAVA algorithm
        if next_lambda < self._lambda or next_lambda == np.inf:
            return None

        # now update group values and merge groups with equal values
        self.__update_group_values(slopes, next_lambda)
        self.__merge_groups(t_values, next_lambda)

        # get new lambda (ast) and set as current lambda
        self._lambda = next_lambda

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
        calibrated = np.zeros_like(X)
        for i in range(self._num_groups):
            bounds = self._group_bounds[i]

            if bounds[0] == 0.0:
                calibrated[(X >= bounds[0]) & (X <= bounds[1])] = self._group_values[i]
            else:
                calibrated[(X > bounds[0]) & (X <= bounds[1])] = self._group_values[i]

        if not self.independent_probabilities:
            # apply normalization on multi class calibration
            if len(X.shape) == 2:
                # normalize to keep probability sum of 1
                normalizer = np.sum(calibrated, axis=1, keepdims=True)
                calibrated = np.divide(calibrated, normalizer)

        return calibrated

    def get_next_model(self) -> Union['NearIsotonicRegression', None]:
        """
        Get next Near Isotonic Regression model based on mPAVA algorithm

        Returns
        -------
        NearIsotonicRegression
            Next instance of :class:`NearIsotonicRegression`.
        """

        next_model = NearIsotonicRegression(self.quick_init)
        if next_model.fit(last_model=self) is None:
            del next_model
            next_model = None

        return next_model

    def get_degrees_of_freedom(self) -> int:
        """
        Needed for BIC. Returns the degree of freedom. This simply returns the
        number of groups

        Returns
        -------
        int
            Integer with degree of freedom.
        """

        return int(self._num_groups)

    # -------------------------------------------------------------
    @dimensions(1, (1, 2))
    def __initial_model_standard(self, X: np.ndarray, y: np.ndarray):
        """
        Initial NIR model standard initialization (like described in original NIR paper).
        Each group holds only a single ground truth value and gets its value.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        """

        # one hot encoded label vector on multi class calibration
        if len(X.shape) == 2:
            y = np.eye(self.num_classes)[y]

        # sort arrays by confidence - always flatten (this has no effect to 1-D arrays)
        X, y = self._sort_arrays(X.flatten(), y.flatten())

        self._num_groups = y.size
        self._group_items = np.split(y, y.size)

        # calculate bounds as median
        bounds = np.divide(X[:-1] + X[1:], 2.)
        lower_bounds = np.insert(bounds, 0, 0.0)
        upper_bounds = np.append(bounds, 1.0)

        self._group_bounds = np.stack((lower_bounds, upper_bounds), axis=1)
        self._group_values = np.array(y, dtype=np.float)

    @dimensions(1, (1, 2))
    def __initial_model_quick(self, X: np.ndarray, y: np.ndarray):
        """
        Initial NIR model quick initialization (own implementation).
        Each group is computed by consecutive equal values of ground truth.
        Therefore, the algorithm starts with perfect fit to data.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        """

        # one hot encoded label vector on multi class calibration
        if len(X.shape) == 2:
            y = np.eye(self.num_classes)[y]

        # sort arrays by confidence - always flatten (this has no effect to 1-D arrays)
        X, y = self._sort_arrays(X.flatten(), y.flatten())

        # get monotony violations directly and create according groups
        # compute differences of consecutive ground truth labels
        differences = y[1:] - y[:-1]

        # monotony violations are found where differences are either 1 (from 0 -> 1) or -1 (from 1 -> 0)
        # store these violations as differences
        violations = np.where(differences != 0.0)[0]
        differences = differences[violations]

        # amount of available groups is amount of differences (+ initial values)
        self._num_groups = differences.size + 1

        # group values are differences (map -1 to 0 and insert first ground truth value as first group value)
        self._group_values = differences
        self._group_values[differences == -1.] = 0.0
        self._group_values = np.insert(differences, 0, y[0]).astype(np.float)

        # get group items as NumPy arrays
        # split arrays where monotony violations are found (index +1 needed)
        self._group_items = np.split(y, violations + 1)

        # group bounds can also be found where monotony violations are present
        bounds = np.divide(X[violations] + X[violations + 1], 2.)

        # include 0 and 1 as bounds, too
        lower_bounds = np.insert(bounds, 0, 0.0)
        upper_bounds = np.append(bounds, 1.0)

        self._group_bounds = np.stack((lower_bounds, upper_bounds), axis=1)

    def __get_slopes(self) -> np.ndarray:
        """
        Get the derivative or slope of each bin value with respect to given lambda.

        Returns
        -------
        np.ndarray, shape=(n_bins,)
            NumPy 1-D array slopes of each bin.
        """

        # determine amount of samples in each group and create Numpy vector
        num_samples_per_group = np.array([self._group_items[i].size for i in range(self._num_groups)], dtype=np.float)

        # calculate monotony violation of consecutive group values (consecutive betas)
        pre_group_values = np.array(self._group_values[:-1])
        post_group_values = np.array(self._group_values[1:])

        # perform compare operations with NumPy methods
        indicator = np.greater(pre_group_values, post_group_values)
        indicator = np.insert(indicator, 0, False)
        indicator = np.append(indicator, False)
        indicator = indicator.astype(np.float)

        # slopes are calculated by previously calculated indicator
        slopes = indicator[:-1] - indicator[1:]
        slopes = np.divide(slopes, num_samples_per_group)

        return slopes

    @dimensions(1)
    def __get_collision_times(self, slopes: np.ndarray) -> np.ndarray:
        """
        Calculate t values. These values give the indices of groups which can be merged.

        Parameters
        ----------
        slopes : np.ndarray, shape=(n_bins,)
            NumPy 1-D array with slopes of each bin.

        Returns
        -------
        np.ndarray, shape=(n_bins-1,)
            NumPy 1-D array with t values.
        """

        # calculate differences of consecutive group values and slopes
        group_difference = self._group_values[1:] - self._group_values[:-1]
        slope_difference = slopes[:-1] - slopes[1:]

        # divide group differences by slope differences
        # if slope differences are 0, set resulting value to inf
        t_values = np.divide(group_difference, slope_difference,
                             out=np.full_like(group_difference, np.inf, dtype=np.float),
                             where=slope_difference != 0)

        # add current lambda to t values
        t_values = t_values + + self._lambda

        return t_values

    @accepts(np.ndarray, float)
    def __update_group_values(self, slopes: np.ndarray, next_lambda: float):
        """
        Perform update of group values by given slopes and value of next lambda.

        Parameters
        ----------
        slopes : np.ndarray, shape=(n_bins,)
            NumPy 1-D array with slopes of each bin.
        next_lambda : float
            Lambda value of next model.
        """

        for i in range(self._num_groups):
            self._group_values[i] += slopes[i] * (next_lambda - self._lambda)

    @accepts(np.ndarray, float)
    def __merge_groups(self, t_values: np.ndarray, next_lambda: float):
        """
        Merge all groups where t_values is equal to next_lambda

        Parameters
        ----------
        t_values : np.ndarray, shape=(n_bins-1,)
            Current t-values.
        next_lambda : float
            Lambda value of next model.
        """

        # groups are denoted as t_i,i+1
        joined_groups = np.where(t_values == next_lambda)[0]
        upper_group = np.array(joined_groups + 1)

        # decrease group amount
        self._num_groups -= joined_groups.size

        # join group elements (ground truth and confidences)
        for i in joined_groups:
            self._group_items[i] = np.concatenate((self._group_items[i], self._group_items[i+1]))
            self._group_bounds[i, 1] = self._group_bounds[i+1, 1]

        new_items = [self._group_items[i] for i in range(len(self._group_items)) if i not in upper_group]
        self._group_items = new_items

        # prefer deletion of second group i+1
        self._group_values = np.delete(self._group_values, upper_group)
        self._group_bounds = np.delete(self._group_bounds, upper_group, axis=0)
