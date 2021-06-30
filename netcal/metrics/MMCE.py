# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable
import logging
from typing import Union
import numpy as np

from netcal import accepts


class MMCE(object):
    """
    Maximum Mean Calibration Error (MMCE) [1]_.
    A differentiable approximation to the Expected Calibration Error (ECE) using a
    reproducing _kernel Hilbert space (RKHS).
    Using a dataset :math:`\\mathcal{D}` of size :math:`N` consisting of the ground truth labels :math:`\\hat{y} \\in \\{1, ..., K \\}`
    with input :math:`\\hat{y} \\in \\mathcal{X}`, the MMCE is calculated by using a scoring classifier :math:`\\hat{p}=h(x)`
    that returns the highest probability for a certain class in conjunction with the predicted label
    information :math:`y \\in \\{1, ..., K \\}` and is defined by

    .. math::

       MMCE = \\sqrt{\\sum_{i, j \\in \\mathcal{D}} \\frac{1}{N^2}(\\mathbb{1}(\\hat{y}_i = y_i) - \\hat{p}_i) (\\mathbb{1}(\\hat{y}_j = y_j) - \\hat{p}_j)k(\\hat{p}_i, \\hat{p}_j)} ,

    with :math:`\\mathbb{1}(*)` as the indicator function and a Laplacian _kernel :math:`k` defined by

    .. math::
       k(\\hat{p}_i, \\hat{p}_j) = \\exp(-2.5 |\\hat{p}_i - \\hat{p}_j|) .

    Parameters
    ----------
    detection : bool, default: False
        Detection mode is currently not supported for MMCE!
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).

    References
    ----------
    .. [1] Kumar, Aviral, Sunita Sarawagi, and Ujjwal Jain:
       "Trainable calibration measures for neural networks from _kernel mean embeddings."
       International Conference on Machine Learning. 2018.
       `Get source online <http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf>`_.
    """

    @accepts(bool)
    def __init__(self, detection: bool = False):
        """ Constructor. For parameter doc see class doc. """

        self.logger = logging.getLogger('calibration')

        assert not detection, "MMCE is currently not supported for object detection."
        self.detection = detection

    def _batched(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray], batched: bool = False):
        # batched: interpret X and y as multiple predictions

        if not batched:
            assert isinstance(X, np.ndarray), 'Parameter \'X\' must be Numpy array if not on batched mode.'
            assert isinstance(y, np.ndarray), 'Parameter \'y\' must be Numpy array if not on batched mode.'
            X, y = [X], [y]

        # if we're in batched mode, create new lists for X and y to prevent overriding
        else:
            assert isinstance(X, (list, tuple)), 'Parameter \'X\' must be type list on batched mode.'
            assert isinstance(y, (list, tuple)), 'Parameter \'y\' must be type list on batched mode.'
            X, y = [x for x in X], [y_ for y_ in y]

        # if input X is of type "np.ndarray", convert first axis to list
        # this is necessary for the following operations
        if isinstance(X, np.ndarray):
            X = [x for x in X]

        if isinstance(y, np.ndarray):
            y = [y0 for y0 in y]

        return X, y

    def _kernel(self, confidence):
        """ Laplacian _kernel """

        diff = confidence[:, None] - confidence
        return np.exp(-2.5 * np.abs(diff))

    def measure(self, X: Union[Iterable[np.ndarray], np.ndarray], y: Union[Iterable[np.ndarray], np.ndarray], batched: bool = False):
        """
        Measure calibration by given predictions with confidence and the according ground truth.

        Parameters
        ----------
        X : iterable of np.ndarray, or np.ndarray of shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            If this is an iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        y : iterable of np.ndarray with same length as X or np.ndarray of shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            If iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.

        Returns
        -------
        float
            Returns Maximum Mean Calibration Error.
        """

        X, y = self._batched(X, y, batched)

        mmce = []
        for X_batch, y_batch in zip(X, y):

            # assert y_batch is one-hot with 2 dimensions
            if y_batch.ndim == 2:
                y_batch = np.argmax(y_batch, axis=1)

            # get max confidence and according label
            if X_batch.ndim == 1:
                confidence, labels = X_batch, np.where(X_batch > 0.5, np.ones_like(X_batch), np.zeros_like(X_batch))
            elif X_batch.ndim == 2:
                confidence, labels = np.max(X_batch, axis=1), np.argmax(X_batch, axis=1)
            else:
                raise ValueError("MMCE currently not defined for input arrays with ndim>3.")

            n_samples = float(confidence.size)

            # get matched flag and difference
            matched = (y_batch == labels).astype(np.float)
            diff = np.expand_dims(matched - confidence, axis=1)

            # now calculate product of differences for each pair
            confidence_pairs = np.matmul(diff, diff.T)

            # caculate _kernel for each pair
            kernel_pairs = self._kernel(confidence)

            miscalibration = np.sqrt(np.sum(confidence_pairs * kernel_pairs) / np.square(n_samples))
            mmce.append(miscalibration)

        mmce = np.mean(mmce)
        return mmce
