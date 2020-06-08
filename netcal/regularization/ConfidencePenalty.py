# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np

from netcal import global_dimensions


@global_dimensions((1, 2), None, None, None)
def confidence_penalty(X: np.ndarray, weight: float, threshold: float = None, base: float = np.e) -> float:
    """
    Confidence Penalty Regularization. This penalty term can be applied to any loss function as a regularizer [1]_.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, [n_classes])
        NumPy array with confidence values for each prediction.
        1-D for binary classification, 2-D for multi class (softmax).
    weight : float
        Weight of entropy.
    threshold : float, optional, default: None
        Entropy threshold (no penalty is assigned above threshold).
    base : float, optional, default: np.e
        Base of logarithm (typically the number of classes to norm entropy).

    Returns
    -------
    float
        Confidence penalty of posterior distribution.

    References
    ----------
    .. [1] G. Pereyra, G. Tucker, J. Chorowski, Lukasz Kaiser, and G. Hinton:
       “Regularizing neural networks by penalizing confident output distributions.”
       CoRR, 2017.
       `Get source online <https://arxiv.org/pdf/1701.06548>`_
    """

    epsilon = np.finfo(np.float32).eps
    confidence = np.clip(X, epsilon, 1.-epsilon)

    # calculate entropy of each sample and sum all samples afterwards
    sample_entropy = -1. * np.sum(np.multiply(confidence, np.log(confidence) / np.log(base)), axis=-1)

    # set entropy threshold if given
    if threshold is not None:
        sample_entropy = np.maximum(0.0, threshold - sample_entropy)

    entropy = np.mean(sample_entropy)

    # weight entropy by penalty
    penalty = weight * -entropy

    return penalty
