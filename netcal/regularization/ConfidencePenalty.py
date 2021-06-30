# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

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


class ConfidencePenalty(_Loss):
    """
    Confidence penalty regularization implementation for PyTorch.
    This penalty term can be applied to any loss function as a regularizer [1]_.

    Parameters
    ----------
    weight : float
        Weight of entropy.
    threshold : float, optional, default: None
        Entropy threshold (no penalty is assigned above threshold).
    reduction : 'str'
        Specifies the reduction to apply to the output.

    References
    ----------
    .. [1] G. Pereyra, G. Tucker, J. Chorowski, Lukasz Kaiser, and G. Hinton:
       “Regularizing neural networks by penalizing confident output distributions.”
       CoRR, 2017.
       `Get source online <https://arxiv.org/pdf/1701.06548>`_
    """

    epsilon = 1e-12

    def __init__(self, weight: float = 1.0, threshold: float = -1., reduction='mean'):
        """ Constructor. For parameter description, see class docstring. """

        super().__init__(reduction=reduction)
        self.weight = weight
        self.threshold = threshold

    def forward(self, input: torch.Tensor):
        """ Forward call. Additional arguments and keyword-arguments are ignored. """

        probs = torch.clamp(torch.softmax(input, dim=1), self.epsilon, 1.-self.epsilon)

        # calculate entropy of each sample and sum all samples afterwards
        loss = -self.weight * torch.mul(probs, torch.log(probs))

        # set entropy threshold if given
        if self.threshold > 0:
            loss = torch.maximum(0.0, self.threshold - loss)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'batchmean':
            return torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise AttributeError("Unknown reduction type \'%s\'." % self.reduction)
