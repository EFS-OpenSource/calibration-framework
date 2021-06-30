# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import torch
from torch.nn.modules.loss import _Loss


class DCAPenalty(_Loss):
    """
    Difference between Confidence and Accuracy (DCA) [1]_. This regularization returns a single scalar indicating
    the difference between mean confidence and accuracy within a single batch.

    Parameters
    ----------
    weight : float
        Weight of DCA regularization.

    References
    ----------
    .. [1] Liang, Gongbo, et al.:
       "Improved trainable calibration method for neural networks on medical imaging classification."
       arXiv preprint arXiv:2009.04057 (2020).
    """

    def __init__(self, weight: float = 1.0):
        """ Constructor. For parameter description, see class docstring. """

        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """ Forward call of module. Providing the target scores is mandatory. """

        # assume logits as input
        probs, labels = torch.max(torch.softmax(input, dim=1), dim=1)

        # get batch accuracy
        matched = torch.where(labels == target, torch.ones_like(labels), torch.zeros_like(labels))
        acc = torch.mean(matched.detach().to(torch.float32))

        # DCA is absolute difference between batch accuracy and mean confidence
        dca = torch.abs(acc - torch.mean(probs))

        return dca
