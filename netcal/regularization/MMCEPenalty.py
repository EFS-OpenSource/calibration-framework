# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.
import torch
from torch.nn.modules.loss import _Loss


class MMCEPenalty(_Loss):
    """
    Maximum mean calibration error (MMCE) [1]_. This term can be used for online confidence calibration directly
    during model training.

    Parameters
    ----------
    weight : float
        Weight of MMCE regularization.

    References
    ----------
    .. [1] Kumar, Aviral, Sunita Sarawagi, and Ujjwal Jain:
       "Trainable calibration measures for neural networks from kernel mean embeddings."
       International Conference on Machine Learning. PMLR, 2018.
       `Get source online: <http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf>`_
    """

    epsilon = 1e-12

    def __init__(self, weight: float = 1.0):
        """ Constructor. For parameter description, see class docstring. """

        super().__init__()
        self.weight = weight

    def kernel(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """ Laplacian kernel """

        diff = c1[:, None] - c2
        return torch.exp(-2.5 * torch.abs(diff))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """ Forward call of module. Returns a single scalar indicating the MMCE for the current batch. """

        # assume logits as input
        probs, labels = torch.max(torch.softmax(input, dim=1), dim=1)
        probs = torch.clamp(probs, min=self.epsilon, max=1. - self.epsilon)

        matched = torch.where(labels == target, torch.ones_like(labels), torch.zeros_like(labels))
        n_samples = len(matched)
        n_correct = torch.sum(matched)

        # divide all probabilities by matched/not matched
        probs_false = probs[matched == 0]
        probs_correct = probs[matched == 1]

        # compute kernels between different combinations
        kernel_false = self.kernel(probs_false, probs_false)
        kernel_correct = self.kernel(probs_correct, probs_correct)
        kernel_mixed = self.kernel(probs_correct, probs_false)

        probs_false = torch.unsqueeze(probs_false, dim=1)
        inv_probs_correct = torch.unsqueeze(1. - probs_correct, dim=1)

        diff_false = torch.matmul(probs_false, probs_false.transpose(1, 0))
        diff_correct = torch.matmul(inv_probs_correct, inv_probs_correct.transpose(1, 0))
        diff_mixed = torch.matmul(inv_probs_correct, probs_false.transpose(1, 0))

        # MMCE calculation scheme (see paper for mathematical details)
        part_false = torch.sum(diff_false * kernel_false) / float((n_samples - n_correct) ** 2) if n_samples - n_correct > 0 else 0.
        part_correct = torch.sum(diff_correct * kernel_correct) / float(n_correct ** 2) if n_correct > 0 else 0.
        part_mixed = 2 * torch.sum(diff_mixed * kernel_mixed) / float((n_samples - n_correct) * n_correct) if (n_samples - n_correct) * n_correct > 0 else 0.

        mmce = self.weight * torch.sqrt(part_false + part_correct - part_mixed)

        return mmce
