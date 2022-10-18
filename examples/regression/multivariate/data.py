# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple
import numpy as np
from scipy.special import expit as sigmoid
import torch
import torch.distributions as dist


def generate(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate samples of an artificial base estimator that are not miscalibrated.
    """

    x1, x2 = np.linspace(-5, 5, n_samples, dtype=np.float32), np.linspace(-5, 5, n_samples, dtype=np.float32)
    x = np.stack((x1, x2), axis=-1)

    y1_mean = np.sin(x1)
    y2_mean = np.cos(x2)
    ymean = np.stack((y1_mean, y2_mean), axis=-1)

    corr = np.tanh(x1) * np.tanh(x2)

    y1_var = sigmoid(x1 * 0.5)
    y2_var = 1. - sigmoid(x1 * 0.25)
    ycovar = corr * np.sqrt(y1_var * y2_var)
    yvar = np.stack((y1_var, y2_var), axis=-1)

    ycov = torch.diag_embed(torch.from_numpy(yvar))
    ycov[..., 0, 1] = ycov[..., 1, 0] = torch.from_numpy(ycovar)

    mvn = dist.MultivariateNormal(loc=torch.from_numpy(ymean), covariance_matrix=ycov)
    y = mvn.sample()

    return x, y.numpy(), ymean, ycov.numpy()
