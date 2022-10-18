# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Dict
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

import torch
import torch.distributions as dist

from netcal import mv_cauchy_log_density, density_from_cumulative


def draw_single_distribution(methods: Dict):

    x, y = np.linspace(-3, 1, 1000, dtype=np.float32), np.linspace(-3, 2, 1000, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xy = torch.from_numpy(np.stack((xx, yy), axis=2)).to(dtype=torch.float32)

    # try to display sample number 256
    idx = 256 % len(x)

    fig, axes = plt.subplots(nrows=len(methods), ncols=2, figsize=(10, 5 * len(methods)))
    for (ax1, ax2), (method, values) in zip(axes, methods.items()):

        # if "t" is in values, we have a cumulative function
        if "t" in values:
            t = values["t"][:, idx]  # (t, 2)
            cdf = values["cdf"][:, idx]  # (t, 2)

            # get pdf from cdf
            pdf = density_from_cumulative(t[:, None], cdf[:, None])  # (t, 1, 2)

            # interpolate pdf to image grid
            independent_density_x = interp1d(t[:, 0], pdf[:, 0, 0], kind="cubic", bounds_error=False, fill_value=0.)(x)  # (x,)
            independent_density_y = interp1d(t[:, 1], pdf[:, 0, 1], kind="cubic", bounds_error=False, fill_value=0.)(y)  # (y,)

            independent_density = np.stack(np.meshgrid(independent_density_x, independent_density_y, indexing="ij"), axis=2)  # (x, y, 2)
            independent_density = np.prod(independent_density, axis=2)  # (x, y)

            multivariate_density = independent_density

        # if "mode" is in values, we have Cauchy
        elif "mode" in values:
            mode = torch.from_numpy(values["mode"][idx])

            # univariate case without correlations
            if "scale" in values:
                scale = torch.from_numpy(values["scale"][idx])
                independent_density = dist.Independent(dist.Cauchy(loc=mode, scale=scale), 1).log_prob(xy).exp().numpy()
                multivariate_density = independent_density

            # multivariate case
            else:
                cov = torch.from_numpy(values["cov"][idx])
                scale = torch.diagonal(cov, offset=0, dim1=-2, dim2=-1)

                multivariate_density = mv_cauchy_log_density(xy, loc=mode, cov=cov).exp().numpy()
                independent_density = dist.Independent(dist.Cauchy(loc=mode, scale=scale), 1).log_prob(xy).exp().numpy()

        # otherwise, treat input as normal
        else:

            ymean = torch.from_numpy(values["mean"])[idx]
            ycov = torch.from_numpy(values["cov"])[idx]
            ycov_independent = torch.diag_embed(torch.diagonal(ycov, dim1=-2, dim2=-1))

            independent_density = dist.MultivariateNormal(loc=ymean, covariance_matrix=ycov_independent).log_prob(xy).exp().numpy()
            multivariate_density = dist.MultivariateNormal(loc=ymean, covariance_matrix=ycov).log_prob(xy).exp().numpy()

        ax1.contourf(xx, yy, independent_density)
        ax2.contourf(xx, yy, multivariate_density)

        ax1.set_title("%s - independent" % method)
        ax2.set_title("%s - multivariate" % method)

    return fig
