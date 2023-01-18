# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import torch
import pyro.distributions as dist


class ScaledNormalLikelihood(dist.Normal):
    """
    Standard likelihood of a normal distribution used within :class:`netcal.regression.gp.GPNormal`
    with additional rescaling parameters for the variance.
    This method provides the likelihood for the :class:`netcal.regression.gp.GPNormal` by applying a rescaling
    of the uncalibrated input variance by a recalibration parameter :math:`\\theta_y` [1]_, so that the recalibrated
    probability density function (PDF) :math:`g_Y(y)` is given by

    .. math::
        g_Y(y) = \\mathcal{N}\\Big(y; \\mu_Y(X), \\big(\\theta_y \\cdot \\sigma_Y(X)\\big)^2\\Big)

    with :math:`\\mu_Y(X) \\in \\mathbb{R}` and :math:`\\sigma_Y(X) \\in \\mathbb{R}_{>0}` as the uncalibrated
    mean and standard deviation, respectively.

    Parameters
    ----------
    loc: torch.Tensor, shape: ([1], n, d)
        Mean vector of the uncalibrated (Gaussian) probability distribution with n samples and d dimensions.
        This class also supports broadcasting of the calculations to r random samples (first dim).
    logvar: torch.Tensor, shape: ([1], n, d)
        Log of the variance vector of the uncalibrated (Gaussian) probability distribution with n samples and d
        dimensions. This class also supports broadcasting of the calculations to r random samples (first dim).
    parameters: torch.Tensor, shape: ([r], n, d)
        Rescaling parameter of the GP-Normal with n samples d dimensions and r random samples.

    References
    ----------
    .. [1] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__
    """

    def __init__(self, loc: torch.Tensor, logvar: torch.Tensor, parameters: torch.Tensor):
        """ Constructor. For detailed parameter description, see class docs. """

        var = torch.exp(logvar)
        transformed_var = var * parameters.exp()
        super().__init__(loc, torch.sqrt(transformed_var), validate_args=False)

    def expand(self, batch_shape, _instance=None):
        """ Expand-method. Reimplementation required when sub-classing the Normal distribution of Pyro. """

        new = self._get_checked_instance(ScaledNormalLikelihood, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(ScaledNormalLikelihood, new).__init__(new.loc, new.scale, validate_args=False)
        new._validate_args = self._validate_args

        return new
