# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import torch
import pyro.distributions as dist


class ScaledMultivariateNormalLikelihood(dist.MultivariateNormal):
    """
    Standard likelihood of a **multivariate** normal distribution used within :class:`netcal.regression.gp.GPNormal`
    for *covariance estimation* and *covariance recalibration*.
    This method provides the multivariate likelihood for the :class:`netcal.regression.gp.GPNormal` when
    "correlations=True".
    The input to this class is given as the LDL* decomposed (uncalibrated) covariance matrix
    :math:`\\boldsymbol{\\Sigma}_{\\mathbf{Y}}(X) = \\mathbf{L}\\mathbf{D}\\mathbf{L}^{\\top}`.
    Afterwards, this method applies a rescaling of the uncalibrated :math:`\\mathbf{L} \\in \\mathbb{R}^{d \\times d}`
    and :math:`\\mathbf{D} \\in \\mathbb{R}^{d \\times d}` matrices by the
    recalibration parameters :math:`\\theta_L \\in \\mathbb{R}^{d \\times d}` and
    :math:`\\theta_D \\in \\mathbb{R}^{d \\times d}_{>0}` [1]_, so that the recalibrated
    probability density function (PDF) :math:`g_{\\mathbf{Y}}(\\mathbf{y})` is given by

    .. math::
        g_{\\mathbf{Y}}(\\mathbf{y}) = \\mathcal{N}\\Big(\\mathbf{y}; \\boldsymbol{\\mu}_{\\mathbf{Y}}(X),
        (\\theta_L \\odot \\mathbf{L})(\\theta_D \\odot \\mathbf{D})(\\theta_L \\odot \\mathbf{L})^{\\top} \\big) \\Big)

    with :math:`\\boldsymbol{\\mu}_{\\mathbf{Y}}(X) \\in \\mathbb{R}` as the uncalibrated mean and :math:`\\odot` as
    the element-wise matrix multiplication operator.

    Parameters
    ----------
    loc: torch.Tensor, shape: ([1], n, d)
        Mean vector of the uncalibrated (Gaussian) probability distribution with n samples and d dimensions.
        This class also supports broadcasting of the calculations to r random samples (first dim).
    var: torch.Tensor, shape: ([1], n, (d^2+d) // 2)
        LDL* decomposed covariance matrix (compressed form) of the uncalibrated (Gaussian) probability distribution
        with n samples and d dimensions. This class also supports broadcasting of the calculations to r random samples
        (first dim).
    parameters: torch.Tensor, shape: ([r], n, (d+d^2) // 2)
        Rescaling parameter of the GP-Normal for the decomposed covariance matrices with n samples d dimensions
        and r random samples.

    References
    ----------
    .. [1] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__
    """

    def __init__(self, loc: torch.Tensor, var: torch.Tensor, parameters: torch.Tensor):
        """ Constructor. For detailed parameter description, see class docs. """

        # reconstruct and rescale the given input covariance matrices
        covariance_matrix = self.rescale(var, parameters)

        super().__init__(loc=loc, covariance_matrix=covariance_matrix, validate_args=False)

    @staticmethod
    def reconstruct(decomposed: torch.Tensor):
        """
        Reconstruct the lower :math:`\\mathbf{L}` and diagonal :math:`\\mathbf{D}` matrices from the flat input vector
        given by "decomposed" since Pyro does not support an additional dimension internally.

        Parameters
        ----------
        decomposed: torch.Tensor, shape: ([1], n, (d^2+d) // 2)
            LDL* decomposed covariance matrix (compressed form) of the uncalibrated (Gaussian) probability distribution
            with n samples and d dimensions.

        Returns
        -------
        Tuple of torch.Tensor with (L, D)
            Reconstructed lower triangular matrix and diagonal vector of the decomposed input covariance matrix.
            L has shape ([1], n, d, d) and is lower triangualr.
            D has shape ([1], n, d) and holds the diagonal elements.
        """

        # decomposed has shape (..., (d^2+d)//2)
        # formula to reconstruct the true data dimension from mean + lower triangular cov shape
        n_dims = int((np.sqrt(8 * decomposed.shape[-1] + 1) - 1) // 2)

        # use LDL* decomposition but reconstruct input var from LL*
        # get lower triangular indices (L matrix)
        tril_idx = torch.tril_indices(row=n_dims, col=n_dims, offset=0)

        # initialize identity matrix and place covariance estimates in the lower off-diagonal
        lower = torch.zeros(
            *decomposed.shape[:-1], n_dims, n_dims,
            dtype=decomposed.dtype,
            device=decomposed.device
        )  # (..., d, d)

        lower[..., tril_idx[0], tril_idx[1]] = decomposed
        diagonal_view = torch.diagonal(lower, dim1=-2, dim2=-1)  # (..., d)
        diagonal = diagonal_view.clone()

        # since diagonal is simply a view on the lower, place ones inplace
        diagonal_view[:] = 1.

        return lower, diagonal

    @staticmethod
    def rescale(var: torch.Tensor, parameters: torch.Tensor):
        """
        This method applies the rescaling of the LDL* decomposed covariance matrices by computing
        :math:`\\theta_L \\odot \\mathbf{L}` and :math:`\\theta_D \\odot \\mathbf{D}`.
        Finally, the covariance matrix is recovered and returned.

        Parameters
        ----------
        var: torch.Tensor, shape: ([1], n, (d^2+d) // 2)
            LDL* decomposed covariance matrix (compressed form) of the uncalibrated (Gaussian) probability distribution
            with n samples and d dimensions.
        parameters: torch.Tensor, shape: ([r], n, (d+d^2) // 2)
            Rescaling parameter of the GP-Normal for the decomposed covariance matrices with n samples d dimensions
            and r random samples.

        Returns
        -------
        torch.Tensor of shape ([r], n, d, d)
            Rescaled covariance matrices with r stochastic forward passes for the rescaling
            parameters, n samples and d dimensions.
        """

        lower_cov, diagonal_var = ScaledMultivariateNormalLikelihood.reconstruct(var)
        lower_scale, diagonal_scale = ScaledMultivariateNormalLikelihood.reconstruct(parameters)

        # diagonal var returns the standard devs on the diagonal
        # use square method to reconstruct variances
        diagonal_var = torch.square(diagonal_var)

        # use exponential on diagonal variance scaling parameters to guarantee positive estimates
        diagonal_scale = torch.exp(diagonal_scale)

        # perform element-wise multiplication
        lower_scale = lower_scale * lower_cov  # ([r], n, d, d)

        # rescale diagonal variances
        diagonal_scale = diagonal_scale * diagonal_var  # ([r], n, d)
        diagonal_scale = torch.diag_embed(diagonal_scale)  # ([r], n, d, d)

        # construct covariance matrix using LDL*
        covariance_matrix = lower_scale @ diagonal_scale @ torch.transpose(lower_scale, dim0=-2, dim1=-1)  # ([r], n, d, d)

        return covariance_matrix

    def expand(self, batch_shape, _instance=None):
        """ Expand-method. Reimplementation required when sub-classing the MultivariateNormal distribution of Pyro. """

        new = self._get_checked_instance(ScaledMultivariateNormalLikelihood, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
            super(ScaledMultivariateNormalLikelihood, new).__init__(loc=new.loc, covariance_matrix=new.covariance_matrix, validate_args=False)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
            super(ScaledMultivariateNormalLikelihood, new).__init__(loc=new.loc, scale_tril=new.scale_tril, validate_args=False)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
            super(ScaledMultivariateNormalLikelihood, new).__init__(loc=new.loc, precision_matrix=new.precision_matrix, validate_args=False)

        new._validate_args = self._validate_args
        return new
