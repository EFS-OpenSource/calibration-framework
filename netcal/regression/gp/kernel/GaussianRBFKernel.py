# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import torch
import gpytorch


class GaussianRBFKernel(gpytorch.kernels.Kernel):
    """
    Computes the Gaussian RBF kernel using input points defined as Gaussians.
    This kernel has been proposed by [1]_.
    Given input samples :math:`\\mathbf{x}_i, \\mathbf{x}_j \\in \\mathbb{R}^d` of dimension :math:`d` with
    according covariances :math:`\\Sigma_i, \\Sigma_j \\in \\mathbb{R}^{d \\times d}` that define a
    Gaussian for each input point. The kernel function is defined as

    .. math::
       k\\Big((\\mathbf{x}_i, \\Sigma_i), (\\mathbf{x}_j, \\Sigma_j)\\Big) = \\theta^d |\\Sigma_{ij}|^{-\\frac{1}{2}} \\exp \\Bigg( -\\frac{1}{2} (\\mathbf{x}_i-\\mathbf{x}_j)^\\top
       \\Sigma_{ij}^{-1}(\\mathbf{x}_i-\\mathbf{x}_j)  \\Bigg) ,

    with

    .. math::
       \\Sigma_{ij} = \\Sigma_i + \\Sigma_j + \\theta^2 \\mathbf{I} ,

    where :math:`\\theta \\in \\mathbb{R}` is a lengthscale parameter. Since we're only using
    independent normal distributions for each dimension, we only have diagonal covariance matrices.

    References
    ----------
    .. [1] Song, L., Zhang, X., Smola, A., Gretton, A., & Schölkopf, B.:
        "Tailoring density estimation via reproducing kernel moment matching."
        In Proceedings of the 25th international conference on Machine learning (pp. 992-999), July 2008.
        `Get source online <https://www.cs.uic.edu/~zhangx/pubDoc/xinhua_icml08.pdf>`__
    """

    has_lengthscale = True

    def __init__(self, *args, cov: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov = cov

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Computes the Gaussian moment-matching RBF kernel. Asserts both input variables as
        2-D tensors holding the mean and variance matrices for each input point:
        x1, x2 with shape (n, 2*d) and (m, 2*d), where n is number of samples in x1 and
        m is the number of samples in x2, and d is number of feature dimensions.
        Thus, mean of x1, x2 is given in (n, 0:d) and the variance in (n, d:).

        Parameters
        ----------
        x1 : torch.Tensor of shape (n, 2*d)
            First set of samples used for kernel computation consisting of mean and variance
            for each input point.
        x2 : torch.Tensor of shape (m, 2*d)
            Second set of samples used for kernel computation consisting of mean and variance
            for each input point.
        diag : bool, optional, default: False
            If True, return only the diagonal elements of the covariance matrix.

        Returns
        -------
        torch.Tensor of shape (n, m)
            Covariance kernel matrix for each input pair.
        """

        assert x1.ndim > 1, "GaussianRBFKernel: x1 must be at least 2-D with shape ([b], n, 2*d) " \
                            "(cov=False) or ([b], n, d + (d^2+d)//2) (cov=True)"
        assert x2.ndim > 1, "GaussianRBFKernel: x2 must be at least 2-D with shape ([b], n, 2*d) " \
                            "(cov=False) or ([b], n, d + (d^2+d)//2) (cov=True)"

        if self.cov:
            # formula to reconstruct the true data dimension from mean + lower triangular cov shape
            n_dims = int((np.sqrt(8 * x1.shape[-1] + 9) - 3) // 2)

            # construct triangular lower matrices
            tril_idx = torch.tril_indices(row=n_dims, col=n_dims, offset=0)
            x1_var = torch.zeros(*x1.shape[:-1], n_dims, n_dims, dtype=x1.dtype, device=x1.device)  # ([b], n, d, d)
            x2_var = torch.zeros(*x2.shape[:-1], n_dims, n_dims, dtype=x2.dtype, device=x2.device)  # ([b], n, d, d)

            # insert parameters into lower triangular
            x1_var[..., tril_idx[0], tril_idx[1]] = x1[..., n_dims:]
            x2_var[..., tril_idx[0], tril_idx[1]] = x2[..., n_dims:]

            # finally, reconstruct covariance matrices by LL^T computation
            x1_var = torch.matmul(x1_var, x1_var.transpose(-2, -1))  # ([b], n, d, d)
            x2_var = torch.matmul(x2_var, x2_var.transpose(-2, -1))  # ([b], n, d, d)

            # construct sum over all covariance matrices
            variances = x1_var[..., None, :, :] + x2_var[..., None, :, :, :] # ([b], n, m, d, d)

            # add lengthscale parameter to diagonal
            diagonal = torch.diagonal(variances, offset=0, dim1=-2, dim2=-1)
            diagonal[:] = diagonal + torch.square(self.lengthscale)

            # get inverse of covariance matrices
            inv_cov = torch.linalg.inv(variances)  # ([b], n, m, d, d)

            # get determinant of covariance matrices
            determinant = torch.linalg.det(variances)  # ([b], n, m)

        else:
            n_dims = x1.shape[-1] // 2
            x1_var = torch.exp(x1[..., n_dims:])  # ([b], n, d)
            x2_var = torch.exp(x2[..., n_dims:])  # ([b], m, d)

            # first, add all variances within a single batch and add squared lengthscale to each dimension
            # the inverse covariance matrix is the inverse of the variances on the diagonal
            variances = x1_var[..., None, :] + x2_var[..., None, :, :] + torch.square(self.lengthscale)  # ([b], n, m, d)
            inv_cov = torch.diag_embed(1. / variances)  # ([b], n, m, d, d)

            # since we only have a diagonal covariance matrix, take the product as the determinant
            determinant = torch.prod(variances, dim=-1)  # ([b], n, m)

        # get mean estimates
        x1_loc = torch.unsqueeze(x1[..., :n_dims], dim=-1)  # ([b], n, d, 1)
        x2_loc = torch.unsqueeze(x2[..., :n_dims], dim=-1)  # ([b], m, d, 1)

        # compute mahalanobis distance
        mean_diff = x1_loc[..., None, :, :] - x2_loc[..., None, :, :, :]  # ([b], n, m, d, 1)
        mahalanobis = torch.matmul(
            mean_diff.transpose(dim0=-1, dim1=-2),  # ([b], n, m, 1, d)
            torch.matmul(inv_cov, mean_diff)  # ([b], n, m, d, 1)
        )  # ([b], n, m, 1, 1)

        # finally, put mahalanobis into exponential and squeeze the two last dims
        exponent = torch.exp(-0.5 * mahalanobis)[..., 0, 0]  # ([b], n, m)
        res = torch.pow(self.lengthscale, n_dims) * torch.div(exponent, torch.sqrt(determinant))  # ([b], n, m)

        if not diag:
            return res
        else:
            if torch.is_tensor(res):
                return res.diagonal(dim1=-1, dim2=-2)
            else:
                return res.diag()  # For LazyTensor
