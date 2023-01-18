# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, Union
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import pyro

from netcal.regression.gp import AbstractGP
from netcal.regression.gp.likelihood import CauchyLikelihood


class GPCauchy(AbstractGP):
    """
    GP-Cauchy recalibration method for regression uncertainty calibration that consumes an uncalibrated Gaussian
    distribution but converts it to a calibrated Cauchy distribution.
    This method uses a Gaussian process (GP) for a flexible estimation of the recalibration parameter (cf. [1]_).
    Similar to :class:`netcal.regression.gp.GPNormal`, the GP-Cauchy [2]_ acts as a kind of temperature scaling for
    the variance of a Gaussian distribution [3]_, [4]_. However, the rescaled variance is interpreted as the
    scaling parameter of a Cauchy distribution. Furthermore, the rescaling parameter is not globally fixed but obtained
    by a GP for each sample individually, so that we are able to convert a Gaussian to a Cauchy distribution [2]_.
    Thus, the GP-Cauchy seeks for **distribution calibration** but for parametric Cauchy distributions.
    Note that this method does not change the mean but only reinterprets the predicted variance as the Cauchy scaling
    parameter. The mode of the Cauchy is assumed to be equal to the input mean.

    **Mathematical background:** Let :math:`f_Y(y)` denote the uncalibrated probability density function (PDF),
    targeting the probability distribution for :math:`Y`. In our case, the uncalibrated PDF is given as a Gaussian, so
    that :math:`f_Y(y) = \\mathcal{N}\\big(y; \\mu_Y(X), \\sigma^2_Y(X)\\big)` with mean :math:`\\mu_Y(X)` and variance
    :math:`\\sigma^2_Y(X)` obtained by a probabilistic regression model that depends on the input :math:`X`.
    The calibrated PDF :math:`g_Y(y)` is a rescaled Cauchy distribution with fixed mode :math:`x_0 \\in \\mathbb{R}`
    and rescaled scaling parameter :math:`\\lambda \\in \\mathbb{R}_{>0}`, so that

    .. math::
        g_Y(y) = \\text{Cauchy}\\Big(y; x_0=\\mu_Y(X), \\lambda=\\big(\\theta_y \\cdot \\sigma_Y(X)\\big) \\Big) ,

    where  :math:`\\theta_y` is the adaptive rescaling weight for a certain :math:`y`.

    The GP-Cauchy utilizes a Gaussian process to obtain :math:`\\theta_y`, so that

    .. math::
        \\theta_y \\sim \\text{gp}(0, k) ,

    where :math:`k` is the kernel function (for a more detailed description of the underlying Gaussian process, see
    documentation of parent class :class:`netcal.regression.gp.AbstractGP`).

    Parameters
    ----------
    n_inducing_points: int
        Number of inducing points used to approximate the input space. These inducing points are also optimized.
    n_random_samples: int
        Number of random samples used to sample from the parameter distribution during optimization and inference.
    n_epochs: int, default: 200
        Number of optimization epochs.
    batch_size: int, default: 256
        Size of batches during optimization.
    num_workers : int, optional, default: 0
        Number of workers used for the dataloader.
    lr: float, optional, default: 1e-2
        Learning rate used for the Adam optimizer.
    use_cuda: str or bool, optional, default: False
        The optimization and inference might also run on a CUDA device. If True, use the first available CUDA device.
        You can also pass a string "cuda:0", "cuda:1", etc. to specify the CUDA device.
        If False, use CPU for optimization and inference.
    jitter: float, optional, default: 1e-5
        Small digit that is added to the diagonal of a covariance matrix to stabilize Cholesky decomposition during
        Gaussian process optimization.
    name_prefix: str, optional, default: "gpcauchy"
        Name prefix internally used in Pyro to distinguish between parameter stores.

    References
    ----------
    .. [1] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
       "Distribution calibration for regression."
       International Conference on Machine Learning. PMLR, 2019.
       `Get source online <http://proceedings.mlr.press/v97/song19a/song19a.pdf>`__

    .. [2] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__

    .. [3] Levi, Dan, et al.:
       "Evaluating and calibrating uncertainty prediction in regression tasks."
       arXiv preprint arXiv:1905.11659 (2019).
       `Get source online <https://arxiv.org/pdf/1905.11659.pdf>`__

    .. [4] Laves, Max-Heinrich, et al.:
       "Well-calibrated regression uncertainty in medical imaging with deep learning."
       Medical Imaging with Deep Learning. PMLR, 2020.
       `Get source online <http://proceedings.mlr.press/v121/laves20a/laves20a.pdf>`__
    """

    precision = torch.float64

    def __init__(
            self,
            n_inducing_points: int = 12,
            n_random_samples: int = 128,
            *,
            name_prefix: str = "gpcauchy",
            **kwargs
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        # call super constructor
        super().__init__(
            n_inducing_points=n_inducing_points,
            n_random_samples=n_random_samples,
            n_parameters=1,
            correlations=False,
            name_prefix=name_prefix,
            **kwargs
        )

        # set likelihood and number of parameters per dim
        self.likelihood = CauchyLikelihood

        # do not use inv_gamma and delta to rescale sampled parameters before exponential
        self._learn_scale_shift = False

    # -------------------------------------------------------------------------
    # Pyro + GPyTorch functions

    def transform(self, X: Union[Iterable[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Transform the given stddev to a distribution-calibrated one using the input
        mean and stddev as priors for the underlying Gaussian process. If correlation=True, perform either recalibration
        of given covariance matrices or learn local correlations between dimensions if only standard deviations
        are provided as input. In this case, the function returns covariance matrices for each input sample.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or Tuple of two np.ndarray, each of shape (n, [d])
            Input data for calibration regression obtained by a model that performs inference with uncertainty.
            Depending on the input format, this method handles the input differently:
            If X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
            first array as mean and the second one as the according stddev predictions with d dimensions (optionally).
            If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
            inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
            case, the mean and stddev is computed automatically.

        Returns
        -------
        np.ndarray of shape (n, d) or (n, d, d)
            Recalibrated standard deviation for each sample in each dimension. If correlations=True, return
            estimated/recalibrated covariance matrices for each input sample in n.
        """

        # get mean, variance and stddev of input
        with torch.no_grad():
            Xmean, Xvariance = self._get_input(X)

        # concatenate mean and var of training points to a single pytorch tensor
        n_dims = Xmean.shape[1]
        X = torch.cat((Xmean, Xvariance), dim=1).to(dtype=self.precision)  # (n, 2*d)

        # create PyTorch dataset and dataloader for optimization
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        # set model in eval mode and use variational distribution for inference
        self.eval()

        # work in distinct pyro parameter store
        with pyro.get_param_store().scope(self.pyro_scope):

            # move self to the proper device and precision
            self.to(device=self.device, dtype=self.precision)

            # placeholder for final calibrated scale
            calibrated_scale = []

            # set model in eval mode and use variational distribution for inference
            self.eval()
            with torch.no_grad():
                for batch_X, in dataloader:

                    # move input to GPU
                    batch_X = batch_X.to(device=self.device)

                    # invoke GP model with learnt variational distribution - output is MultivariateNormal
                    output = self(batch_X)

                    # sample from the variational distribution
                    function_samples = output(torch.Size([self.n_random_samples]))  # (r, n, d) or (r, n, (d+d^2)/2)

                    # unsqueeze to distribute computation
                    x_logvar = torch.unsqueeze(batch_X[:, n_dims:], dim=0)  # (1, n, d)
                    x_std = torch.exp(x_logvar / 2.)

                    cauchy_scale = x_std * torch.exp(function_samples)  # (r, n, d)
                    cauchy_scale = torch.mean(cauchy_scale, dim=0)

                    calibrated_scale.append(cauchy_scale.to(dtype=torch.float32, device="cpu"))

            calibrated_scale = torch.cat(calibrated_scale, dim=0)

        return calibrated_scale.numpy()
