# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, Union, Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import pyro

from netcal.regression.gp import AbstractGP
from netcal.regression.gp.likelihood import BetaLikelihood


class GPBeta(AbstractGP):
    """
    GP-Beta recalibration method for regression uncertainty calibration using the well-known Beta calibration method
    from  classification calibration in combination with a Gaussian process (GP) parameter estimation.
    The basic idea of GP-Beta [1]_ is to apply recalibration on the uncalibrated cumulative density function (CDF),
    similar as to the Isotonic Regression method [2]_.
    Since the CDF is restricted to the :math:`[0, 1]` interval, the authors in [1]_ propose to use the
    Beta calibration scheme [3]_ known in the scope of confidence calibration.
    Furthermore, the authors use a GP to obtain the recalibration parameters of the Beta function for each
    sample individually, so that it should finally achieve *distribution calibration* [1]_.

    **Mathematical background:** Let :math:`f_Y(y)` denote the uncalibrated probability density function (PDF),
    targeting the probability distribution for :math:`Y`.
    Let :math:`\\tau_y \\in [0, 1]` denote a certain quantile on the uncalibrated CDF which
    is denoted by :math:`\\tau_y = F_Y(y)`.
    Furthermore, let :math:`g_Y(y)` and :math:`G_Y(y)` denote the recalibrated PDF and CDF, respectively.
    The Beta calibration function :math:`\\mathbf{c}_\\beta(\\tau_y)` known from [3]_ is given by

    .. math::
        \\mathbf{c}_\\beta(\\tau_y) = \\phi\\big( a \\log(\\tau_y) - b \\log(1-\\tau_y) + c \\big)

    with recalibration parameters :math:`a,b \\in \\mathbb{R}_{>0}` and :math:`c \\in \\mathbb{R}`, and
    :math:`\\phi(\\cdot)` as the sigmoid function [3]_.
    This method serves as a mapping from the uncalibrated CDF to the calibrated one, so that

    .. math::
        G_Y(y) = \\mathbf{c}_\\beta\\big( F_Y(y) \\big)

    holds. The PDF is the derivative of the CDF, so that the calibrated PDF is given by

    .. math::
        g_Y(y) = \\frac{\\partial \\mathbf{c}_\\beta}{\\partial y}
        = \\frac{\\partial \\mathbf{c}_\\beta}{\\partial \\tau_y} \\frac{\\partial \\tau_y}{\\partial y}
        = \\mathbf{r}_\\beta(\\tau_y) f_Y(y) ,

    with :math:`\\mathbf{r}_\\beta(\\tau_y)` as a beta link function [1]_ given by

    .. math::
        \\mathbf{r}_\\beta(\\tau_y) = \\Bigg(\\frac{a}{\\tau_y} + \\frac{b}{1-\\tau_y} \\Bigg)
        \\mathbf{c}_\\beta(\\tau_y) \\big(1 - \\mathbf{c}_\\beta(\\tau_y)\\big) .

    Finally, the recalibration parameters :math:`a, b` and :math:`c` are obtained using a Gaussian process scheme.
    In this way, it is possible to apply non-parametric *distribution calibration* [1]_.

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
    name_prefix: str, optional, default: "gpbeta"
        Name prefix internally used in Pyro to distinguish between parameter stores.

    References
    ----------
    .. [1] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
       "Distribution calibration for regression."
       International Conference on Machine Learning. PMLR, 2019.
       `Get source online <http://proceedings.mlr.press/v97/song19a/song19a.pdf>`__

    .. [2] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon:
       "Accurate uncertainties for deep learning using calibrated regression."
       International Conference on Machine Learning. PMLR, 2018.
       `Get source online <http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf>`__

    .. [3] Kull, Meelis, Telmo Silva Filho, and Peter Flach:
       "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
       Artificial Intelligence and Statistics, PMLR 54:623-631, 2017
       `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`__
    """

    precision = torch.float64

    def __init__(
            self,
            n_inducing_points: int = 12,
            n_random_samples: int = 128,
            *,
            name_prefix: str = "gpbeta",
            **kwargs
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        # call super constructor
        super().__init__(
            n_inducing_points=n_inducing_points,
            n_random_samples=n_random_samples,
            n_parameters=3,
            correlations=False,
            name_prefix=name_prefix,
            **kwargs
        )

        # set likelihood and number of parameters per dim
        self.likelihood = BetaLikelihood

        # use inv_gamma and delta to rescale sampled parameters before exponential
        self._learn_scale_shift = True

    # -------------------------------------------------------------------------
    # Pyro + GPyTorch functions

    def transform(
            self, X: Union[Iterable[np.ndarray], np.ndarray],
            t: Union[int, np.ndarray] = 512
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the given stddev to a distribution-calibrated one using the input
        mean and stddev as priors for the underlying Gaussian process.

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
        t : int or np.ndarray of shape (t, [n, [d]])
            Parameter to control the output sample distribution. If integer, the given amount of
            sample points for each input X is created. If np.ndarray, use this as distribution values for each
            sample and each dim in X at location t (either for all samples equally with shape (t,) or for each sample
            individually with shape (t, n, [d])).

        Returns
        -------
        tuple of 3 np.ndarray, each of shape (t, n, d)
            First array holds the points where the density and cumulative functions are defined, shape: (t, n ,d).
            Second array is the recalibrated probability density function (PDF), shape: (t, n, d).
            Third array is the recalibrated cumulative density function (CDF), shape: (t, n, d).
            Note that the data dimension d is always present (also for d=1).
        """

        # get mean, variance and stddev of input
        with torch.no_grad():
            Xmean, Xvariance = self._get_input(X)

        # concatenate mean and var of training points to a single pytorch tensor
        n_samples, n_dims = Xmean.shape
        X = torch.cat((Xmean, Xvariance), dim=1).to(dtype=self.precision)  # (n, 2*d)

        # -------------------------------------------------
        # check parameter t and prepare sampling points

        # if t is np.ndarray, use this points as the base for the calibrated output distribution
        if isinstance(t, np.ndarray):

            # distribute 1D array - assert only sampling points dim
            if t.ndim ==1:
                sampling_points = torch.from_numpy(t).reshape(t.shape[0], 1, 1)  # (t, 1, 1)
                sampling_points = sampling_points.expand(t.shape[0], n_samples, n_dims)  # (t, n, d)

            # on 2D, assert sampling points for each sample individually
            elif t.ndim == 2:
                sampling_points = torch.from_numpy(t).unsqueeze(dim=2)  # (t, n, 1)
                sampling_points = sampling_points.expand(t.shape[0], n_samples, n_dims)  # (t, n, d)

            elif t.ndim == 3:
                sampling_points = torch.from_numpy(t)  # (t, n, d)

            else:
                raise RuntimeError("Invalid shape for parameter \'t\'.")

            # guarantee monotonically increasing sampling points
            sampling_points, _ = torch.sort(sampling_points, dim=0)  # (t, n, d)

            # create PyTorch dataset and dataloader for optimization
            # we need to transpose the sampling points into (n, t, d) order so that TensorDataset
            # does not throw an error
            sampling_points = sampling_points.permute(1, 0, 2)
            sampling_points = sampling_points.to(dtype=self.precision)
            dataset = TensorDataset(X, sampling_points)

        else:
            dataset = TensorDataset(X)

        # -------------------------------------------------

        # initialize dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        # set model in eval mode and use variational distribution for inference
        self.eval()

        # work in distinct pyro parameter store
        with pyro.get_param_store().scope(self.pyro_scope):

            # move self and sampling points to the proper precision. Move self to proper device
            self.to(device=self.device, dtype=self.precision)

            # placeholders for final calibrated density and cumulative
            sampling_points_list, calibrated_density, calibrated_cumulative = [], [], []

            with torch.no_grad():
                for batch in dataloader:

                    # let the calibration method itself create the sampling points of the cumulative
                    # if only the amount of sampling points is given
                    if len(batch) == 1:
                        batch_X, batch_t = batch[0], t

                    # otherwise, use the given sampling points to build the recalibrated cumulative
                    else:
                        batch_X, batch_t = batch

                        batch_t = batch_t.to(device=self.device).permute(1, 0, 2)  # (t, n, d)
                        batch_t = batch_t.unsqueeze(dim=1)  # (t, 1, n, d)

                    # move input to GPU
                    n_batch = len(batch_X)
                    batch_X = batch_X.to(device=self.device)  # (n, 2*d)

                    # invoke GP model with learnt variational distribution
                    output = self(batch_X)

                    # sample from the variational distribution
                    function_samples = output(torch.Size([self.n_random_samples]))  # (r, n, 3*d)
                    function_samples = function_samples * self.inv_gamma + self.delta

                    # unsqueeze to distribute computation
                    x_loc = torch.reshape(batch_X[:, :n_dims], (1, 1, n_batch, n_dims))  # (1, 1, n, d)
                    x_logvar = torch.reshape(batch_X[:, n_dims:], (1, 1, n_batch, n_dims))  # (1, 1, n, d)
                    function_samples = function_samples.unsqueeze(0)  # (1, r, n, 3*d)

                    # init likelihood of beta link
                    likelihood = BetaLikelihood(x_loc, x_logvar, function_samples)

                    # get calibrated cumulative distribution
                    sampling_points_batch, calibrated_cumulative_batch = likelihood.cdf(batch_t)  # (t, 1, n, d), (t, r, n, d)
                    calibrated_cumulative_batch = torch.mean(calibrated_cumulative_batch, dim=1).to(dtype=torch.float32, device='cpu')  # (t, n, d)

                    # get calibrated density distribution
                    calibrated_density_batch = likelihood.log_prob(sampling_points_batch)  # (t, r, n, d)
                    calibrated_density_batch = torch.mean(torch.exp(calibrated_density_batch), dim=1).to(dtype=torch.float32, device='cpu')  # (t, n, d)

                    sampling_points_list.append(torch.squeeze(sampling_points_batch, dim=1).to(dtype=torch.float32, device="cpu"))
                    calibrated_density.append(calibrated_density_batch)
                    calibrated_cumulative.append(calibrated_cumulative_batch)

            # concatenate batch tensors to a single tensor
            sampling_points = torch.cat(sampling_points_list, dim=1)  # (t, n, d)
            calibrated_density = torch.cat(calibrated_density, dim=1)  # (t, n, d)
            calibrated_cumulative = torch.cat(calibrated_cumulative, dim=1)  # (t, n, d)

        return sampling_points.numpy(), calibrated_density.numpy(), calibrated_cumulative.numpy()
