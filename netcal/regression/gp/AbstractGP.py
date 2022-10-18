# Copyright (C) 2021-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from tqdm import tqdm
from copy import deepcopy
from typing import Tuple, Iterable, Union, Optional, Dict
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import pyro
import gpytorch

from netcal import AbstractCalibration, meanvar
from netcal.regression.gp.kernel import GaussianRBFKernel


class AbstractGP(AbstractCalibration, gpytorch.models.ApproximateGP):
    """
    Distribution recalibration of regression models using a Gaussian process parameter estimation.
    The goal of regression calibration using a GP scheme is to achieve *distribution calibration*,
    i.e., to match the predicted moments (mean, variance) to the true observed ones. In contrast to *quantile
    calibration* [3]_, where only the marginal calibration is of interest, the *distribution calibration* is more
    restrictive. It requires that the predicted moments should match the observed ones *given a certain probability
    distribution*. Therefore, the authors in [1]_ propose to use Gaussian process to estimate the recalibration
    parameters of a Beta calibration function locally (i.e., matching the observed moments of neighboring samples).

    In this framework, we use the base GP scheme to implement the Beta calibration for regression [1]_ as well as
    to derive a novel parametric recalibration that yields a parametric Gaussian or Cauchy distribution as
    calibration output [2]_.

    On the one hand, this method accepts as input X either a tuple X = (mean, stddev) using two NumPy arrays of
    shape N with N number of samples that express the estimated mean and standard deviation of a probabilistic
    forecaster. On the other hand, a NumPy array of shape (R, N) is also accepted where R denotes the number of
    probabilistic forecasts. For example, if probabilistic outputs are obtained by Monte-Carlo sampling using N samples
    and R stochastic forward passes, it is possible to pass all outputs to the calibration function in a single
    NumPy array.

    This method is capable of multiple independent data dimensions that are jointly optimized using a single Gaussian
    process. This method outputs a tuple consisting of three NumPy arrays:

    - 1st array: T points where the density/cumulative distribution functions are defined, shape: (T, N, D)
    - 2nd array: calibrated probability density function, shape: (T, N, D)
    - 3rd array: calibrated cumulative density function, shape: (T, N, D)

    **Mathematical background:** In [1]_, regression calibration is defined in terms of *distribution calibration*.
    A probabilistic forecaster :math:`h(X)` outputs for any input :math:`X \\in \\mathbb{R}` a probability density
    distribution :math:`f_Y(y) \\in \\mathcal{F}_Y` (where :math:`\\mathcal{F}` denotes the set of all possible
    probability distributions) for the target domain :math:`Y \\in \\mathcal{Y} = \\mathbb{R}`. Furthermore, let
    :math:`S = h(X)` denote the random variable of model predictions.
    Using this notation, *distribution calibration* [1]_ is defined as

    .. math::
        f_Y(Y=y | S = s) = s(y), \\quad \\forall s \\in \\mathcal{F}_Y, \\forall y \\in \\mathcal{Y} .

    In other words, "*this definition implies that if a calibrated model predicts a distribution with some mean*
    :math:`\\mu` *and variance* :math:`\\sigma^2` *, then it means that on average over all cases with the same
    prediction the mean of the target is* :math:`\\mu` *and variance is* :math:`\\sigma^2`" (cf. [1]_, p. 4 ll. 25-28).

    For uncertainty recalibration, a standard calibration function can be used, e. g., Variance Scaling.
    In contrast to the standard methods, we can use a Gaussian process to estimate the rescaling parameter of the
    scaling method. This offers more flexibility and a more "local" or focused recalibration of a single sample.

    Since not all calibration methods yield an analytically tractable likelihood (e.g., the GP-Beta uses a non-standard
    likelihood), the current GP scheme is a sampling based variational one.
    Furthermore, as the amount of training data might grow very large, we use an approximate GP with so called
    inducing points that are learnt from the data and used during inference to obtain the calibration parameters.
    Therefore, we the computational complexity keeps fixed during inference.
    For a detailed derivation of the GP scheme, cf. [1]_.
    For a detailed description of the kernel function, cf. :class:`netcal.regression.gp.kernel.GaussianRBFKernel`.

    Parameters
    ----------
    n_inducing_points: int
        Number of inducing points used to approximate the input space. These inducing points are also optimized.
    n_random_samples: int
        Number of random samples used to sample from the parameter distribution during optimization and inference.
    n_parameters: int
        Number of parameters that are required for a dedicated implementation of the GP method (mainly used internally).
    correlations: bool, default: False
        If True, perform covariance estimation or covariance recalibration on multivariate input data.
        Only works for GPNormal.
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
    name_prefix: str, optional, default: "abstractgp"
        Name prefix internally used in Pyro to distinguish between parameter stores.

    References
    ----------
    .. [1] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
           "Distribution calibration for regression."
           International Conference on Machine Learning (pp. 5897-5906), 2019.
           `Get source online <http://proceedings.mlr.press/v97/song19a/song19a.pdf>`__
    .. [2] Song, L., Zhang, X., Smola, A., Gretton, A., & Schölkopf, B.:
           "Tailoring density estimation via reproducing kernel moment matching."
           In Proceedings of the 25th international conference on Machine learning (pp. 992-999), July 2008.
           `Get source online <https://www.cs.uic.edu/~zhangx/pubDoc/xinhua_icml08.pdf>`__
    """

    precision = torch.float64
    precision_np = np.float64

    def __init__(
            self,
            n_inducing_points: int,
            n_random_samples: int,
            *,
            n_parameters: int,
            correlations: bool = False,
            n_epochs: int = 200,
            batch_size: int = 256,
            num_workers : int = 0,
            lr: float = 1e-2,
            use_cuda: Union[str, bool] = False,
            jitter: float = 1e-5,
            name_prefix="abstractgp"
    ):

        if isinstance(use_cuda, str):
            # this line will throw an exception if the cuda device does not exist
            device = torch.device(use_cuda)
            torch.cuda.get_device_name(use_cuda)

        else:
            device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

        # optimization hyperparameters
        self.n_inducing_points = n_inducing_points
        self.n_random_samples = n_random_samples
        self.n_parameters = n_parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.device = device
        self.correlations = correlations
        self.jitter = jitter
        self.name_prefix = name_prefix

        self.marginal_correlations = None
        self.variational_strategy = None
        self.mean_module = None
        self.covar_module = None
        self.inv_gamma = None
        self.delta = None
        self.pyro_scope = {'params': {}, 'constraints': {}}

        # Standard initializtation
        AbstractCalibration.__init__(self, detection=False, independent_probabilities=False)
        gpytorch.models.ApproximateGP.__init__(self, None)

    def clear(self):
        """ Clear module parameters. """

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        self.marginal_correlations = None
        self.variational_strategy = None
        self.mean_module = None
        self.covar_module = None
        self.inv_gamma = None
        self.delta = None
        self.pyro_scope = {'params': {}, 'constraints': {}}

    def get_params(self, deep=True) -> Dict:
        """
        Overwrite base method's get_params function to also capture child parameters as variational strategy, LMC
        coefficients, etc.
        """

        parameters = {
            "get_params": super().get_params(deep=deep),
            "n_dims": self.ndim,
            "state_dict": self.state_dict(),
        }

        return parameters

    def load_model(self, filename, use_cuda: Union[str, bool, None] = None) -> 'AbstractGP':
        """
        Overwrite base method's load_model function as the parameters for the GP methods are stored differently
        compared to the remaining methods.

        Parameters
        ----------
        filename : str
            String with filename.
        use_cuda : str or bool, optional, default: None
            Specify if CUDA should be used. If str, you can also specify the device
            number like 'cuda:0', etc. If unset, use the device id that has been stored on disk.

        Returns
        -------
        AbstractGP
            Instance of a child class of `AbstractGP`.
        """

        with open(filename, 'rb') as read_object:
            loaded_params = torch.load(read_object, map_location=torch.device('cpu'))

        # in a first step, recover all base parameters
        params = loaded_params["get_params"]
        n_dims = loaded_params["n_dims"]
        self.set_params(**params)

        # ---------------------------------------
        # overwrite default computing device
        # if not set, leave self.device untouched
        if use_cuda is None:
            pass

        # otherwise, overwrite default device
        else:
            if isinstance(use_cuda, str):
                # this line will throw an exception if the cuda device does not exist
                device = torch.device(use_cuda)
                torch.cuda.get_device_name(use_cuda)

            else:
                device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

            self.device = device

        # ---------------------------------------

        # initialize (empty) mean/covar module, LMC variational strategy and inducing points
        mean, var = torch.zeros(1, n_dims, dtype=self.precision), torch.ones(1, n_dims, dtype=self.precision)
        if self.correlations:
            var = torch.diag_embed(var)
            tril_idx = torch.tril_indices(row=n_dims, col=n_dims)
            var = var[:, tril_idx[0], tril_idx[1]]

        # using the "empty" initialization, we can recover the learnt parameter by the passed state dict
        self._init_gp_model(mean, var)
        self.load_state_dict(loaded_params["state_dict"])

        return self

    @property
    def inducing_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return inducing points mean and variance/covariance as tuple. """

        if self.variational_strategy is None:
            raise RuntimeError("AbstractGP: could not return inducing points: call fit() first.")

        # get raw inducing points
        inducing_points = self.variational_strategy.base_variational_strategy.inducing_points.detach()  # (m, d) or (m, d + (d^2+d)//2))

        # if covariance estimation, recover cov from decomposed lower triangular
        if self.correlations:
            n_dims = int((np.sqrt(8 * inducing_points.shape[-1] + 9) - 3) // 2)

            # get lower triangular indices and recover cov
            tril_idx = torch.tril_indices(row=n_dims, col=n_dims)
            var = torch.zeros(self.n_inducing_points, n_dims, n_dims, dtype=inducing_points.dtype, device=inducing_points.device)
            var[:, tril_idx[0], tril_idx[1]] = inducing_points[:, n_dims:]
            var = var @ var.transpose(dim0=2, dim1=1)

        # simply return the (diagonal) variances
        else:
            n_dims = int(inducing_points.shape[-1] // 2)
            var = inducing_points[:, n_dims:]

        # extract the mean and return everything as NumPy arrays
        mean = inducing_points[:, :n_dims]
        return mean.numpy(), var.numpy()

    @property
    def ndim(self):
        """ Get number of dimensions for which this method was trained for """

        if self.variational_strategy is None:
            raise RuntimeError("AbstractGP: could not return ndim: call fit() first.")

        # get raw inducing points
        inducing_points = self.variational_strategy.base_variational_strategy.inducing_points  # (m, d) or (m, d + (d^2+d)//2))
        if self.correlations:
            n_dims = int((np.sqrt(8 * inducing_points.shape[-1] + 9) - 3) // 2)
        else:
            n_dims = int(inducing_points.shape[-1] // 2)

        return n_dims

    # -------------------------------------------------------------------------
    # Pyro + GPyTorch functions

    def forward(self, x) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward method defines the prior for the GP.

        Parameters
        ----------
        x : torch.Tensor of shape (n, 2*d)
            Set of samples used for kernel computation consisting of mean and variance for each input point.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Multivariate normal distribution holding the GP prior information.
        """

        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        """
        Pyro guide that defines the variational distribution for the Gaussian process.

        Parameters
        ----------
        x : torch.Tensor of shape (n, 2*d)
            Set of samples used for kernel computation consisting of mean and variance for each input point.
        y : torch.Tensor of shape (n, d)
            Ground-truth regression information (not used within the guide).
        """

        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x: torch.Tensor, y: torch.Tensor):
        """
        Model function that defines the computation graph. Get the variational
        distribution for the Gaussian process and sample function parameters for recalibration.

        Parameters
        ----------
        x : torch.Tensor of shape (n, 2*d)
            Set of samples used for kernel computation consisting of mean and variance for each input point.
        y : torch.Tensor of shape (n, d)
            Ground-truth regression information used for the ELBO loss.

        Returns
        -------
        pyro.sample statement
            Sample statement using the specified likelihood.
        """

        # decompose input arguments
        assert x.ndim == 2, "GP optimization: x must be 2-D with shape (n, 2*d) or (n, d + (d^2+d)//2)."

        # if correlations, x is the mean and decomposed cov
        if self.correlations:
            n_dims = int((np.sqrt(8 * x.shape[-1] + 9) - 3) // 2)
        else:
            n_dims = x.shape[1] // 2

        x_loc = x[:, :n_dims]  # (n, d)
        x_var = x[:, n_dims:]  # (n, d) or (n, (d^2 +d) // 2)

        # register module in pyro
        pyro.module(self.name_prefix + ".gp", self)

        # get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)  # ([r], n, p*d)
            function_samples = function_samples * self.inv_gamma + self.delta  # ([r], n, p*d)

            # if we're in the random sampling mode, prepend a distribution dim to data
            if function_samples.ndim == 3:
                x_loc = x_loc.unsqueeze(0)  # (1, n, d)
                x_var = x_var.unsqueeze(0)  # (1, n, d)
                y = y.unsqueeze(0)  # (1, n, d)

            # sample from observed distribution and use custom likelihood
            # to_event(1) marks the last dim d as event_dim so that pyro infers a single likelihood over all d
            # e.g., for independent normals, this is simply the sum of the log_probs
            likelihood = self.likelihood(x_loc, x_var, parameters=function_samples)

            # if no correlations are modelled across the different dimensions, use independent distributions
            # by calling "to_event" method
            if not self.correlations:
                likelihood = likelihood.to_event(1)

            return pyro.sample(
                self.name_prefix + ".y",
                likelihood,
                obs=y
            )

    # -------------------------------------------------------------------------

    def _init_gp_model(self, Xmean: torch.Tensor, Xlogvariance: torch.Tensor):
        """
        Initialize the GP model. This method prepares the mean and variance to have the right shape.
        If "correlations=True", the input variance/covariance gets LDL* decomposed.
        Furthermore, the inducing points are initialized and the variational distribution is set up.

        Parameters
        ----------
        Xmean : torch.Tensor of shape (n, d)
            The mean of the training data (see _get_input output description).
        Xlogvariance : torch.Tensor of shape (n, d, [d])
            The log-variance of covariance of the training data (see _get_input output description).
        """

        # convert log of variance back to variance
        Xvariance = torch.exp(Xlogvariance)

        # get number of dimensions
        n_samples, n_dims = Xmean.shape

        # -------------------------------------------------
        # initialize inducing points based on training data
        # initialize range for inducing points
        loc_min, loc_max = torch.min(Xmean, dim=0).values, torch.max(Xmean, dim=0).values

        # initialize inducing points with ones
        inducing_points_var = torch.ones(self.n_inducing_points, n_dims, dtype=self.precision) * 3 # (n, d)

        # if input is given as cov, we also need inducing points with covariance
        if self.correlations:

            tril_idx = torch.tril_indices(row=n_dims, col=n_dims)
            stddev_diagonal = Xvariance[:, tril_idx[0] == tril_idx[1]]  # (n, d)
            std_max = torch.max(stddev_diagonal, dim=0).values  # (d,)

            # initialize off-diagonal and concatenate to "flat" array
            inducing_points_var = torch.cat(
                (inducing_points_var,
                 torch.ones((self.n_inducing_points, (n_dims ** 2 - n_dims) // 2),
                            dtype=inducing_points_var.dtype,
                            device=inducing_points_var.device
                            ).normal_(std=1e-3)
                 ),
                dim=1
            )  # (m, (d^d + d)//2)

        else:
            # get maximum standard deviation to init space of inducing points
            std_max = torch.sqrt(torch.max(Xvariance, dim=0).values)  # (d,)

            # use log of inducing points variance to guarantee positive variances during optimization
            inducing_points_var = torch.log(inducing_points_var)

        # use NumPy linspace as it supports vectorized start and end points (in contrast to PyTorch)
        inducing_points_mean = torch.from_numpy(
            np.linspace(
                (loc_min - 3 * std_max).numpy(),
                (loc_max + 3 * std_max).numpy(),
                self.n_inducing_points
            )
        ).to(self.precision)  # (m, d)

        # concatenate mean and var of inducing points to a single pytorch tensor
        inducing_points = torch.cat((inducing_points_mean, inducing_points_var), dim=1)

        # variational distribution for inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.n_inducing_points, batch_shape=torch.Size([1])
        )

        # convert dtype of variational distribution
        variational_distribution.variational_mean.data = variational_distribution.variational_mean.data.to(
            dtype=self.precision)
        variational_distribution.chol_variational_covar.data = variational_distribution.chol_variational_covar.data.to(
            dtype=self.precision)

        # scale the number of estimated parameters depending on the correlations parameter
        # if self.correlations is True, assert variance rescaling using multivariate normal
        # in this case, the number of output parameters is (n^2 + n)/2
        if self.correlations:
            n_outputs = (n_dims + n_dims ** 2) // 2
        else:
            n_outputs = n_dims * self.n_parameters

        # initialize variational multitask strategy
        self.variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True,
            ), num_tasks=n_outputs, num_latents=1
        )

        # num_latents is 1 so we have a batch_shape of 1 to approximate an intrinsic model of coregionalization (ICM)
        # that is used in the original paper. See: https://github.com/cornellius-gp/gpytorch/issues/1035
        # the linear dependencies are modelled by the 'LMCVariationalStrategy'
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GaussianRBFKernel(cov=self.correlations),
            batch_shape=torch.Size([1])
        )

        if self._learn_scale_shift:
            # scaling parameters for sampled weights
            self.inv_gamma = nn.Parameter(data=torch.ones(size=(1, n_outputs)), requires_grad=True)
            self.delta = nn.Parameter(data=torch.zeros(size=(1, n_outputs)), requires_grad=True)

            self.register_parameter('inv_gamma', self.inv_gamma)
            self.register_parameter('delta', self.delta)

        else:
            self.inv_gamma = 1.
            self.delta = 0.

    def _get_input(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        Perform some input checks on the data and convert to PyTorch.

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
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.

        Returns
        -------
        tuple of size 2
            If y is given, the first entry in the tuple is a tuple consisting of the
            transformed input mean and variance. The second entry is the transformed y.
            If y is None, simply return the transformed input mean and variance.
        """

        # preprocess inputs to extract mean, variance and according target scores
        if y is not None:
            (Xmean, Xvariance), y, input_cov = meanvar(X, y)
        else:
            Xmean, Xvariance, input_cov = meanvar(X)

        # perform consistency check
        if input_cov and not self.correlations:
            raise RuntimeError("Found input covariance matrices but \'correlations\' is set to False.")

        # if Xmean or Xvariance is 1-D, append a second axis
        Xmean = np.expand_dims(Xmean, axis=1) if Xmean.ndim == 1 else Xmean  # (n, d)
        Xvariance = np.expand_dims(Xvariance, axis=1) if Xvariance.ndim == 1 else Xvariance  # (n, d)

        # convert mean, variance and target scores to PyTorch tensors
        Xmean = torch.from_numpy(Xmean).to(dtype=self.precision)
        Xvariance = torch.from_numpy(Xvariance).to(dtype=self.precision)

        # if capturing correlations is enabled, prepare the given variance/covariance matrices
        if self.correlations:

            # if input is given as variance vector, convert it to diagonal covariance matrix first
            if not input_cov:
                Xvariance = torch.diag_embed(Xvariance)  # (n, d, d)

                # in the next step, capture the marginal correlation coefficients (correlation between d dimensions
                # across the whole dataset)
                # use NumPy function as PyTorch does not serve this functionality
                if self.marginal_correlations is None:
                    assert y is not None, "Fatal: building correlation coefficients but y is None. " \
                                          "You maybe trained a GP model using covariance matrices but try to " \
                                          "transform using variances only."

                    self.marginal_correlations = torch.from_numpy(
                        np.corrcoef(y, rowvar=False)
                    ).unsqueeze(dim=0)  # (1, d, d)

                # use the previously defined correlation coefficients to compute the covariances for each
                # sample individually
                Xvariance = torch.sqrt(Xvariance) @ self.marginal_correlations @ torch.sqrt(Xvariance)  # (n, d, d)

            # use cholesky decomposed covariance matrices during optimization
            # return "flattened" lower triangular
            n_dims = Xmean.shape[-1]
            tril_idx = torch.tril_indices(row=n_dims, col=n_dims)

            # catch possible errors in covariance decomposition and turn it into
            # a more useful error message
            try:
                Xvariance = torch.linalg.cholesky(Xvariance)  # (n, d, d)
            except RuntimeError:
                raise RuntimeError("Input covariance matrices are not positive semidefinite.")

            # only use the lower decomposed matrix during optimization
            Xvariance = Xvariance[:, tril_idx[0], tril_idx[1]]  # (n, (d^2+d)//2)

        # in the univariate case, use the log variance to guarantee positive variance estimates for the inducing points
        else:
            Xvariance = torch.log(Xvariance)

        if y is not None:
            y = np.expand_dims(y, axis=1) if y.ndim == 1 else y
            y = torch.from_numpy(y).to(dtype=self.precision)

            return (Xmean, Xvariance), y
        else:
            return Xmean, Xvariance

    def fit(self, X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray], y: np.ndarray,
            tensorboard: Optional[SummaryWriter] = None) -> 'AbstractGP':
        """
        Fit a GP model to the provided data using Gaussian process optimization.

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
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        tensorboard: torch.utils.tensorboard.SummaryWriter, optional, default: None
            Instance of type "SummaryWriter" to log training statistics.

        Returns
        -------
        AbstractGP
            (Child) instance of class :class:`AbstractGP`.
        """

        self.clear()
        with torch.no_grad():
            (Xmean, Xvariance), y = self._get_input(X, y)

        with pyro.get_param_store().scope() as scope:

            # initialize covariance module, variational distribution and inducing points based on the input data
            self._init_gp_model(Xmean, Xvariance)

            # concatenate mean and var of training points to a single pytorch tensor
            X = torch.cat((Xmean, Xvariance), dim=1)  # (n, 2*d) or (n, d + (d^2+d)//2)

            # -------------------------------------------------
            # initialize training

            # create PyTorch dataset and dataloader for optimization
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers, pin_memory=True)

            optimizer = pyro.optim.Adam({"lr": self.lr})
            elbo = pyro.infer.Trace_ELBO(num_particles=self.n_random_samples, vectorize_particles=True, retain_graph=True)
            svi = pyro.infer.SVI(self.model, self.guide, optimizer, elbo)

            # log hyperparameters to tensorboard
            if tensorboard is not None:
                prefix = "%s/train" % self.name_prefix
                tensorboard.add_scalar("%s/lr" % prefix, self.lr)
                tensorboard.add_scalar("%s/batch_size" % prefix, self.batch_size)
                tensorboard.add_scalar("%s/n_epochs" % prefix, self.n_epochs)
                tensorboard.add_scalar("%s/n_inducing_points" % prefix, self.n_inducing_points)
                tensorboard.add_scalar("%s/n_random_samples" % prefix, self.n_random_samples)
                tensorboard.add_scalar("%s/jitter" % prefix, self.jitter)
                tensorboard.add_scalar("%s/correlations" % prefix, int(self.correlations))

            self.train()
            self.to(self.device, dtype=self.precision)

            best_loss, best_parameters = float('inf'), {}

            # enter optimization loop and iterate over epochs and batches
            with gpytorch.settings.cholesky_jitter(float=self.jitter, double=self.jitter), tqdm(total=self.n_epochs) as pbar:
                step = 0
                for epoch in range(self.n_epochs):
                    train_loss = []
                    for batch_X, batch_y in dataloader:

                        # set proper device for optimization
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        self.zero_grad()
                        batch_loss = svi.step(batch_X, batch_y) / len(batch_X)
                        train_loss.append(batch_loss)

                        # log batch-wise loss
                        if tensorboard is not None:
                            tensorboard.add_scalar("%s/train/loss/batch" % self.name_prefix, batch_loss, step)

                        step += 1

                    # store best weights after each epoch
                    epoch_loss = np.mean(train_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_parameters = deepcopy(self.state_dict())

                    # log epoch loss - assign "step" as global step to sync to batch loss
                    if tensorboard is not None:
                        tensorboard.add_scalar("%s/train/loss/epoch" % self.name_prefix, epoch_loss, step)

                    pbar.set_description("Loss: %.4f" % epoch_loss)
                    pbar.update(1)

            # restore best weights with lowest loss and set into eval mode
            self.load_state_dict(best_parameters)
            self.cpu()
            self.eval()

            # draw inducing points space
            if tensorboard is not None:

                # log lengthscale parameter of kernel module
                tensorboard.add_scalar("%s/train/lengthscale" % self.name_prefix, self.covar_module.outputscale)

                # get learned inducing points
                inducing_mean, inducing_var = self.inducing_points  # (m, d)
                n_dims = inducing_mean.shape[-1]

                # only use diagonal if capturing correlations is enabled
                if self.correlations:
                    inducing_var = np.diagonal(inducing_var, axis1=1, axis2=2)
                else:
                    inducing_var = np.exp(inducing_var)

                x = np.linspace(
                    np.min(inducing_mean, axis=0) - 3 * np.sqrt(np.max(inducing_var, axis=0)),
                    np.max(inducing_mean, axis=0) + 3 * np.sqrt(np.max(inducing_var, axis=0)),
                    1000,
                )  # (n, d)

                # get Gaussian mixture model
                inducing_density = norm.pdf(x[:, None, :], loc=inducing_mean[None, ...], scale=np.sqrt(inducing_var)[None, ...])  # (n, m, d)
                inducing_density = np.sum(inducing_density / n_dims, axis=1)  # (n, d)

                # iterate over dimensions and log stats
                for dim in range(n_dims):

                    # visualize as Gaussian mixture model
                    fig, ax = plt.subplots()
                    ax.plot(x[:, dim], inducing_density[:, dim], "-")
                    ax.grid(True)
                    ax.set_title("GP inducing points dim %02d" % dim)

                    # add matplotlib figure to SummaryWriter
                    tensorboard.add_figure("%s/train/inducing_points/dim%02d" % (self.name_prefix, dim), fig, close=True)

                    # also log gamma and delta parameter (mostly used for gpbeta)
                    if self._learn_scale_shift:
                        tensorboard.add_scalar("%s/train/gamma/dim%02d" % (self.name_prefix, dim), 1./self.inv_gamma[0, dim])
                        tensorboard.add_scalar("%s/train/delta/dim%02d" % (self.name_prefix, dim), self.delta[0, dim])

        # store pyro's scope
        self.pyro_scope = scope
        return self

    def transform(self, X: Union[Iterable[np.ndarray], np.ndarray]) -> np.ndarray:
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
        """

        raise NotImplementedError("AbstractGP.transform must be implemented by child class.")

    def extra_repr(self) -> str:
        """ Additional information used to print if str(method) is called. """

        content = "n_inducing_points=%d\n" \
                  "n_random_samples=%d\n" \
                  "n_epochs=%d\n" \
                  "batch_size=%d\n" \
                  "lr=%.2E\n" \
                  "correlations=%r\n" \
                  "jitter=%.2E" % \
                  (
                      self.n_inducing_points, self.n_random_samples, self.n_epochs,
                      self.batch_size, self.lr, self.correlations, self.jitter
                  )
        return content

    def __repr__(self) -> str:
        """ Returns a string representation of the calibration method with the most important parameters. """

        return str(nn.Module.__repr__(self))
