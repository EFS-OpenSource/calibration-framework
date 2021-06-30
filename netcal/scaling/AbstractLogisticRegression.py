# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple, List
import warnings
from collections import OrderedDict, defaultdict
from typing import Union
import abc
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit as safe_logit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributions as tdist
import torch.distributions.constraints as constraints
from torch.utils.tensorboard import SummaryWriter

import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.distributions as dist

from netcal import AbstractCalibration, dimensions, accepts, manual_seed


class AbstractLogisticRegression(AbstractCalibration):
    """
    Abstract class for all calibration methods that base on logistic regression. We extended common
    scaling calibration methods by Bayesian epistemic uncertainty modelling [1]_.
    On the one hand, this class supports Maximum Likelihood (MLE) estimates without uncertainty.
    This method is commonly solved by negative log likelihood optimization given by

    .. math::
       \\theta_\\text{MLE} = \\underset{\\theta}{\\text{min}} \\, -\\sum_{i=1}^N \\log p(y | x_i, \\theta)

    with samples :math:`X`, label :math:`y`, weights :math:`\\theta` and likelihood :math:`p(y|X, \\theta)`.
    See the implementations of the methods for more details.

    On the other hand, methods to obtain uncertainty in calibration are currently Variational Inference (VI) and
    Markov-Chain Monte-Carlo (MCMC) sampling. Instead of estimating the weights :math:`\\theta` of the logistic
    regression directly, we place a probability distribution over the weights by

    .. math::
       p(\\theta | X, y) = \\frac{p(y | X, \\theta) p(\\theta)}{\\int p(y | X, \\theta) p(\\theta) d\\theta}

    Since the marginal likelihood cannot be evaluated analytically for logistic regression, we need to approximate the
    posterior by either MCMC sampling or Variational Inference. Using several techniques, we sample multiple times from
    the posterior in order to get multiple related calibration results with a mean and a deviation for each sample.

    MCMC sampling allows the sampling of a posterior without knowing the marginal likelihood. This method is unbiased
    but computational expensive. In contrast, Variational Inference defines an easy variational
    distribution :math:`q_\\Phi(\\theta)` (e.g. a normal distribution) for each weight parametrized by :math:`\\Phi`.
    The optimization objective is then the minimization of the Kullback-Leibler divergence between the
    variational distribution :math:`q_\\Phi(\\theta))` and the true posterior :math:`p(\\theta | X, y)`.
    This can be solved using the ELBO method [2]_. Variational Inference is faster than MCMC but also biased.

    Parameters
    ----------
    method : str, default: "mle"
        Method that is used to obtain a calibration mapping:
        - 'mle': Maximum likelihood estimate without uncertainty using a convex optimizer.
        - 'momentum': MLE estimate using Momentum optimizer for non-convex optimization.
        - 'variational': Variational Inference with uncertainty.
        - 'mcmc': Markov-Chain Monte-Carlo sampling with uncertainty.
    momentum_epochs : int, optional, default: 1000
            Number of epochs used by momentum optimizer.
    mcmc_steps : int, optional, default: 20
        Number of weight samples obtained by MCMC sampling.
    mcmc_chains : int, optional, default: 1
        Number of Markov-chains used in parallel for MCMC sampling (this will result
        in mcmc_steps * mcmc_chains samples).
    mcmc_warmup_steps : int, optional, default: 100
        Warmup steps used for MCMC sampling.
    vi_epochs : int, optional, default: 1000
        Number of epochs used for ELBO optimization.
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).
    use_cuda : str or bool, optional, default: False
        Specify if CUDA should be used. If str, you can also specify the device
        number like 'cuda:0', etc.

    References
    ----------
    .. [1] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021

    .. [2] Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul:
       "An introduction to variational methods for graphical models." Machine learning, 37(2): 183–233, 1999.
    """

    @accepts(str, int, int, int, int, int, bool, bool, (str, bool))
    def __init__(self,
                 method: str = 'mle',
                 momentum_epochs: int = 1000,

                 mcmc_steps: int = 250,
                 mcmc_chains: int = 1,
                 mcmc_warmup_steps: int = 100,

                 vi_epochs: int = 1000,

                 detection: bool = False,
                 independent_probabilities: bool = False,
                 use_cuda: Union[str, bool] = False,
                 **kwargs):
        """ Create an instance of `AbstractLogisticRegression`. Detailed parameter description given in class docs. """

        super().__init__(detection=detection, independent_probabilities=independent_probabilities)

        if 'num_samples' in kwargs:
            warnings.warn("Parameter \'num_samples\' in constructor is deprecated and will be removed. "
                          "Use this parameter in \'transform\' function call instead.")

        if method == "mcmc":
            warnings.warn("Optimization type \'MCMC\' is implemented but needs revision. Use \'variational\' instead.")

        self.method = method.lower()
        self.num_features = None

        # epochs for momentum optimization
        self.momentum_epochs = momentum_epochs

        # properties for MCMC
        self.mcmc_model = None
        self.mcmc_steps = mcmc_steps
        self.mcmc_chains = mcmc_chains
        self.mcmc_warmup = mcmc_warmup_steps

        # properties for Variational Inference
        self.vi_model = None
        self.vi_epochs = vi_epochs

        if isinstance(use_cuda, str):
            # this line will throw an exception if the cuda device does not exist
            self._device = torch.device(use_cuda)
            torch.cuda.get_device_name(use_cuda)

        else:
            self._device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

        # mask negative: for some methods like beta calibration, repeat optimization on MLE if
        # negative values occur on the first run
        self.mask_negative = False
        self._sites = None

    def save_model(self, filename: str):
        """
        Save model instance as with torch's save function as this is safer for torch tensors.

        Parameters
        ----------
        filename : str
            String with filename.
        """

        # overwrite is necessary because we want to copy everything back on CPU before we store anything
        self.to(torch.device('cpu'))
        super().save_model(filename)

    def clear(self):
        """
        Clear model parameters.
        """

        # call parental clear method and clear parameter store of pyro
        super().clear()
        pyro.clear_param_store()

        self.num_features = None
        self._sites = None

        self.mcmc_model = None
        self.vi_model = None

    @abc.abstractmethod
    def prepare(self, X: np.ndarray) -> torch.Tensor:
        """
        Preprocessing of input data before called at the beginning of the fit-function.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        torch.Tensor
            Prepared data vector X as torch tensor.
        """

        return torch.Tensor(X).to(self._device)

    @abc.abstractmethod
    def prior(self):
        """
        Prior definition of the weights and intercept used for log regression. This function has to set the
        sites at least for "weights" and "bias".
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def model(self, X: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        """
        Definition of the log regression model.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_log_regression_features)
            Input data that has been prepared by "self.prepare" function call.
        y : torch.Tensor, shape=(n_samples, [n_classes])
            Torch tensor with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D) (for multiclass MLE only).

        Returns
        -------
        torch.Tensor, shape=(n_samples, [n_classes])
            Logit of the log regression model.
        """

        raise NotImplementedError()

    def mask(self) -> Tuple[np.ndarray, List]:
        """
        Seek for all relevant weights whose values are negative. Mask those values with optimization constraints
        in the interval [0, 0].
        Constraints on the intercepts might also be set.

        Returns
        -------
        tuple of (np.ndarray, list)
            Indices of masked values and list of boundary constraints for optimization.
        """

        raise NotImplementedError()

    def guide(self, X: torch.Tensor = None, y: torch.Tensor = None):
        """
        Variational substitution definition for each parameter. The signature is the same as for the
        "self.model" function but the variables are not used.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_log_regression_features)
            Input data that has been prepared by "self.prepare" function call.
        y : torch.Tensor, shape=(n_samples, [n_classes])
            Torch tensor with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D) (for multiclass MLE only).
        """

        # iterate over all sites
        for name, site in self._sites.items():

            # get mean and scale as pyro parameters with (default) constraints
            mean = pyro.param("%s_mean" % name, site['init']['mean'], constraint=site['constraint'])
            scale = pyro.param("%s_scale" % name, site['init']['scale'], constraint=constraints.positive)

            # use LogNormal if values are restricted to be positive
            # use Normal distribution otherwise
            guide_dist = dist.LogNormal if isinstance(site['constraint'], (constraints._GreaterThan, constraints._GreaterThanEq)) else dist.Normal

            pyro.sample(
                name, guide_dist(mean, scale, validate_args=True).independent(1)
            )

    def to(self, device: torch.device):
        """ Set distribution parameters to the desired device in order to compute either on CPU or GPU. """

        def get_base(distribution: dist.Distribution):
            """ Get base distribution recursively (only works for derived Gaussians at the moment) """

            if isinstance(distribution, (dist.Independent, dist.LogNormal)):
                return get_base(distribution.base_dist)
            elif isinstance(distribution, (dist.Normal, tdist.Normal)):
                return distribution
            else:
                raise ValueError("Method is currently not implemented for other distributions than 'Independent', 'LogNormal' or 'Normal'.")

        assert isinstance(self._sites, OrderedDict), "Method \'prior\' has to set all necessary initialization values and priors."

        for name, site in self._sites.items():

            # assert some member variables set by the 'prior' function
            assert isinstance(site['prior'], dist.Distribution), "Method \'prior\' has to set prior dist for site %s." % name
            assert isinstance(site['init']['mean'], torch.Tensor), "Method \'prior\' has to set initial mean for site %s." % name
            assert isinstance(site['init']['scale'], torch.Tensor), "Method \'prior\' has to set initial scale for site %s." % name

            # on some derived distributions (e.g. LogNormal), we need to set the base distribution parameters
            # instead of the distribution parameters itself
            prior_base = get_base(site['prior'])
            prior_base.loc = prior_base.loc.to(device)
            prior_base.scale = prior_base.scale.to(device)

            # set initial values for mean and scale also to the proper device
            site['init']['mean'] = site['init']['mean'].to(device)
            site['init']['scale'] = site['init']['scale'].to(device)

        # variational model is ParamStoreDict from pyro
        if self.vi_model is not None:
            for key, param in self.vi_model['params'].items():
                self.vi_model['params'][key] = param.detach().to(device)

        # MCMC samples are also dictionary
        if self.mcmc_model is not None:
            for key, param in self.vi_model.items():
                self.vi_model[key] = param.detach().to(device)

    @dimensions((1, 2), (1, 2), None, None, None)
    def fit(self, X: np.ndarray, y: np.ndarray, random_state: int = None, tensorboard: bool = True,
                 log_dir: str = None) -> 'AbstractLogisticRegression':
        """
        Build logitic calibration model either conventional with single MLE estimate or with
        Variational Inference (VI) or Markov-Chain Monte-Carlo (MCMC) algorithm to also obtain uncertainty estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
        random_state : int, optional, default: None
            Fix the random seed for the random number

        Returns
        -------
        AbstractLogisticRegression
            Instance of class :class:`AbstractLogisticRegression`.
        """

        X, y = super().fit(X, y)

        # prepare data input for algorithm
        data = self.prepare(X).to(self._device)

        # if y is given as one-hot, convert back to categorical encoding
        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        y = torch.from_numpy(y).to(self._device)
        self.num_features = X.shape[1] if self.detection else 1

        # initialize priors
        self.prior()

        # mark first dimension as independent
        for site in self._sites.values():
            site['prior'] = site['prior'].independent(1)

        self.to(self._device)

        with manual_seed(seed=random_state):

            # markov-chain monte-carlo sampling (with uncertainty estimates)
            if self.method == 'mcmc':
                self.mcmc(data, y, tensorboard, log_dir)

            # variational inference (with uncertainty estimates)
            elif self.method == 'variational':
                self.variational(data, y, tensorboard, log_dir)

            # Maximum likelihood estimate (without uncertainty)
            elif self.method == 'mle':
                self.convex(data, y, tensorboard, log_dir)

            # momentum is for non-convex optimization
            elif self.method == 'momentum':
                self.momentum(data, y, tensorboard, log_dir)
            else:
                raise AttributeError("Unknown method \'%s\'." % self.method)

        # delete torch tensors
        del data
        del y

        # if device is cuda, empty GPU cache to free memory
        if self._device.type == 'cuda':
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()

        return self

    # -----------------------------------------------------------------

    def mcmc(self, data: torch.Tensor, y: torch.Tensor, tensorboard: bool, log_dir: str):
        """
        Perform Markov-Chain Monte-Carlo sampling on the (unknown) posterior.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels as 1-D vector (binary).
        """

        if tensorboard:
            writer = SummaryWriter(log_dir=log_dir)
            distribution = defaultdict(list)

            def log(kernel, samples, stage, i):
                """ Log after each MCMC iteration """

                # loop through all sites and log their value as well as the underlying distribution
                # approximated by a Gaussian
                for key, value in samples.items():
                    distribution[key].append(value)
                    stacked = torch.stack(distribution[key], dim=0)
                    mean, scale = torch.mean(stacked, dim=0), torch.std(stacked, dim=0)

                    for d, x in enumerate(value):
                        writer.add_scalar("%s_%s_%d" % (stage, key, d), x, i)
                        writer.add_scalar("%s_%s_mean_%d" % (stage, key, d), mean[d], i)
                        writer.add_scalar("%s_%s_scale_%d" % (stage, key, d), scale[d], i)

                        writer.add_histogram("%s_histogram_%s_%d" % (stage, key, d), stacked[:, d], i)

        # if logging is not requested, return empty lambda
        else:
            log = lambda kernel, samples, stage, i: None

        # set up MCMC kernel
        kernel = NUTS(self.model)

        # initialize MCMC sampler and run sampling algorithm
        mcmc = MCMC(kernel, num_samples=self.mcmc_steps,
                    warmup_steps=self.mcmc_warmup,
                    num_chains=self.mcmc_chains,
                    hook_fn=log)
        mcmc.run(data.float(), y.float())

        # get samples from MCMC chains and store weights
        samples = mcmc.get_samples()
        self.mcmc_model = samples

        if tensorboard:
            writer.close()

    def variational(self, data: torch.Tensor, y: torch.Tensor, tensorboard: bool, log_dir: str):
        """
        Perform variational inference using the guide.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels as 1-D vector (binary).
        """

        # explicitly define datatype
        data = data.float()
        y = y.float()

        num_samples = data.shape[0]

        # create dataset
        lr_dataset = torch.utils.data.TensorDataset(data, y)
        data_loader = DataLoader(dataset=lr_dataset, batch_size=1024, pin_memory=False)

        # define optimizer
        optim = Adam({'lr': 0.01})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        # add tensorboard writer if requested
        if tensorboard:
            writer = SummaryWriter(log_dir=log_dir)

        # start variational process
        with tqdm(total=self.vi_epochs) as pbar:
            for epoch in range(self.vi_epochs):
                epoch_loss = 0.
                for i, (x, y) in enumerate(data_loader):
                    epoch_loss += svi.step(x, y)

                # get loss of complete epoch
                epoch_loss = epoch_loss / num_samples

                # logging stuff
                if tensorboard:

                    # add loss to logging
                    writer.add_scalar("SVI loss", epoch_loss, epoch)

                    # get param store and log current state of parameter store
                    param_store = pyro.get_param_store()
                    for key in self._sites.keys():
                        for d, (loc, scale) in enumerate(zip(param_store["%s_mean" % key], param_store["%s_scale" % key])):
                            writer.add_scalar("%s_mean_%d" % (key, d), loc, epoch)
                            writer.add_scalar("%s_scale_%d" % (key, d), scale, epoch)

                            # also represent the weights as distributions
                            density = np.random.normal(loc=loc.detach().cpu().numpy(),
                                                       scale=scale.detach().cpu().numpy(),
                                                       size=1000)
                            writer.add_histogram("histogram_%s_%d" % (key, d), density, epoch)

                # update progress bar
                pbar.set_description("SVI Loss: %.5f" % epoch_loss)
                pbar.update(1)

        self.vi_model = pyro.get_param_store().get_state()

        if tensorboard:
            writer.close()

    def convex(self, data: torch.Tensor, y: torch.Tensor, tensorboard: bool, log_dir: str):
        """
        Convex optimization to find the global optimum of current parameter search.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels as 1-D vector (binary).
        """

        # optimization objective function
        # compute NLL loss - fix weights given of the model for the current iteration step
        def MLE(w, x, y):

            data = {}
            start = 0
            for name, site in self._sites.items():
                num_weights = len(site['init']['mean'])
                data[name] = torch.from_numpy(w[start:start+num_weights]).to(self._device)
                start += num_weights

            return loss_op(torch.squeeze(pyro.condition(self.model, data=data)(x)), y).item()

        # convert input data to double as well as the weights
        # this might be necessary for the optimizer
        data = data.double()
        initial_weights = np.concatenate(
            [site['init']['mean'].cpu().numpy().astype(np.float64) for site in self._sites.values()]
        )

        # on detection or binary classification, use binary cross entropy loss and convert target vector to double
        if self.detection or self._is_binary_classification():

            # for an arbitrary reason, binary_cross_entropy_with_logits returns NaN
            # thus, we need to use the bce loss with sigmoid
            def loss_op(x, y):
                return torch.nn.BCELoss(reduction='mean')(torch.sigmoid(x), y)

            y = y.double()

        # on multiclass classification, use multiclass cross entropy loss and convert target vector to long
        else:
            loss_op = torch.nn.CrossEntropyLoss(reduction='mean')
            y = y.long()

        # convert pytorch optim bounds to scipy optimization format
        optim_bounds = self._get_scipy_constraints()

        # invoke SciPy's optimization function as this is very light-weight and fast
        result = minimize(fun=MLE, x0=initial_weights, args=(data, y), bounds=optim_bounds)

        # assign weights to according sites
        start = 0
        for name, site in self._sites.items():
            num_weights = len(site['init']['mean'])
            site['values'] = result.x[start:start + num_weights].astype(np.float32)
            start += num_weights

        # on some methods like Beta calibration, it is necessary to repeat the optimization
        # process if negative parameter estimates occur after training
        if self.mask_negative:

            # this method has to be implemented by the child class if it should be used
            masked_weights, bounds = self.mask()
            if bounds:
                # rerun minimization routine
                initial_weights[masked_weights] = 0.0
                result = minimize(fun=MLE, x0=initial_weights, args=(data, y), bounds=bounds)

        # get intercept and weights after optimization
        start = 0
        for name, site in self._sites.items():
            num_weights = len(site['init']['mean'])
            site['values'] = result.x[start:start + num_weights].astype(np.float32)
            start += num_weights

    def momentum(self, data: torch.Tensor, y: torch.Tensor, tensorboard: bool, log_dir: str):
        """
        Momentum optimization to find the global optimum of current parameter search.
        This method is slow but tends to find the global optimum for non-convex optimization.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels as 1-D vector (binary).
        """

        # initial learning rate, min delta for early stopping and patience
        # for early stopping (number of epochs without improvement)
        init_lr = 1e-3
        batch_size = 1024

        # criterion is Binary Cross Entropy on logits (numerically more stable)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # create dataset
        lr_dataset = torch.utils.data.TensorDataset(data.double(), y.double())
        data_loader = DataLoader(dataset=lr_dataset, batch_size=batch_size, pin_memory=False)

        # init model and optimizer
        parameters = [nn.Parameter(site['init']['mean']).to(self._device) for site in self._sites.values()]
        optimizer = torch.optim.Adam(parameters, lr=init_lr)

        best_loss = np.infty

        # use tqdm to log loop action
        with tqdm(total=self.momentum_epochs) as pbar:
            for epoch in range(self.momentum_epochs):

                # iterate over batches
                for train_x, train_y in data_loader:

                    condition = {}
                    for name, param in zip(self._sites.keys(), parameters):
                        condition[name] = param

                    logit = pyro.condition(self.model, data=condition)(train_x.to(self._device))
                    loss = criterion(logit, train_y.to(self._device))

                    # perform optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # early stopping
                    # if current loss is best so far, refresh memory
                    if loss < best_loss:
                        best_loss = loss

                        pbar.set_description("Best Loss: %.6f" % best_loss)
                        pbar.refresh()

                # refresh progress bar
                pbar.update(1)

        # convert pytorch optim bounds to scipy optimization format
        optim_bounds = self._get_scipy_constraints()

        # get parameter estimates for each site
        for site, param in zip(self._sites.values(), parameters):
            site['values'] = param.detach().cpu().numpy()

        # clip to optimization bounds afterwards because the last update step might not capture the
        # optimization boundaries
        if optim_bounds is not None:

            start = 0
            for name, site in self._sites.items():
                num_weights = len(site['init']['mean'])

                # use NumPy's clip function as this also supports arrays for clipping instead for
                # single scalars only
                site['values'] = np.clip(
                    site['values'],
                    [b[0] for b in optim_bounds[start:start+num_weights]],
                    [b[1] for b in optim_bounds[start:start+num_weights]]
                )
                start += num_weights

    # -----------------------------------------------------------------

    def transform(self, X: np.ndarray, num_samples: int = 1000, random_state: int = None,
                  mean_estimate: bool = False) -> np.ndarray:
        """
        After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        num_samples : int, optional, default: 1000
            Number of samples generated on MCMC sampling or Variational Inference.
        random_state : int, optional, default: None
            Fix the random seed for the random number
        mean_estimate : bool, optional, default: False
            If True, directly return the mean on probabilistic methods like MCMC or VI instead of the full
            distribution. This parameter has no effect on MLE.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes]) on MLE or on MCMC/VI if 'mean_estimate' is True
        or shape=(n_parameters, n_samples, [n_classes]) on VI, MCMC if 'mean_estimate' is False
            On MLE without uncertainty, return NumPy array with calibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).
            On VI or MCMC, return NumPy array with leading dimension as the number of sampled parameters from the
            log regression parameter distribution obtained by VI or MCMC.
        """

        def process_model(weights: dict) -> torch.Tensor:
            """ Fix model weights to the weight vector given as the parameter and return calibrated data. """

            # model will return pytorch tensor
            model = pyro.condition(self.model, data=weights)
            logit = model(data)

            # distinguish between detection, binary and multiclass classification
            if self.detection or self._is_binary_classification():
                calibrated = torch.sigmoid(logit)
            else:
                calibrated = torch.softmax(logit, dim=1)

            return calibrated

        # prepare input data
        X = super().transform(X)
        self.to(self._device)

        # convert input data and weights to torch (and possibly to CUDA)
        data = self.prepare(X).float().to(self._device)

        # if weights is 2-D matrix, we are in sampling mode
        # treat each row as a separate weights vector
        if self.method in ['variational', 'mcmc']:

            if mean_estimate:
                weights = {}

                # on MCMC sampling, use mean over all weights as mean weight estimate
                # TODO: we need to find another way since the parameters are conditionally dependent
                # TODO: revise!!! We often have log-normals instead of normal distributions,
                #  thus the mean will be a different
                if self.mcmc_model is not None:
                    for name, site in self._sites.items():
                        weights[name] = torch.from_numpy(np.mean(self.mcmc_model[name])).to(self._device)

                # on variational inference, use mean of the variational distribution for inference
                elif self.vi_model is not None:
                    for name, site in self._sites.items():
                        weights[name] = torch.from_numpy(self.vi_model['params']['%s_mean' % name]).to(self._device)

                else:
                    raise ValueError("Internal error: neither MCMC nor variational model given.")

                # on MLE without uncertainty, only return the single model estimate
                calibrated = process_model(weights).cpu().numpy()
                calibrated = self.squeeze_generic(calibrated, axes_to_keep=0)
            else:

                parameter = []
                if self.mcmc_model is not None:

                    with manual_seed(seed=random_state):
                        idxs = torch.randint(0, self.mcmc_steps, size=(num_samples,), device=self._device)
                        samples = {k: v.index_select(0, idxs) for k, v in self.mcmc_model.items()}

                elif self.vi_model is not None:

                    # restore state of global parameter store of pyro and use this parameter store for the predictive
                    pyro.get_param_store().set_state(self.vi_model)
                    predictive = Predictive(self.model, guide=self.guide,
                                            num_samples=num_samples,
                                            return_sites=tuple(self._sites.keys()))

                    with manual_seed(seed=random_state):
                        samples = predictive(data)

                else:
                    raise ValueError("Internal error: neither MCMC nor variational model given.")

                # remove unnecessary dims that possibly occur on MCMC or VI
                samples = {k: torch.reshape(v, (num_samples, -1)) for k, v in samples.items()}

                # iterate over all parameter sets
                for i in range(num_samples):
                    param_dict = {}

                    # iterate over all sites and store into parameter dict
                    for site in self._sites.keys():
                        param_dict[site] = samples[site][i].detach().to(self._device)

                    parameter.append(param_dict)

                calibrated = []

                # iterate over all parameter collections and compute calibration mapping
                for param_dict in parameter:
                    cal = process_model(param_dict)
                    calibrated.append(cal)

                # stack all calibrated estimates along axis 0 and calculate stddev as well as mean
                calibrated = torch.stack(calibrated, dim=0).cpu().numpy()
                calibrated = self.squeeze_generic(calibrated, axes_to_keep=(0, 1))
        else:

            # extract all weight values of sites and store into single dict
            weights = {}
            for name, site in self._sites.items():
                weights[name] = torch.from_numpy(site['values']).to(self._device)

            # on MLE without uncertainty, only return the single model estimate
            calibrated = process_model(weights).cpu().numpy()
            calibrated = self.squeeze_generic(calibrated, axes_to_keep=0)

        # delete torch data tensor
        del data

        # if device is cuda, empty GPU cache to free memory
        if self._device.type == 'cuda':
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()

        return calibrated

    @dimensions((1, 2))
    def _inverse_sigmoid(self, confidence: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """ Calculate inverse of Sigmoid to get Logit. """

        # on torch tensors, use torch built-in functions
        if isinstance(confidence, torch.Tensor):

            # clip normal and inverse separately due to numerical stability
            clipped = torch.clamp(confidence, self.epsilon, 1. - self.epsilon)
            inv_clipped = torch.clamp(1. - confidence, self.epsilon, 1. - self.epsilon)

            logit = torch.log(clipped) - torch.log(inv_clipped)
            return logit

        # use NumPy method otherwise
        else:
            clipped = np.clip(confidence, self.epsilon, 1. - self.epsilon)
            return safe_logit(clipped)

    @dimensions(2)
    def _inverse_softmax(self, confidences: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """ Calculate inverse of multi class softmax. """

        # on torch tensors, use torch built-in functions
        if isinstance(confidences, torch.Tensor):
            clipped = torch.clamp(confidences, self.epsilon, 1. - self.epsilon)
            return torch.log(clipped)

        # use NumPy methods otherwise
        else:
            clipped = np.clip(confidences, self.epsilon, 1. - self.epsilon)
            return np.log(clipped)

    def _get_scipy_constraints(self) -> List:
        """ Convert list of optimization constraints defined in Pytorch to list of tuples for NumPy/Scipy. """

        numpy_bounds = []

        # iterate over bias and weights constraints
        for site in self._sites.values():

            bound = [-np.infty, np.infty]
            constraint = site['constraint']
            num_parameters = len(site['init']['mean'])

            # check if constraint object has attributes for lower_bound or upper_bound
            if constraint is not None:
                if hasattr(constraint, 'lower_bound'):
                    bound[0] = constraint.lower_bound
                if hasattr(constraint, 'upper_bound'):
                    bound[1] = constraint.upper_bound

            numpy_bounds.extend([tuple(bound), ] * num_parameters)

        return numpy_bounds
