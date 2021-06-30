# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple
from collections import OrderedDict
import numpy as np

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist

from netcal.scaling import AbstractLogisticRegression


class BetaCalibrationDependent(AbstractLogisticRegression):
    """
    This calibration method uses a multivariate variant of a Beta distribution to obtain a
    calibration mapping by means of the confidence as well as additional features. This method is originally
    proposed by [1]_. This calibration scheme
    tries to model several dependencies in the variables given by the input ``X``.

    It is necessary to provide all data in input parameter ``X`` as an NumPy array of shape ``(n_samples, n_features)``,
    whereas the confidence must be the first feature given in the input array. The ground-truth samples ``y``
    must be an array of shape ``(n_samples,)`` consisting of binary labels :math:`y \\in \\{0, 1\\}`. Those
    labels indicate if the according sample has matched a ground truth box :math:`\\text{m}=1` or is a false
    prediction :math:`\\text{m}=0`.

    **Mathematical background:** For confidence calibration in classification tasks, a
    confidence mapping :math:`g` is applied on top of a miscalibrated scoring classifier :math:`\\hat{p} = h(x)` to
    deliver a calibrated confidence score :math:`\\hat{q} = g(h(x))`.

    For detection calibration, we can also use the additional box regression output which we denote as
    :math:`\\hat{r} \\in [0, 1]^J` with :math:`J` as the number of dimensions used for the box encoding (e.g.
    :math:`J=4` for x position, y position, width and height).
    Therefore, the calibration map is not only a function of the confidence score, but also of :math:`\\hat{r}`.
    To define a general calibration map for binary problems, we use the logistic function and the combined
    input :math:`s = (\\hat{p}, \\hat{r})` of size K by

    .. math::

       g(s) = \\frac{1}{1 + \\exp(-z(s))} ,

    According to [1]_, we can interpret the logit :math:`z` as the logarithm of the posterior odds

    .. math::

       z(s) = \\log \\frac{f(\\text{m}=1 | s)}{f(\\text{m}=0 | s)} \\approx
       \\log \\frac{f(s | \\text{m}=1)}{f(s | \\text{m}=1)} = \\ell r(s)

    For a multivariate probability density function :math:`f(s|\\text{m})`, we use a variant of the beta distribution
    described in [2]_ and given by

    .. math::

       f(s|\\text{m}) =  \\frac{1}{B(\\alpha_0, ..., \\alpha_K)}
                         \\frac{\\prod^K_{k=1} \\lambda_k^{\\alpha_k}(s_k^\\ast)^{\\alpha_k - 1}
                         \\Big(\\frac{s_k^\\ast}{s_k}\\Big)^2}
                         {\\Big[1 + \\sum^K_{k=1} \\lambda_k
                           s_k^\\ast\\Big]^{\\sum^K_{k=0} \\alpha_k}
                         }

    with shape parameters :math:`\\alpha_k, \\beta_k > 0`, :math:`\\forall k \\in \\{0, ..., K \\}`. For notation
    easyness, we denote :math:`\\lambda_k=\\frac{\\beta_k}{\\beta_0}` and :math:`s^\\ast=\\frac{s}{1-s}`.
    Inserting this density function into this framework with :math:`\\alpha_k^+`, :math:`\\beta_k^+` and
    :math:`\\alpha_k^-`, :math:`\\beta_k^-` as the distribution parameters for :math:`\\text{m}=1` and
    :math:`\\text{m}=0`, respectively, we get a likelihood ratio of

    .. math::

       \\ell r(s) &= \\sum^K_{k=1} \\alpha_k^+ \\log(\\lambda_k^+) - \\alpha_k^- \\log(\\lambda_k^-) \\\\
                  &+ \\sum^K_{k=1} (\\alpha_k^+ - \\alpha_k^-) \\log(s^\\ast) \\\\
                  &+ \\sum^K_{k=0} \\alpha_k^- \\log\\Bigg[\\sum^K_{j=1} \\lambda_j^- s^\\ast_j\\Bigg] \\\\
                  &- \\sum^K_{k=0} \\alpha_k^+ \\log\\Bigg[\\sum^K_{j=1} \\lambda_j^+ s^\\ast_j\\Bigg] \\\\
                  &+ c ,

    where  and
    :math:`c=\\log B(\\alpha_0^-, ..., \\alpha_k^-) - \\log B(\\alpha_0^+, ..., \\alpha^+_k)`.

    This is optimized by an Adam optimizer with a learning rate of 1e-3 and a batch size of 256 for
    1000 iterations (default).


    Parameters
    ----------
    method : str, default: "momentum"
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
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).
    use_cuda : str or bool, optional, default: False
        Specify if CUDA should be used. If str, you can also specify the device
        number like 'cuda:0', etc.

    References
    ----------
    .. [1] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

    .. [2] Libby, David L., and Melvin R. Novick:
       "Multivariate generalized beta distributions with applications to utility assessment"
       Journal of Educational Statistics 7.4, pp. 271-294, 1982

    .. [3] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021
    """

    def __init__(self, *args, method: str = 'momentum', **kwargs):
        """ Create an instance of `BetaCalibrationDependent`. Detailed parameter description given in class docs. """

        # an instance of this class is definitely of type detection
        if 'detection' in kwargs and kwargs['detection'] == False:
            print("WARNING: On BetaCalibrationDependent, attribute \'detection\' must be True.")

        kwargs['detection'] = True
        super().__init__(*args, method=method, **kwargs)

    # -------------------------------------------------

    @property
    def intercept(self) -> float:
        """ Getter for intercept of dependent beta calibration. """
        if self._sites is None:
            raise ValueError("Intercept is None. You have to call the method 'fit' first.")

        return self._sites['bias']['values']

    @property
    def alphas(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Getter for alpha values of dependent beta calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        index_2 = self.num_features + 1
        index_3 = index_2 + self.num_features + 1
        weights = self._sites['weights']['values']

        return weights[:index_2], weights[index_2:index_3]

    @property
    def betas(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Getter for beta values of dependent beta calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        index_1 = 2 * (self.num_features + 1)
        index_2 = index_1 + self.num_features + 1
        weights = self._sites['weights']['values']

        return weights[index_1:index_2], weights[index_2:]

    # -------------------------------------------------

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

        assert self.detection, "Detection mode must be enabled for dependent beta calibration."

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # clip seperately due to numerical stability
        data = np.clip(X, self.epsilon, 1. - self.epsilon) / np.clip((1. - X), self.epsilon, 1. - self.epsilon)
        return torch.tensor(data)

    def prior(self):
        """
        Prior definition of the weights used for log regression. This function has to set the
        variables 'self.weight_prior_dist', 'self.weight_mean_init' and 'self.weight_stddev_init'.
        """

        # number of weights
        num_weights = 4 * (self.num_features + 1)
        self._sites = OrderedDict()

        # initial values for mean, scale and prior dist
        init_mean = torch.ones(num_weights).uniform_(1. + self.epsilon, 2.)
        init_scale = torch.ones(num_weights)
        prior = dist.Normal(init_mean, 10 * init_scale, validate_args=True)

        # we have constraints on the weights to be positive
        # this is usually solved by defining constraints on the MLE optimizer
        # however, on MCMC and VI this is not possible
        # instead, we are using a "shifted" LogNormal to obtain only positive samples

        if self.method in ['variational', 'mcmc']:

            # for this purpose, we need to transform the prior mean first and set the
            # distribution to be a LogNormal
            init_mean = torch.log(torch.exp(init_mean) - 1)
            prior = dist.LogNormal(init_mean, 10 * init_scale, validate_args=True)

        # set properties for "weights": weights must be positive
        self._sites['weights'] = {
            'values': None,
            'constraint': constraints.greater_than(self.epsilon),
            'init': {
                'mean': init_mean,
                'scale': init_scale
            },
            'prior': prior
        }

        # set properties for "bias"
        self._sites['bias'] = {
            'values': None,
            'constraint': constraints.real,
            'init': {
                'mean': torch.ones(1) * self.epsilon,
                'scale': torch.ones(1)
            },
            'prior': dist.Normal(torch.zeros(1), 10 * torch.ones(1), validate_args=True),
        }

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

        # get indices of weights
        index_1 = self.num_features+1
        index_2 = index_1 + self.num_features+1
        index_3 = index_2 + self.num_features+1

        # sample from prior - on MLE, this weight will be set as conditional
        bias = pyro.sample("bias", self._sites['bias']['prior'])
        weights = pyro.sample("weights", self._sites['weights']['prior'])

        # on MCMC or VI, use samples obtained by a "shifted" LogNormal
        # the "shifted" +1 term guarantees positive samples only
        if self.method in ['variational', 'mcmc']:
            weights = torch.log(weights + 1)
            assert (weights >= 0).all().item() == True, "Negative weights are not allowed."

            # on MCMC sampling, extreme values might occur and can cause an 'inf'
            # this will result in invalid prob values - catch infs and set to log of max value
            weights[torch.isinf(weights)] = torch.log(torch.tensor(torch.finfo(weights.dtype).max))

        # the first dimension of the given input data is the "independent" sample dimension
        with pyro.plate("data", X.shape[0]):

            # clip values to range (0, inf]
            alpha_pos = torch.clamp(weights[:index_1], self.epsilon, np.infty)
            alpha_neg = torch.clamp(weights[index_1:index_2], self.epsilon, np.infty)
            beta_pos = torch.clamp(weights[index_2:index_3], self.epsilon, np.infty)
            beta_neg = torch.clamp(weights[index_3:], self.epsilon, np.infty)

            # lambdas are ratio between all betas and beta_0
            lambda_pos = beta_pos[1:] / beta_pos[0]
            lambda_neg = beta_neg[1:] / beta_neg[0]
            log_lambdas_upper = alpha_pos[1:] * torch.log(lambda_pos) - alpha_neg[1:] * torch.log(lambda_neg)

            # parameter differences
            differences_alpha_upper = alpha_pos[1:] - alpha_neg[1:]
            log_values_upper = torch.log(X)

            # calculate upper part
            upper_part = torch.sum(log_lambdas_upper + (differences_alpha_upper * log_values_upper), dim=1)

            # start with summation of alphas for lower part of equation
            sum_alpha_pos = torch.sum(alpha_pos)
            sum_alpha_neg = torch.sum(alpha_neg)

            # calculate lower part
            log_sum_lower_pos = torch.log(1. + torch.sum(lambda_pos * X, dim=1))
            log_sum_lower_neg = torch.log(1. + torch.sum(lambda_neg * X, dim=1))
            lower_part = (sum_alpha_neg * log_sum_lower_neg) - (sum_alpha_pos * log_sum_lower_pos)

            # combine both parts and bias to logits
            logit = torch.squeeze(bias + upper_part + lower_part)

            # if MLE, (slow) sampling is not necessary. However, this is needed for 'variational' and 'mcmc'
            if self.method in ['variational', 'mcmc']:
                probs = torch.sigmoid(logit)
                pyro.sample("obs", dist.Bernoulli(probs=probs, validate_args=True), obs=y)

        return logit
