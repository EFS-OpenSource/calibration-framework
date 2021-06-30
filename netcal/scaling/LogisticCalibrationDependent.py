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


class LogisticCalibrationDependent(AbstractLogisticRegression):
    """
    This calibration method is for detection only and uses multivariate normal distributions to obtain a
    calibration mapping by means of the confidence as well as additional features. This calibration scheme
    tries to model several dependencies in the variables given by the input ``X`` [1]_.

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

    According to [2]_, we can interpret the logit :math:`z` as the logarithm of the posterior odds

    .. math::

       z(s) = \\log \\frac{f(\\text{m}=1 | s)}{f(\\text{m}=0 | s)} \\approx
       \\log \\frac{f(s | \\text{m}=1)}{f(s | \\text{m}=1)} = \\ell r(s)

    Inserting multivariate normal density distributions into this framework with
    :math:`\\mu^+, \\mu^- \\in \\mathbb{R}^K` and :math:`\\Sigma^+, \\Sigma^- \\in \\mathbb{R}^{K \\times K}`
    as the mean vectors and covariance matrices for :math:`\\text{m}=1` and
    :math:`\\text{m}=0`, respectively, we get a likelihood ratio of

    .. math::

       \\ell r(s) = \\log \\frac{\\Sigma^-}{\\Sigma^+}
       + \\frac{1}{2} (s_-^T \\Sigma_-^{-1}s^-) - (s_+^T \\Sigma_+^{-1}s^+),

    with :math:`s^+ = s - \\mu^+` and :math:`s^- = s - \\mu^-`.

    To keep the restrictions to covariance matrices (symmetric and positive semidefinit), we optimize a decomposed
    matrix V as

    .. math::
       \\Sigma = V^T V

    instead of estimating :math:`\\Sigma` directly. This guarantees both requirements.

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

    .. [2] Kull, Meelis, Telmo Silva Filho, and Peter Flach:
       "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
       Artificial Intelligence and Statistics, PMLR 54:623-631, 2017
       `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`_

    .. [3] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021
    """

    def __init__(self, *args, **kwargs):
        """ Create an instance of `LogisticCalibrationDependent`. Detailed parameter description given in class docs. """

        # an instance of this class is definitely of type detection
        if 'detection' in kwargs and kwargs['detection'] == False:
            print("WARNING: On LogisticCalibrationDependent, attribute \'detection\' must be True.")

        kwargs['detection'] = True
        super().__init__(*args, **kwargs)

    # -------------------------------------------------

    @property
    def intercept(self) -> float:
        """ Getter for intercept of dependent logistic calibration. """
        if self._sites is None:
            raise ValueError("Intercept is None. You have to call the method 'fit' first.")

        return self._sites['bias']['values']

    @property
    def means(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Getter for mean vectors of dependent logistic calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        index_1 = 2 * (self.num_features ** 2)
        index_2 = index_1 + self.num_features

        weights = self._sites['weights']['values']
        return weights[index_1:index_2], weights[index_2:]

    @property
    def covariances(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Getter for covariance matrices of dependent logistic calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        index_1 = self.num_features ** 2
        index_2 = index_1 + self.num_features ** 2

        weights = self._sites['weights']['values']

        decomposed_inv_cov_pos = np.reshape(weights[:index_1], (self.num_features, self.num_features))
        decomposed_inv_cov_neg = np.reshape(weights[index_1:index_2], (self.num_features, self.num_features))

        inv_cov_pos = np.matmul(decomposed_inv_cov_pos.T, decomposed_inv_cov_pos.T)
        inv_cov_neg = np.matmul(decomposed_inv_cov_neg.T, decomposed_inv_cov_neg.T)

        cov_pos = np.linalg.inv(inv_cov_pos)
        cov_neg = np.linalg.inv(inv_cov_neg)

        return cov_pos, cov_neg

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

        assert self.detection, "Detection mode must be enabled for dependent logistic calibration."

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # on detection mode, convert confidence to sigmoid and append the remaining features
        data_input = np.concatenate((self._inverse_sigmoid(X[:, 0]).reshape(-1, 1), X[:, 1:]), axis=1)
        return torch.Tensor(data_input)

    def prior(self):
        """
        Prior definition of the weights used for log regression. This function has to set the
        variables 'self.weight_prior_dist', 'self.weight_mean_init' and 'self.weight_stddev_init'.
        """

        # number of weights
        num_weights = 2 * (self.num_features ** 2 + self.num_features)

        # prior estimates for decomposed inverse covariance matrices and mean vectors
        decomposed_inv_cov_prior = torch.diag(torch.ones(self.num_features))
        mean_mean_prior = torch.ones(self.num_features)

        # initial stddev for all weights is always the same
        weights_mean_prior = torch.cat((decomposed_inv_cov_prior.flatten(),
                                        decomposed_inv_cov_prior.flatten(),
                                        mean_mean_prior.flatten(),
                                        mean_mean_prior.flatten()))

        self._sites = OrderedDict()

        # set properties for "weights"
        self._sites['weights'] = {
            'values': None,
            'constraint': constraints.real,
            'init': {
                'mean': weights_mean_prior,
                'scale': torch.ones(num_weights)
            },
            'prior': dist.Normal(weights_mean_prior, 10 * torch.ones(num_weights), validate_args=True),
        }

        # set properties for "bias"
        self._sites['bias'] = {
            'values': None,
            'constraint': constraints.real,
            'init': {
                'mean': torch.zeros(1),
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
        index_1 = int(np.power(self.num_features, 2))
        index_2 = index_1 + int(np.power(self.num_features, 2))
        index_3 = index_2 + self.num_features

        # sample from prior - on MLE, this weight will be set as conditional
        bias = pyro.sample("bias", self._sites["bias"]["prior"])
        weights = pyro.sample("weights", self._sites["weights"]["prior"])

        # the first dimension of the given input data is the "independent" sample dimension
        with pyro.plate("data", X.shape[0]):

            # get weights of decomposed cov matrices V^(-1)
            decomposed_inv_cov_pos = torch.reshape(weights[:index_1], (self.num_features, self.num_features))
            decomposed_inv_cov_neg = torch.reshape(weights[index_1:index_2], (self.num_features, self.num_features))

            mean_pos = weights[index_2:index_3]
            mean_neg = weights[index_3:]

            # get logits by calculating gaussian ratio between both distributions
            # calculate covariance matrices
            # COV^(-1) = V^(-1) * V^(-1,T)
            inverse_cov_pos = torch.matmul(decomposed_inv_cov_pos, decomposed_inv_cov_pos.transpose(1, 0))
            inverse_cov_neg = torch.matmul(decomposed_inv_cov_neg, decomposed_inv_cov_neg.transpose(1, 0))

            # calculate data without means
            difference_pos = X - mean_pos
            difference_neg = X - mean_neg

            # add a new dimensions. This is necessary for torch to distribute dot product
            difference_pos = torch.unsqueeze(difference_pos, 2)
            difference_neg = torch.unsqueeze(difference_neg, 2)

            logit = 0.5 * (torch.matmul(difference_neg.transpose(2, 1),
                                        torch.matmul(inverse_cov_neg, difference_neg)) -
                           torch.matmul(difference_pos.transpose(2, 1),
                                        torch.matmul(inverse_cov_pos, difference_pos))
                           )

            # remove unnecessary dimensions
            logit = torch.squeeze(logit)

            # add bias ratio to logit
            logit = bias + logit

            # if MLE, (slow) sampling is not necessary. However, this is needed for 'variational' and 'mcmc'
            if self.method in ['variational', 'mcmc']:
                probs = torch.sigmoid(logit)
                pyro.sample("obs", dist.Bernoulli(probs=probs, validate_args=True), obs=y)

        return logit
