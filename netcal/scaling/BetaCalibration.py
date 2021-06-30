# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple, List
from collections import OrderedDict
from typing import Union
import numpy as np

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist

from netcal.scaling import AbstractLogisticRegression


class BetaCalibration(AbstractLogisticRegression):
    """
    On classification, apply the beta calibration method to obtain a calibration mapping. The original method was
    proposed by [1]_.
    For the multiclass case, we extended this method to work with multinomial logistic regression instead of a
    one vs. all calibration mapping.
    On detection mode, this calibration method uses multiple independent Beta distributions to obtain a
    calibration mapping by means of the confidence as well as additional features [2]_. This calibration scheme
    assumes independence between all variables.

    On detection, it is necessary to provide all data in input parameter ``X`` as an NumPy array
    of shape ``(n_samples, n_features)``,
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

    If we assume independence of all variables given in :math:`s`, we can use multiple univariate probability
    density distributions to obtain a calibration mapping. Using multiple beta distributions (one for each variable
    and one for each case :math:`\\text{m}=1` and :math:`\\text{m}=0`), the log-likelihood ratio can be
    expressed as

    .. math::
       \\ell r(s) = \\sum^K_{k=1} \\log \\Bigg(\\frac{\\text{Beta}(s_k | \\alpha^+_k, \\beta^+_k)}
                                                     {\\text{Beta}(s_k | \\alpha^-_k, \\beta^-_k)}\\Bigg)

    with the beta distribution :math:`\\text{Beta}(s_k|\\alpha, \\beta)`. The shape parameters
    for :math:`\\text{m}=1` and :math:`\\text{m}=0` are denoted by :math:`\\alpha_k^+, \\beta_k^+` and
    :math:`\\alpha_k^-, \\beta_k^-`, respectively. We can reparametrize this expression by using

    .. math::
       \\ell r(s) = c + \\sum^K_{k=1} a_k \\log(s_k) - b_k \\log(1-s_k) ,

    where the distribution parameters are summarized by :math:`a_k=\\alpha_k^+-\\alpha_k^-`,
    :math:`b_k=\\beta_k^--\\beta_k^+` and
    :math:`c=\\sum_k \\log B(\\alpha_k^-, \\beta_k^-) - \\log B(\\alpha_k^+, \\beta_k^+)`.
    We utilize standard optimization methods to determine the calibration mapping :math:`g(s)`.

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
    .. [1] Kull, Meelis, Telmo Silva Filho, and Peter Flach:
       "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
       Artificial Intelligence and Statistics, PMLR 54:623-631, 2017
       `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`_

    .. [2] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

    .. [3] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021
    """

    def __init__(self, *args, **kwargs):
        """ Create an instance of `BetaCalibration`. Detailed parameter description given in class docs. """

        super().__init__(*args, **kwargs)
        self.mask_negative = True

    # -------------------------------------------------

    @property
    def intercept(self) -> Union[np.ndarray, float]:
        """ Getter for intercept of logistic calibration. """
        if self._sites is None:
            raise ValueError("Intercept is None. You have to call the method 'fit' first.")

        return self._sites['bias']['values']

    @property
    def weights(self) -> Union[np.ndarray, float]:
        """ Getter for weights of beta calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        return self._sites['weights']['bias']

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

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # build 2-D array with [log(x), -log(1-x)]
        # convert confidences array and clip values to open interval (0, 1)

        # on detection, create array of shape (n_samples, 2*n_features)
        if self.detection:
            features = []
            for i in range(X.shape[1]):
                features.append(X[:, i])
                features.append(1. - X[:, i])

            data_input = np.stack(features, axis=1)
            data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)
            data_input = np.log(data_input)
            data_input[:, 1::2] *= -1

        # on classification, create an array
        # - binary classification: shape (n_samples, 2)
        # - multiclass classification: shape (n_samples, n_classes, 2)
        else:
            features = []
            for i in range(X.shape[1]):
                features.append(np.stack([X[:, i], 1. - X[:, i]], axis=1))

            data_input = features[0] if self._is_binary_classification() else np.stack(features, axis=1)
            data_input = np.clip(data_input, self.epsilon, 1. - self.epsilon)
            data_input = np.log(data_input)
            data_input[..., 1] *= -1

        return torch.tensor(data_input)

    def prior(self):
        """
        Prior definition of the weights used for log regression. This function has to set the
        variables 'self.weight_prior_dist', 'self.weight_mean_init' and 'self.weight_stddev_init'.
        """

        self._sites = OrderedDict()

        # on detection mode or binary classification, we have a weight for each given feature (one for binary
        # classification) and bias
        if self.detection or self._is_binary_classification():
            num_bias = 1
            num_weights = 2
            feature_weights = 2 * (self.num_features - 1)

            # store feature weights separately from confidence weights as they are unconstrained
            # in constrast to the confidence weights
            if feature_weights > 0:
                self._sites['feature_weights'] = {
                    'values': None,
                    'constraint': constraints.real,
                    'init': {
                        'mean': torch.ones(feature_weights),
                        'scale': torch.ones(feature_weights)
                    },
                    'prior': dist.Normal(torch.ones(feature_weights), 10 * torch.ones(feature_weights), validate_args=True),
                }

        # on multiclass classification, we have one weight and one bias for each class separately
        else:
            num_bias = self.num_classes
            num_weights = 2*self.num_classes

        # initialize weight mean by ones and set prior distribution
        init_mean = torch.ones(num_weights)
        init_scale = torch.ones(num_weights)
        prior = dist.Normal(init_mean, 10 * init_scale, validate_args=True)

        # we have constraints on the weights for the confidence
        # this is usually solved by removing dims with invalid weights on MLE
        # however, on MCMC and VI this is not possible
        # instead, we are using a "shifted" LogNormal to obtain only positive samples
        if self.method in ['variational', 'mcmc']:

            # for this purpose, we need to transform the prior mean first and set the
            # distribution to be a LogNormal
            init_mean = torch.log(torch.exp(init_mean)-1)
            prior = dist.LogNormal(init_mean, 10 * init_scale, validate_args=True)

        # set sites for weights and bias
        self._sites['weights'] = {
            'values': None,
            'constraint': constraints.positive,
            'init': {
                'mean': init_mean,
                'scale': init_scale
            },
            'prior': prior
        }

        self._sites['bias'] = {
            'values': None,
            'constraint': constraints.real,
            'init': {
                'mean': torch.zeros(num_bias),
                'scale': torch.ones(num_bias)
            },
            'prior': dist.Normal(torch.zeros(num_bias), 10 * torch.ones(num_bias), validate_args=True)
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

        # sample from prior - on MLE, this weight will be set as conditional
        bias = pyro.sample("bias", self._sites['bias']['prior'])
        weights = pyro.sample("weights", self._sites['weights']['prior'])

        # on MCMC or VI, use samples obtained by a "shifted" LogNormal
        # the "shifted" +1 term guarantees positive samples only
        if self.method in ['variational', 'mcmc']:
            weights = torch.log(weights + 1)
            assert (weights >= 0).all().item() == True, "Negative confidence weights are not allowed."

            # on MCMC sampling, extreme values might occur and can cause an 'inf'
            # this will result in invalid prob values - catch infs and set to log of max value
            weights[torch.isinf(weights)] = torch.log(torch.tensor(torch.finfo(weights.dtype).max))

        # additional weights are extra weights for extra features (on detection mode)
        if "feature_weights" in self._sites.keys():
            feature_weights = pyro.sample("feature_weights", self._sites['feature_weights']['prior'])
            weights = torch.cat((weights, feature_weights))

        # the first dimension of the given input data is the "independent" sample dimension
        with pyro.plate("data", X.shape[0]):

            if self.detection or self._is_binary_classification():
                weights = torch.reshape(weights, (-1, 1))

                # compute logits and remove unnecessary dimensions
                logit = torch.squeeze(torch.matmul(X, weights) + bias)
                probs = torch.sigmoid(logit)
                dist_op = dist.Bernoulli

            else:

                # get number of weights and biases according to number of classes
                weights = torch.reshape(weights, (-1, 2, 1))

                # use broadcast mechanism of pytorch: if 3 dimensions are provided, treat first dimension
                # as a stack of matrices -> this speeds up the calculation
                result = torch.matmul(X.permute(1, 0, 2), weights)

                # as a result, we obtain an array of shape (n_classes, n_samples, 1)
                # remove last dim and swap axes
                logit = torch.transpose(torch.squeeze(result, dim=2), 0, 1) + bias
                probs = torch.softmax(logit, dim=1)
                dist_op = dist.Categorical

            # if MLE, (slow) sampling is not necessary. However, this is needed for 'variational' and 'mcmc'
            if self.method in ['variational', 'mcmc']:
                pyro.sample("obs", dist_op(probs=probs, validate_args=True), obs=y)

            return logit

    def mask(self) -> Tuple[np.ndarray, List]:
        """
        Seek for all relevant weights whose values are negative. Mask those values with optimization constraints
        in the interval [0, 0].

        Returns
        -------
        list
            Indices of masked values.
        """

        # get weights from sites
        weights = self._sites['weights']['values']
        num_weights = len(weights)

        # --------------------------------------------------------------------------------
        # check calculated weights
        # if either param a or param b < 0, the second distribution's parameter is fixed to zero
        # the logistic fit is repeated afterwards with remaining distribution

        # now check if negative entries are present
        masked_weights = np.where((weights.flatten() < 0))[0]

        # weights below 0 found?
        # we need to make sure that not both weights are masked out for a single class
        if len(masked_weights):

            # on detection or binary classification, only keep the confidence dimension monotonically increasing
            # this is equivalent to the first dimension (weights 0 and 1)
            if self.detection or self._is_binary_classification():
                if len(masked_weights) == 2:
                    masked_weights = np.array([masked_weights[0]])

            # same on multiclass classification but for each class
            else:
                for cls in range(self.num_classes):
                    index_a = 2 * cls
                    index_b = 2 * cls + 1

                    # remove index of the second weight for this certain class from the masked weights array
                    if index_a in masked_weights and index_b in masked_weights:
                        masked_weights = np.delete(masked_weights, np.argwhere(masked_weights == index_b))

            # increase indices for optimization routine due to bias values
            masked_weights += 1 if self.detection or self._is_binary_classification() else self.num_classes

            # masking of the values is done by setting linear constraints for the masked values
            # make sure that the remaining values are also >0 by settings the bounds accordingly
            # first part: bounds for the bias (None); second part: bounds for the parameters
            if self.detection or self._is_binary_classification():
                bounds = [(None, None), ] + \
                         [(0, 0) if i in masked_weights else (0, None) for i in range(num_weights - 1)]
            else:
                bounds = [(None, None), ] * self.num_classes + \
                         [(0, 0) if i in masked_weights else (0, None) for i in range(num_weights - self.num_classes)]

            # prepend bounds by (non-present) limits for intercept
            bounds = [(None, None), ] * len(self._sites['bias']['values']) + bounds
            return masked_weights, bounds
        else:
            return masked_weights, []
