# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist

from netcal.scaling import AbstractLogisticRegression


class LogisticCalibration(AbstractLogisticRegression):
    """
    On classification, apply the logistic calibration method aka Platt scaling to obtain a
    calibration mapping. This method is originally proposed by [1]_.
    For the multiclass case, we use the Vector scaling proposed in [2]_.
    On detection mode, this calibration method uses multiple independent normal distributions to obtain a
    calibration mapping by means of the confidence as well as additional features [3]_. This calibration scheme
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
    density distributions with the same variance to obtain a calibration mapping. Using this formulation, we can
    simply extend the scaling factor (from classification logistic calibration) to a scaling
    vector :math:`w \\in \\mathbb{R}^K`.
    However, instead of using the uncalibrated confidence estimate :math:`\\hat{p}`, we use the logit of the
    network as part of :math:`s` to be conform with the original formulation in [1]_ and [2]_. Thus,
    the log-likelihood ratio can be expressed as

    .. math::
       \\ell r(s) = s^T w + c,

    with bias :math:`c \\in \\mathbb{R}`.
    We utilize standard optimization methods to determine the calibration mapping :math:`g(s)`.

    Parameters
    ----------
    temperature_only : bool, default: False
        If True, use Temperature Scaling instead of Platt/Vector Scaling.
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
    .. [1] Platt, John:
       "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods."
       Advances in large margin classifiers 10.3: 61-74, 1999
       `Get source online <https://www.researchgate.net/profile/John_Platt/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000.pdf>`_

    .. [2] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
       "On Calibration of Modern Neural Networks."
       Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
       `Get source online <https://arxiv.org/abs/1706.04599>`_

    .. [3] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

    .. [4] Fabian Küppers, Jan Kronenberger, Jonas Schneider  and Anselm Haselhoff:
       "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling."
       2021 IEEE Intelligent Vehicles Symposium (IV), 2021
    """

    def __init__(self, *args, temperature_only: bool = False, **kwargs):
        """ Create an instance of `LogisticCalibration`. Detailed parameter description given in class docs. """

        super().__init__(*args, **kwargs)
        self.temperature_only = temperature_only

    # -------------------------------------------------

    @property
    def intercept(self) -> Union[np.ndarray, float]:
        """ Getter for intercept of logistic calibration. """
        if self._sites is None:
            raise ValueError("Intercept is None. You have to call the method 'fit' first.")

        if self.temperature_only:
            raise ValueError("There is no intercept for temperature scaling.")

        return self._sites['bias']['values']

    @property
    def weights(self) -> Union[np.ndarray, float]:
        """ Getter for weights of logistic calibration. """
        if self._sites is None:
            raise ValueError("Weights is None. You have to call the method 'fit' first.")

        return self._sites['weights']['values']

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

        # on detection mode, convert confidence to sigmoid and append the remaining features
        if self.detection:
            data_input = np.concatenate((self._inverse_sigmoid(X[:, 0]).reshape(-1, 1), X[:, 1:]), axis=1)

        # on binary classification, simply convert the confidences to logits
        elif self._is_binary_classification():
            data_input = self._inverse_sigmoid(X)

        # on multiclass classification, use inverse softmax instead
        else:
            data_input = self._inverse_softmax(X)

        return torch.Tensor(data_input)

    def prior(self):
        """
        Prior definition of the weights used for log regression. This function has to set the
        variables 'self.weight_prior_dist', 'self.weight_mean_init' and 'self.weight_stddev_init'.
        """

        self._sites = OrderedDict()

        # on temperature scaling, we only have one single weight for all classes
        if self.temperature_only:
            self._sites['weights'] = {
                'values': None,
                'constraint': constraints.real,
                'init': {
                    'mean': torch.ones(1),
                    'scale': torch.ones(1)
                    },
                'prior': dist.Normal(torch.ones(1), 10 * torch.ones(1), validate_args=True)
            }

        else:

            # on detection mode or binary classification, we have a weight for each given feature (one for binary
            # classification) and bias
            if self.detection or self._is_binary_classification():
                num_bias = 1
                num_weights = self.num_features

            # on multiclass classification, we have one weight and one bias for each class separately
            else:
                num_bias = self.num_classes
                num_weights = self.num_classes

            # set properties for "weights"
            self._sites['weights'] = {
                'values': None,
                'constraint': constraints.real,
                'init': {
                    'mean': torch.ones(num_weights),
                    'scale': torch.ones(num_weights)
                },
                'prior': dist.Normal(torch.ones(num_weights), 10 * torch.ones(num_weights), validate_args=True),
            }

            # set properties for "bias"
            self._sites['bias'] = {
                'values': None,
                'constraint': constraints.real,
                'init': {
                    'mean': torch.zeros(num_bias),
                    'scale': torch.ones(num_bias)
                },
                'prior': dist.Normal(torch.zeros(num_bias), 10 * torch.ones(num_bias), validate_args=True),
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
        weights = pyro.sample("weights", self._sites["weights"]["prior"])

        if self.temperature_only:
            bias = 0.
        else:
            bias = pyro.sample("bias", self._sites["bias"]["prior"])

        # on detection or binary classification, use dot product to sum up all given features to one logit
        if self.detection or self._is_binary_classification():

            # we need squeeze to remove last (unnecessary) dim to avoid site-effects
            # temperature scaling: sinlge scalar
            if self.temperature_only:
                def logit_op(x, w, b): return torch.squeeze(torch.sum(torch.mul(x, w), dim=1))

            # platt scaling: one weight for each feature given
            else:
                weights = torch.reshape(weights, (-1, 1))
                def logit_op(x, w, b): return torch.squeeze(torch.matmul(x, w) + b)

            # define as probabilistic output the sigmoid and a bernoulli distribution
            prob_op = torch.sigmoid
            dist_op = dist.Bernoulli

        else:

            # the op for calculating the logit is an element-wise multiplication
            # for vector scaling and to keep multinomial output
            def logit_op(x, w, b): return torch.mul(x, w) + b

            # define as probabilistic output the softmax and a categorical distribution
            def prob_op(logit): return torch.softmax(logit, dim=1)
            dist_op = dist.Categorical

        # the first dimension of the given input data is the "independent" sample dimension
        with pyro.plate("data", X.shape[0]):

            # calculate logit
            logit = logit_op(X, weights, bias)

            # if MLE, (slow) sampling is not necessary. However, this is needed for 'variational' and 'mcmc'
            if self.method in ['variational', 'mcmc']:
                probs = prob_op(logit)
                pyro.sample("obs", dist_op(probs=probs, validate_args=True), obs=y)

        return logit
