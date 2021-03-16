# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import logging
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from netcal import AbstractCalibration, TqdmHandler, dimensions, accepts


class BetaCalibrationDependent(AbstractCalibration):
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
    momentum : bool, default: False
        If True, momentum optimizer will be used instead of standard SciPy optimizer.
    max_iter : int, default: 1000
        Maximum iteration of optimizer.
    detection : bool, default: True
        IMPORTANT: this parameter is only for compatibility reasons. It MUST be set to True.
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).

    References
    ----------
    .. [1] Fabian Küppers, Jan Kronenberger, Amirhossein Shantia and Anselm Haselhoff:
       "Multivariate Confidence Calibration for Object Detection."
       The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

    .. [2] Libby, David L., and Melvin R. Novick:
       "Multivariate generalized beta distributions with applications to utility assessment"
       Journal of Educational Statistics 7.4, pp. 271-294, 1982
    """

    @accepts(bool, int, bool)
    def __init__(self, momentum: bool = True, max_iter: int = 1000, detection: bool = True):
        """
        Constructor.

        Parameters
        ----------
        momentum : bool, default: True
            If True, momentum optimizer will be used instead of standard SciPy optimizer.
        max_iter : int
            Maximum iteration of optimizer.
        detection : bool, default: True
            IMPORTANT: this parameter is only for compatibility reasons. It MUST be set to True.
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        """

        assert detection, "Classification mode (detection=False) is not supported for class BetaCalibrationDependent."
        super().__init__(detection=True, independent_probabilities=False)

        self.max_iter = max_iter
        self.momentum = momentum

        # amount of additional properties
        self._alpha_pos = np.empty(0)
        self._alpha_neg = np.empty(0)
        self._beta_pos = np.empty(0)
        self._beta_neg = np.empty(0)
        self._bias = 0.0

    def clear(self):
        """
        Clear model parameters.
        """

        super().clear()
        self._alpha_pos = np.empty(0)
        self._alpha_neg = np.empty(0)
        self._beta_pos = np.empty(0)
        self._beta_neg = np.empty(0)
        self._bias = 0.0

    @dimensions((1, 2), (1, 2), None)
    def fit(self, X: np.ndarray, y: np.ndarray, device: str = None) -> 'BetaCalibrationDependent':
        """
        Build dependent Beta Calibration model for multivariate Beta distributions.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).

        Returns
        -------
        BetaCalibrationDependent
            Instance of class :class:`BetaCalibrationDependent`.
        """

        X, y = super().fit(X, y)
        solver = 'SLSQP'
        #solver = 'L-BFGS-B'

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        # determine number of additional properties
        num_features = X.shape[1]

        # clip seperately due to numerical stability
        data_input = np.clip(X, self.epsilon, 1. - self.epsilon) / np.clip((1. - X), self.epsilon, 1. - self.epsilon)
        # --------------------------------------------------------------------------------

        # additional weights for: bias, confidences, shared parameter a0

        # momentum optimizer is slow but very accurate
        if self.momentum:
            self._alpha_pos, self._alpha_neg, self._beta_pos, self._beta_neg, self._bias = \
                self._momentum_optimization(data_input, y, device)

        # scipy optimizer is very fast. SLSQP has shown similar performance than momentum optimizer
        else:

            weights_per_feature = num_features+1
            num_weights = 4*weights_per_feature + 1
            theta_0 = np.random.uniform(low=1. + self.epsilon, high=2., size=num_weights)
            result = minimize(method=solver,
                              fun=self._loss_function, x0=theta_0,
                              args=(data_input, y, num_features),
                              bounds=[(self.epsilon, None),] * (num_weights-1) + [(None, None),])

            self._alpha_pos = result.x[:weights_per_feature]
            self._alpha_neg = result.x[weights_per_feature: 2*weights_per_feature]
            self._beta_pos = result.x[2*weights_per_feature:3*weights_per_feature]
            self._beta_neg = result.x[3*weights_per_feature:4*weights_per_feature]
            self._bias = result.x[-1]

        return self

    @dimensions((1, 2))
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with calibrated confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).
        """

        X = super().transform(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # transform input data to appropriate input format
        data_input = np.clip(X, self.epsilon, 1. - self.epsilon) / np.clip((1. - X), self.epsilon, 1. - self.epsilon)

        # get likelihood ratio which equals logit and calculate sigmoid
        likelihood_ratio = self._get_likelihood_ratio_beta(data_input, self._alpha_pos, self._alpha_neg,
                                                           self._beta_pos, self._beta_neg, self._bias)
        calibrated = self._sigmoid(likelihood_ratio)

        return calibrated

    @dimensions(2, 1, 1, 1, 1, None)
    def _get_likelihood_ratio_beta(self, data_input: np.ndarray,
                                   alpha_pos: np.ndarray, alpha_neg: np.ndarray,
                                   beta_pos: np.ndarray, beta_neg: np.ndarray, bias: float) -> np.ndarray:
        """
        Get likelihood ratio by given input data, weights and bias.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            Prepared input data.
        alpha_pos : np.ndarray, shape=(n_features+1,)
            Alpha values for positive distribution.
        alpha_neg : np.ndarray, shape=(n_features+1,)
            Alpha values for negative distribution.
        beta_pos : np.ndarray, shape=(n_features+1,)
            Beta values for positive distribution.
        beta_neg : np.ndarray, shape=(n_features+1,)
            Beta values for negative distribution.
        bias : float
            Bias of logit.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            Likelihood ratio as logits.
        """

        # lambdas are ratio between all betas and beta_0
        lambda_pos = beta_pos[1:] / beta_pos[0]
        lambda_neg = beta_neg[1:] / beta_neg[0]
        log_lambdas_upper = alpha_pos[1:] * np.log(lambda_pos) - alpha_neg[1:] * np.log(lambda_neg)

        # parameter differences
        differences_alpha_upper = alpha_pos[1:] - alpha_neg[1:]
        log_values_upper = np.log(data_input)

        # calculate upper part
        upper_part = np.sum(log_lambdas_upper + differences_alpha_upper * log_values_upper, axis=1)

        # start with summation of alphas for lower part of equation
        sum_alpha_pos = np.sum(alpha_pos)
        sum_alpha_neg = np.sum(alpha_neg)

        # calculate lower part
        log_sum_lower_pos = np.log(1. + np.sum(lambda_pos * data_input, axis=1))
        log_sum_lower_neg = np.log(1. + np.sum(lambda_neg * data_input, axis=1))
        lower_part = (sum_alpha_neg * log_sum_lower_neg) - (sum_alpha_pos * log_sum_lower_pos)

        # combine to likelihood ratio and return
        likelihood_ratio = bias + upper_part + lower_part

        return likelihood_ratio

    def _loss_function(self, weights: np.ndarray, data_input: np.ndarray,
                       y: np.ndarray, num_features: int) -> float:
        """
        Wrapper function for SciPy's loss. This is simply NLL-Loss.
        This wrapper is necessary because the first parameter is interpreted as the optimization parameter.

        Parameters
        ----------
        weights : np.ndarray, shape=(3,) or (2*num_features + 1,)
            Weights for logistic fit with dependencies.
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
        num_features : int
            Number of features for multivariate distribution.

        Returns
        -------
        float
            NLL-Loss
        """

        weights_per_feature = num_features+1
        alpha_pos = weights[:weights_per_feature]
        alpha_neg = weights[weights_per_feature: 2 * weights_per_feature]
        beta_pos = weights[2 * weights_per_feature:3 * weights_per_feature]
        beta_neg = weights[3 * weights_per_feature:4 * weights_per_feature]
        bias = weights[-1]

        # get likelihood ratio as logits and get NLL loss
        likelihood_ratio = self._get_likelihood_ratio_beta(data_input, alpha_pos, alpha_neg, beta_pos, beta_neg, bias)
        return self._nll_loss(logit=likelihood_ratio, ground_truth=y)

    @dimensions(2, 1, None)
    def _momentum_optimization(self, data_input: np.ndarray, y: np.ndarray, device: str = None) -> tuple:
        """
        Momentum optimization to find the global optimum of current parameter search.
        This method is slow but tends to find the global optimum.

        Parameters
        ----------
        data_input : np.ndarray, shape=(n_samples, n_features)
            NumPy 2-D array with data input.
        y : np.ndarray, shape=(n_samples,)
            NumPy array with ground truth labels as 1-D vector (binary).

        Returns
        -------
        tuple of length 5
            Estimated parameters.
        """

        # select device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        # initial learning rate, min delta for early stopping and patience
        # for early stopping (number of epochs without improvement)
        init_lr = 1e-3
        batch_size = 256

        # crierion is Binary Cross Entropy on logits (numerically more stable)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # create PyTorch dataset directly on GPU
        torch_data_input = torch.Tensor(data_input).to(device)
        torch_y = torch.Tensor(y).to(device)
        dataset = TensorDataset(torch_data_input, torch_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

        # init model and optimizer
        logistic_regression = DependentLR(n_features=data_input.shape[-1]).to(device)
        optimizer = torch.optim.Adam(logistic_regression.parameters(), lr=init_lr)

        # enable training mode of model
        logistic_regression.train()

        # set number of epochs
        num_batches = len(dataloader)
        num_epochs = int(np.ceil(self.max_iter / num_batches))

        best_loss = np.infty

        # use tqdm logger to pipe tqdm output to logger
        logger = logging.getLogger(__name__)
        tqdm_logger = TqdmHandler(logger=logger, level=logging.INFO)
        with tqdm(total=num_epochs * num_batches, file=tqdm_logger) as pbar:
            for epoch in range(num_epochs):

                # iterate over batches
                for i, (train_x, train_y) in enumerate(dataloader, start=(epoch*num_batches)):
                    logits = logistic_regression(train_x)
                    loss = criterion(logits, train_y)

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

        # extract weights from layer and clip to interval (0, inf]
        # this is necessary because negative values might be occur after last
        # optimization step - these values are not clipped then
        alpha_pos = np.clip(logistic_regression.alpha_pos.data.cpu().numpy(), self.epsilon, np.infty)
        alpha_neg = np.clip(logistic_regression.alpha_neg.data.cpu().numpy(), self.epsilon, np.infty)
        beta_pos = np.clip(logistic_regression.beta_pos.data.cpu().numpy(), self.epsilon, np.infty)
        beta_neg = np.clip(logistic_regression.beta_neg.data.cpu().numpy(), self.epsilon, np.infty)

        bias = logistic_regression.bias.data.cpu().numpy()

        return alpha_pos, alpha_neg, beta_pos, beta_neg, bias


class DependentLR(nn.Module):
    """
    PyTorch nn.Module for dependent logistic regression of multivariate beta distribution.
    """

    epsilon = np.finfo(np.float).eps

    @accepts(int)
    def __init__(self, n_features: int):
        """
        Constructor.
        """

        super().__init__()
        n_dims = n_features + 1

        self.alpha_pos = nn.Parameter(torch.empty(n_dims).uniform_(1. + self.epsilon, 2.))
        self.alpha_neg = nn.Parameter(torch.empty(n_dims).uniform_(1. + self.epsilon, 2.))
        self.beta_pos = nn.Parameter(torch.empty(n_dims).uniform_(1. + self.epsilon, 2.))
        self.beta_neg = nn.Parameter(torch.empty(n_dims).uniform_(1. + self.epsilon, 2.))
        self.bias = nn.Parameter(torch.empty(1).uniform_(-1., 1.))

    def forward(self, x: torch.Tensor):
        """
        Function for forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
            Logits of multivariate beta logistic regression.
        """

        # clip values to range (0, inf]
        alpha_pos = torch.clamp(self.alpha_pos, self.epsilon, np.infty)
        alpha_neg = torch.clamp(self.alpha_neg, self.epsilon, np.infty)
        beta_pos = torch.clamp(self.beta_pos, self.epsilon, np.infty)
        beta_neg = torch.clamp(self.beta_neg, self.epsilon, np.infty)

        # lambdas are ratio between all betas and beta_0
        lambda_pos = beta_pos[1:] / beta_pos[0]
        lambda_neg = beta_neg[1:] / beta_neg[0]
        log_lambdas_upper = alpha_pos[1:] * torch.log(lambda_pos) - alpha_neg[1:] * torch.log(lambda_neg)

        # parameter differences
        differences_alpha_upper = alpha_pos[1:] - alpha_neg[1:]
        log_values_upper = torch.log(x)

        # calculate upper part
        upper_part = torch.sum(log_lambdas_upper + (differences_alpha_upper * log_values_upper), dim=1)

        # start with summation of alphas for lower part of equation
        sum_alpha_pos = torch.sum(alpha_pos)
        sum_alpha_neg = torch.sum(alpha_neg)

        # calculate lower part
        log_sum_lower_pos = torch.log(1. + torch.sum(lambda_pos * x, dim=1))
        log_sum_lower_neg = torch.log(1. + torch.sum(lambda_neg * x, dim=1))
        lower_part = (sum_alpha_neg * log_sum_lower_neg) - (sum_alpha_pos * log_sum_lower_pos)

        # combine both parts and bias to logits
        logits = self.bias + upper_part + lower_part

        return logits
