# Copyright (C) 2021-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import pyro.distributions as dist


class BetaLikelihood(dist.TorchDistribution):
    """
    Beta likelihood used to implement the basic functionality of the :class:`netcal.regression.gp.GPBeta`
    recalibration method.
    This function serves the standard beta calibration mapping for the cumulative distribution function (CDF) as well
    as the derived beta link function for the recalibration of the probability density function (PDF).
    Since the likelihood of the Gaussian process is calculated during model training using the PDF, this method
    is required not only for the inference but also for the training of the :class:`netcal.regression.gp.GPBeta` method
    [1]_.

    **Mathematical background:** From the docstring of :class:`netcal.regression.gp.GPBeta`, we know that the GP-Beta
    utilizes the beta calibration method [2]_ from confidence calibration (cf. :class:`netcal.scaling.BetaCalibration`)
    for the recalibration of the CDF.
    Let :math:`f_Y(y)` denote the uncalibrated probability density function (PDF), targeting the probability
    distribution for :math:`Y`. Let :math:`\\tau_y \\in [0, 1]` denote a certain quantile on the uncalibrated CDF which
    is denoted by :math:`\\tau_y = F_Y(y)`.
    Furthermore, let :math:`g_Y(y)` and :math:`G_Y(y)` denote the recalibrated PDF and CDF, respectively.
    The Beta calibration function :math:`\\mathbf{c}_\\beta(\\tau_y)` known from [2]_ is given by

    .. math::
        \\mathbf{c}_\\beta(\\tau_y) &= \\phi\\big( a \\log(\\tau_y) - b \\log(1-\\tau_y) + c \\big) \\\\
        &= \\phi\\big( z(\\tau_y) \\big) ,

    with recalibration parameters :math:`a,b \\in \\mathbb{R}_{>0}` and :math:`c \\in \\mathbb{R}`, and
    :math:`\\phi(\\cdot)` as the sigmoid function [2]_.
    In this case, we denote the logit of the beta calibration function :math:`\\mathbf{c}_\\beta(\\tau_y)` by

    .. math::
        z(\\tau_y) = a \\log(\\tau_y) - b \\log(1-\\tau_y) + c .

    The beta calibration method serves as a mapping from the uncalibrated CDF to the calibrated one, so that

    .. math::
        G_Y(y) = \\mathbf{c}_\\beta\\big( F_Y(y) \\big)

    holds. The PDF is recalibrated using the beta link function :math:`\\mathbf{r}_\\beta(\\tau_y)` [1]_ by

    .. math::
        g_Y(y) = \\frac{\\partial \\mathbf{c}_\\beta}{\\partial y}
        = \\frac{\\partial \\mathbf{c}_\\beta}{\\partial \\tau_y} \\frac{\\partial \\tau_y}{\\partial y}
        = \\mathbf{r}_\\beta(\\tau_y) f_Y(y) ,

    where the beta link function is given by

    .. math::
        \\mathbf{r}_\\beta(\\tau_y) = \\Bigg(\\frac{a}{\\tau_y} + \\frac{b}{1-\\tau_y} \\Bigg)
        \\mathbf{c}_\\beta(\\tau_y) \\big(1 - \\mathbf{c}_\\beta(\\tau_y)\\big) .

    The recalibration parameters :math:`a,b` and :math:`c` are not directly infered by the GP but rather use the
    underlying function parameters :math:`w_a, w_b, w_c \\in \\mathbb{R}`, so that

    .. math::
       a &= \\exp(\\gamma_a^{-1} w_a + \\delta_a) \\\\
       b &= \\exp(\\gamma_b^{-1} w_b + \\delta_b) \\\\
       c &= \\gamma_c^{-1} w_c + \\delta_c

    are given as the beta distribution parameters to guarantee :math:`a, b > 0`.
    During optimization, we need the log of the link function. We use a numerical more stable version:

    .. math::
       \\log\\big(\\mathbf{r}_\\beta(\\tau_y)\\big) = \\log \\Bigg(\\frac{a}{\\tau_y} + \\frac{b}{1-\\tau_y} \\Bigg) +
       \\log \\Big( \\mathbf{c}_\\beta(\\tau_y) \\big(1 - \\mathbf{c}_\\beta(\\tau_y)\\big) \\Big)

    The first part can be rewritten as:

    .. math::
       & \\log \\Bigg(\\frac{a}{\\tau_y} + \\frac{b}{1-\\tau_y} \\Bigg) \\\\
       &= \\log \\Bigg( \\exp \\Big( \\log(a) - \\log(\\tau_y) \\Big) + \\exp \\Big( \\log(b) - \\log(1-\\tau_y) \\Big) \\Bigg) \\\\
       &= \\log \\Bigg( \\exp \\Big( \\gamma_a^{-1} w_a + \\delta_a - \\log(\\tau_y) \\Big) + \\exp \\Big( \\gamma_b^{-1} w_b + \\delta_b - \\log(1-\\tau_y) \\Big) \\Bigg) \\\\
       &= \\text{logsumexp} \\Bigg[ \\Big( \\gamma_a^{-1} w_a + \\delta_a - \\log(\\tau_y) \\Big), \\Big( \\gamma_b^{-1} w_b + \\delta_b - \\log(1-\\tau_y) \\Big) \\Bigg]

    where :math:`\\text{logsumexp}` is a numerical stable version of :math:`\\log \\Big( \\sum_x \\exp(x) \\Big)` in PyTorch.
    In addition, we can also simplify the expression for the log of the sigmoid functions that are used within
    the computation of :math:`\\mathbf{c}_\\beta(\\tau_y)` by


    .. math::
       \\log(\\phi(x)) &= -\\log \\Big( \\exp(-x) + 1 \\Big), \\\\
       \\log(1-\\phi(x)) &= -\\log \\Big( \\exp(x) + 1 \\Big)

    which can also be expressed in terms of PyTorch's numerical more stable :math:`\\text{softplus}`
    function :math:`\\log \\Big( \\exp(x) + 1 \\Big)`. Thus, the log of the link function can be expressed as

    .. math::
       \\log(r_\\beta(\\tau_y)) = &\\text{logsumexp} \\Bigg[ \\Big( \\gamma_a^{-1} w_a + \\delta_a - \\log(\\tau_y) \\Big),
       \\Big( \\gamma_b^{-1} w_b + \\delta_b - \\log(1-\\tau_y) \\Big) \\Bigg] \\\\
       &- \\Big(\\text{softplus}\\big(-z(x)\\big) + \\text{softplus}\\big(z(x)\\big)\\Big)

    Parameters
    ----------
    loc: torch.Tensor, shape: ([[1], 1], n, d)
        Mean vector of the uncalibrated (Gaussian) probability distribution with n samples and d dimensions.
        This class also supports broadcasting of the calculations to t sampling points (defining the non-parametric
        PDF/CDF, first dim) and r random samples (second dim).
    logvar: torch.Tensor, shape: ([[1], 1], n, d)
        Log of the variance vector of the uncalibrated (Gaussian) probability distribution with n samples and d
        dimensions. This class also supports broadcasting of the calculations to t sampling points (defining the
        non-parametric PDF/CDF, first dim) and r random samples (second dim).
    parameters: torch.Tensor, shape: ([[1], r], n, p*d)
        Rescaling parameters of the beta link function **before applying the exponential** (for parameters :math:`a, b`)
        with n samples d dimensions, r random samples and p as the number of parameters (p=3 in this case).

    References
    ----------
    .. [1] Hao Song, Tom Diethe, Meelis Kull and Peter Flach:
       "Distribution calibration for regression."
       International Conference on Machine Learning (pp. 5897-5906), 2019.
       `Get source online <http://proceedings.mlr.press/v97/song19a/song19a.pdf>`__

    .. [2] Kull, Meelis, Telmo Silva Filho, and Peter Flach:
       "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
       Artificial Intelligence and Statistics, PMLR 54:623-631, 2017
       `Get source online <http://proceedings.mlr.press/v54/kull17a/kull17a.pdf>`__
    """

    def __init__(self, loc: torch.Tensor, logvar: torch.Tensor, parameters: torch.Tensor):
        """ Constructor. For detailed parameter description, see class docs. """

        # call super constructor with right batch_shape (necessary for Pyro) and save parameters as member variable
        super().__init__(batch_shape=parameters.shape[:-1] + loc.shape[-1:], validate_args=False)
        self.parameters = parameters

        # recover the variance and initialize a normal distribution that reflects the uncalibrated distribution
        var = torch.exp(logvar)
        self._normal = dist.Normal(loc, torch.sqrt(var))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-density at the points given by 'value' using the underlying uncalibrated distribution
        :math:`f_Y(y)` that is rescaled by the calibration parameters.
        This function implements the beta link function :math:`\\mathbf{r}_\\beta(\\tau_y)` and provides the
        log-likelihood for the recalibrated PDF

        .. math::
            \\log\\big(g_Y(y)\\big) = \\log\\big(\\mathbf{r}_\\beta(\\tau_y) f_Y(y) \\big) .

        Parameters
        ----------
        value : torch.Tensor, shape: ([[t], 1], n, d)
            Points at which to calculate the log-likelihood with n samples and d dimensions.
            The dimension t represents the points that define the non-parametric PDF/CDF on the x-axis.
            The intermediate (second) dim is required when using multiple stochastic forward passes for
            the rescaling parameters.

        Returns
        -------
        torch.Tensor, shape: ([[t], r], n, d)
            Log-likelihood of the recalibrated non-parametric distribution with
            t points that define the non-parametric PDF/CDF on the x-axis, r stochastic forward passes for the rescaling
            parameters, n samples and d dimensions.
        """

        # get cumulative and log_density for input
        with torch.no_grad():
            cumulative = self._normal.cdf(value)  # ([[t], 1], n, d)
            log_density = self._normal.log_prob(value)  # ([[t], 1], n, d)

            # prevent inf or NaN during log for cumulatives that ran into saturation {0, 1}
            torch.clamp(cumulative, min=torch.finfo(cumulative.dtype).eps, max=1.-torch.finfo(cumulative.dtype).eps, out=cumulative)

        param_a = self.parameters[..., 0::3]  # ([[1], r], n, d)
        param_b = self.parameters[..., 1::3]  # ([[1], r], n, d)

        # numerical stable variant of log
        log_derivative_logit = torch.logsumexp(
            torch.stack((
                param_a - torch.log(cumulative),
                param_b - torch.log(1. - cumulative),
            ), dim=-1),  # ([[t], r], n, d, 2)
            dim=-1
        )  # ([[t], r], n, d)

        # get logit of calibrated cumulative
        logit = self._get_beta_logit(cumulative)

        # log(sigmoid(x)) is -softplus(-x)
        # log(1-sigmoid(x)) is -softplus(x)
        log_derivative_sigmoid = -F.softplus(-logit) - F.softplus(logit)  # ([[t], r], n, d)

        log_link = log_derivative_logit + log_derivative_sigmoid  # ([[t], r], n, d)

        # broadcast log_density to sample dimension
        linklikelihood = log_link + log_density  # ([[t], r], n, d)

        return linklikelihood

    def cdf(self, t: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cumulative distribution function (CDF) at the points given by 't' using
        the underlying uncalibrated cumulative :math:`F_Y(y)` that is rescaled by the calibration parameters.
        This function implements the beta calibration function :math:`\\mathbf{c}_\\beta(\\tau_y)` and provides the
        recalibrated CDF

        .. math::
            G_Y(y) = \\mathbf{c}_\\beta\\big(F_y(y)\\big).

        Furthermore, this method provides the sampling points to properly represent the x-values of the recalibrated
        CDF. To obtain the relevant points (where the CDF is >>0 and <<1), we use an iterative approach to approximate
        the most relevant x-values of the CDF.

        Parameters
        ----------
        t : torch.Tensor, shape: ([[t], r], n, d)
            Points at which to calculate the CDF with n samples and d dimensions.
            The dimension t represents the points that define the non-parametric PDF/CDF on the x-axis.
            The dimension r is required when using multiple stochastic forward passes for
            the rescaling parameters.

        Returns
        -------
        Tuple of torch.Tensor, both of shape: ([[t], r], n, d)
            X- and y-values of the recalibrated cumulative distribution function (CDF) with
            t points that define the non-parametric PDF/CDF on the x-axis, r stochastic forward passes for the rescaling
            parameters, n samples and d dimensions.
        """

        # target quantile determines the points at which the CDF is not relevant any more
        target_quantile = 1e-7

        # max_iter is the maximum amount of iterations used to refine the sampling points
        max_iter = 5

        # if t is int, use t sampling points to describe the output distribution
        if isinstance(t, int):

            # get base boundaries by the inverse cumulative function
            lower_bound_0 = self._normal.icdf(torch.tensor(target_quantile))  # ([[1], 1], n, d)
            upper_bound_0 = self._normal.icdf(torch.tensor(1. - target_quantile))  # ([[1], 1], n, d)

            # use the base quantile width 4 times (in each direction) to rescale the boundaries
            width_0 = upper_bound_0 - lower_bound_0
            lower_bound = lower_bound_0 - 4 * width_0
            upper_bound = upper_bound_0 + 4 * width_0

            # generate initial guess for the sampling points
            sampling_points, cumulative = self._generate_sampling_points(
                t, lower_bound.cpu().numpy(), upper_bound.cpu().numpy()
            )

            # use iterative approach to refine the sampling points
            for it in range(max_iter + 1):
                logit = self._get_beta_logit(cumulative)
                calibrated = torch.sigmoid(logit)

                # reached end of refinement loop
                if it == max_iter:
                    break

                # copy back to numpy as we use the 'flip' function in the following
                # in contrast to PyTorch, NumPy returns a view instead of making a copy of the data
                calibrated = calibrated.cpu().numpy()  # # ([[t], r], n, d)

                # get the indices where the desired relevance level is reached
                # move the indices to the outer point (useful especially for very low resolutions) by decreasing
                # lb - 1 and increasing ub + 1
                lb_idx = np.argmax(calibrated >= target_quantile, axis=0) - 1  # ([r], n, d)
                ub_idx = t - np.argmax(np.flip(calibrated, axis=0) < (1 - target_quantile), axis=0)  # ([r], n, d)

                # depending on the random samples, use the minimum/maximum boundary idx
                lb_idx = np.min(lb_idx, axis=0, keepdims=True)  # (1, n, d)
                ub_idx = np.max(ub_idx, axis=0, keepdims=True)  # (1, n, d)

                # avoid out-of-bounds indices
                lb_idx[lb_idx < 0] = 0
                ub_idx[ub_idx >= t] = t - 1

                # take indices along first axis to get the right x-scores for the cumulative
                lower_bound = np.take_along_axis(
                    sampling_points.numpy(),
                    indices=lb_idx[None, ...],
                    axis=0,
                )  # ([[1], r], n, d)

                upper_bound = np.take_along_axis(
                    sampling_points.numpy(),
                    indices=ub_idx[None, ...],
                    axis=0,
                )  # ([[1], r], n, d)

                # generate new set of sampling points
                sampling_points, cumulative = self._generate_sampling_points(t, lower_bound, upper_bound)

        # if t is np.ndarray, use this points as the base for the calibrated output distribution
        elif isinstance(t, torch.Tensor):

            sampling_points = t

            # get cumulative and log_density for input
            with torch.no_grad():
                cumulative = self._normal.cdf(sampling_points)  # ([[t], 1], n, d)
                torch.clamp(cumulative, min=torch.finfo(cumulative.dtype).eps,
                            max=1. - torch.finfo(cumulative.dtype).eps, out=cumulative)

            logit = self._get_beta_logit(cumulative)
            calibrated = torch.sigmoid(logit)

        else:
            raise AttributeError("Parameter \'t\' must be either of type int or np.ndarray.")

        # finally, move sampling points to the same device
        sampling_points = sampling_points.to(device=calibrated.device)
        return sampling_points, calibrated

    def _generate_sampling_points(
            self,
            t: int,
            lower_bound: np.ndarray,
            upper_bound: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the sampling points between lower and upper bound.
        This function is iteratively called to refine the set of sampling points to get a meaningful representation
        of the recalibrated non-parametric distribution.

        Parameters
        ----------
        t : int
            Number of sampling points that define a non-parametric PDF/CDF on the x-axis.
        lower_bound : np.ndarray, shape: ([[1], r], n, d)
            Lower bound of the sampling points with r random samples, n data points and d dimensions.
        upper_bound : np.ndarray, shape: ([[1], r], n, d)
            Upper bound of the sampling points with r random samples, n data points and d dimensions.

        Returns
        -------
        Tuple of torch.Tensor, both of shape ([[t], r], n, d)
            X- and y-values of the recalibrated cumulative distribution function (CDF) with
            t points that define the non-parametric PDF/CDF on the x-axis, r stochastic forward passes for the rescaling
            parameters, n samples and d dimensions.
        """

        # use NumPy's linspace as it is capable of arrays as start- and endpoints.
        sampling_points = torch.from_numpy(
            np.linspace(lower_bound, upper_bound, t, axis=0)
        )  # ([[t, 1], r], n, d)

        # remove the unnecessary additional dimension
        sampling_points = torch.squeeze(
            sampling_points,
            dim=1
        ) if sampling_points.ndim == 5 else sampling_points  # ([[t], r], n, d)

        # bring sampling points to the target precision
        sampling_points = sampling_points.to(dtype=self._normal.loc.dtype)

        # cumulative is required to examine the y-scores of the CDF to determine the new boundaries
        cumulative = self._normal.cdf(
            sampling_points.to(device=self._normal.loc.device)
        )  # ([[t], r], n, d)

        # clamp cumulative to be in open interval (0, 1)
        torch.clamp(
            cumulative,
            min=torch.finfo(cumulative.dtype).eps,
            max=1. - torch.finfo(cumulative.dtype).eps,
            out=cumulative
        )

        return sampling_points, cumulative

    def _get_beta_logit(self, cumulative: torch.Tensor) -> torch.Tensor:
        """
        Calculate the logit of the beta calibration function at the points given by "cumulative" using the
        recalibration parameters.
        The logit :math:`z(\\tau_y)` of the beta calibration function for a quantile :math:`\\tau_y \\in [0, 1]` is
        defined by

        .. math::
            z(\\tau_y) = a \\log(\\tau_y) - b \\log(1-\\tau_y) + c .

        Parameters
        ----------
        cumulative : torch.Tensor, shape: ([[t], 1], n, d)
            Y-values of the (uncalibrated) CDF in :math:`[0, 1]` with t points that define the non-parametric PDF/CDF,
            n samples and d dimensions.

        Returns
        -------
        torch.Tensor, shape: ([[t], r], n, d)
            Logit of the beta calibration function with t points that define the non-parametric PDF/CDF,
            r stochastic forward passes for the rescaling parameters, n samples and d dimensions.
        """

        param_a = self.parameters[..., 0::3]  # ([[1], r], n, d)
        param_b = self.parameters[..., 1::3]  # ([[1], r], n, d)
        param_c = self.parameters[..., 2::3]  # ([[1], r], n, d)

        # call exponential function on the first two parameters
        parameters = torch.stack((
            param_a.exp(),
            param_b.exp(),
            param_c),
            dim=-1
        )  # ([[1], r], n, d, 3)

        # construct input array
        input = torch.stack(
            (torch.log(cumulative), -torch.log(1. - cumulative), torch.ones_like(cumulative)),
            dim=-1
        )  # ([[t], 1], n, d, 3)

        # get calibrated cumulative
        logit = torch.sum(parameters * input, dim=-1)  # ([[t], r], n, d)

        return logit
