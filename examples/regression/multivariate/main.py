# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from netcal import manual_seed
from netcal.regression import GPNormal, GPBeta, GPCauchy, IsotonicRegression, VarianceScaling
from netcal.presentation import ReliabilityRegression, ReliabilityQCE
from netcal.metrics import NLL, QCE, PinballLoss, ENCE, UCE

from examples.regression.multivariate import generate, draw_single_distribution


def get_meanstd_from_cdf(t: np.ndarray, cdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate mean and stddev from given cumulative distribution. """

    delta_cdf_iso = np.diff(cdf, axis=0)  # (t-1, n, d)
    t_mid = (t[:-1, ...] + t[1:, ...]) / 2  # (t-1, n, d)
    mean = np.sum(t_mid * delta_cdf_iso, axis=0)  # (n, d)
    std = np.sqrt(np.sum(np.square(t_mid) * delta_cdf_iso, axis=0) - np.square(mean))  # (n, d)

    return mean, std


def fit_isotonic() -> Tuple[np.ndarray, np.ndarray]:
    """ Fit isotonic regression model """

    isotonic = IsotonicRegression()
    isotonic.fit((ymean, ystd), y, tensorboard=tensorboard)
    t_iso, s_iso, q_iso = isotonic.transform((ymean, ystd), t=t)

    return t_iso, q_iso


def fit_varscaling() -> Tuple[np.ndarray, np.ndarray]:
    """ Fit VarianceScaling recalibration model """

    varscaling = VarianceScaling()
    varscaling.fit((ymean, ystd), y, tensorboard=tensorboard)
    ystd_varscaling = varscaling.transform((ymean, ystd))

    # convert stddev to covariance matrix
    ymean_varscaling = ymean
    ycov_varscaling = torch.diag_embed(torch.from_numpy(ystd_varscaling) ** 2).numpy()

    return ymean_varscaling, ycov_varscaling


def fit_gpbeta() -> Tuple[np.ndarray, np.ndarray]:
    """ Fit GPBeta recalibration model """

    gpbeta = GPBeta(
        n_inducing_points=n_inducing_points,
        n_random_samples=n_random_samples,
        n_epochs=n_epochs_independent,
        use_cuda=use_cuda
    )
    gpbeta.fit((ymean, ystd), y, tensorboard=tensorboard)
    t_gpbeta, s_gpbeta, q_gpbeta = gpbeta.transform((ymean, ystd), t=t)

    return t_gpbeta, q_gpbeta


def fit_gpnormal_variance_only() -> Tuple[np.ndarray, np.ndarray]:
    """ Fit GPNormal model - variant 1: no input correlations, no output correlations """

    gpnormal = GPNormal(
        n_inducing_points=n_inducing_points,
        n_random_samples=n_random_samples,
        n_epochs=n_epochs_independent,
        use_cuda=use_cuda,
        correlations=False,
        name_prefix="gpnormal_fully_independent"
    )
    gpnormal.fit((ymean, ystd), y, tensorboard=tensorboard)
    ystd_gpnormal = gpnormal.transform((ymean, ystd))

    # convert stddev to covariance matrix
    ymean_gpnormal = ymean
    ycov_gpnormal = torch.diag_embed(torch.from_numpy(ystd_gpnormal) ** 2).numpy()

    return ymean_gpnormal, ycov_gpnormal


def fit_gpnormal_correlations(use_input_cov: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit GPNormal model.
    If use_input_cov is False: variant 2: no input correlations, model output correlations
    If use_input_cov is True:  variant 3: provide input correlations, model output correlations
    """

    var_in = ystd if not use_input_cov else ycov
    name_prefix = "gpnormal_without_input_cov" if not use_input_cov else "gpnormal_with_input_cov"

    gpnormal = GPNormal(
        n_inducing_points=n_inducing_points,
        n_random_samples=n_random_samples,
        n_epochs=n_epochs_multivariate,
        use_cuda=use_cuda,
        correlations=True,
        name_prefix=name_prefix
    )
    gpnormal.fit((ymean, var_in), y, tensorboard=tensorboard)
    ycov_gpnormal = gpnormal.transform((ymean, var_in))
    ymean_gpnormal = ymean

    return ymean_gpnormal, ycov_gpnormal


def fit_gpcauchy() -> Tuple[np.ndarray, np.ndarray]:
    """ Fit GPCauchy model - no input correlations, no output correlations """

    gpcauchy = GPCauchy(
        n_inducing_points=n_inducing_points,
        n_random_samples=n_random_samples,
        n_epochs=n_epochs_independent,
        use_cuda=use_cuda,
        name_prefix="gpcauchy_fully_independent"
    )
    gpcauchy.fit((ymean, ystd), y, tensorboard=tensorboard)
    ysscale_gpcauchy = gpcauchy.transform((ymean, ystd))

    # convert stddev to covariance matrix
    ymode_gpcauchy = ymean

    return ymode_gpcauchy, ysscale_gpcauchy


if __name__ == '__main__':

    # parameter for regression calibration
    n_samples = 1000  # number of training/evaluation samples
    seed = 0
    bins = 20  # used for evaluation metrics

    # the following parameters are used during Gaussian process optimization
    n_inducing_points = 12  # number of inducing points used within the GP
    n_random_samples = 128  # number of random samples used for GP training/inference
    n_epochs_independent = 256  # number of epochs for the univariate (uncorrelated) GP methods
    n_epochs_multivariate = 256  # number of epochs for the multivariate (correlated) GP methods
    use_cuda = False  # or: "cuda:0", "cuda:1", etc.

    # number of samples to describe a cumulative distribution
    t = 256

    # quantile levels used for evaluation
    q = np.linspace(0.05, 0.95, 19)

    # initialize tensorboard logging
    tensorboard = SummaryWriter(log_dir="logs/netcal/regression/multivariate")
    hparam_dict = {
        'n_samples': n_samples,
        'seed': seed,
        'bins': bins,
        'n_inducing_points': n_inducing_points,
        'n_random_samples': n_random_samples,
        'n_epochs_independent': n_epochs_independent,
        'n_epochs_multivariate': n_epochs_multivariate,
        't': t,
        'q': torch.from_numpy(q),
    }

    # initialize metrics
    pinball = PinballLoss()
    ence = ENCE(bins=bins)
    uce = UCE(bins=bins)
    nll = NLL()
    qce = QCE(bins=bins, marginal=False)

    # initialize visualization tools
    reliability = ReliabilityRegression(quantiles=bins + 1)
    reliability_qce = ReliabilityQCE(bins=bins)
    figures = []

    # use fixed random seed
    with manual_seed(seed):

        # generate training/evaluation samples
        x, y, ymean, ycov = generate(n_samples)
        yvar = np.diagonal(ycov, axis1=-2, axis2=-1)
        ystd = np.sqrt(yvar)

        # run recalibration methods
        t_iso, q_iso = fit_isotonic()

        ymean_varscaling, ycov_varscaling = fit_varscaling()
        t_gpbeta, q_gpbeta = fit_gpbeta()
        ymean_gpnormal_independent, ycov_gpnormal_independent = fit_gpnormal_variance_only()
        ymean_gpnormal_without_cov, ycov_gpnormal_without_cov = fit_gpnormal_correlations(use_input_cov=False)
        ymean_gpnormal_with_cov, ycov_gpnormal_with_cov = fit_gpnormal_correlations(use_input_cov=True)
        ymode_cauchy, yscale_cauchy = fit_gpcauchy()

    # store all methods in a single dict
    methods = {
        "Uncalibrated": {"mean": ymean, "cov": ycov},
        "IsotonicRegression": {"t": t_iso, "cdf": q_iso},
        "VarianceScaling": {"mean": ymean_varscaling, "cov": ycov_varscaling},
        "GPBeta": {"t": t_gpbeta, "cdf": q_gpbeta},
        "GPNormal (independent)": {"mean": ymean_gpnormal_independent, "cov": ycov_gpnormal_independent},
        "GPNormal (without cov)": {"mean": ymean_gpnormal_without_cov, "cov": ycov_gpnormal_without_cov},
        "GPNormal (with cov)": {"mean": ymean_gpnormal_with_cov, "cov": ycov_gpnormal_with_cov},
        "GPCauchy (independent)": {"mode": ymode_cauchy, "scale": yscale_cauchy},
    }

    # draw a single distribution and store the result in tensorboard
    dist_fig = draw_single_distribution(methods)

    # iterate over methods and visualize reliability plots
    for method, values in methods.items():

        # use mean and stddev (normal distributions)
        # extract the stddev from the covariance matrices
        if "mean" in values:
            std = np.sqrt(np.diagonal(values["cov"], axis1=-2, axis2=-1))
            univariate = (values["mean"], std)
            multivariate = (values["mean"], values["cov"])
            kind = "meanstd"
        elif "t" in values:
            univariate = (values["t"], values["cdf"])
            multivariate = univariate
            kind = "cumulative"
        elif "scale" in values:
            univariate = (values["mode"], values["scale"])
            multivariate = univariate
            kind = "cauchy"
        else:
            cov = values["cov"]
            scale = np.diagonal(values["cov"], axis1=-2, axis2=-1)
            univariate = (values["mode"], scale)
            multivariate = (values["mode"], cov)
            kind = "cauchy"

        # -------------------------------------------------
        # metrics
        pinball_loss = np.mean(
            pinball.measure(univariate, y, q, kind=kind, reduction="none"),  # (q, n, d)
            axis=(0, 1)
        )  # (d,)

        uce_loss = uce.measure(univariate, y, kind=kind)  # (d,)
        ence_loss = ence.measure(univariate, y, kind=kind)  # (d,)

        nll_loss_independent = nll.measure(univariate, y, kind=kind, reduction="batchmean")  # (d,)
        qce_loss_independent = qce.measure(univariate, y, q, kind=kind, reduction="batchmean")  # (d,)
        nll_loss_joint = nll.measure(multivariate, y, kind=kind, reduction="batchmean")  # scalar or (d,)

        if not kind == "cauchy":
            qce_loss_joint = qce.measure(multivariate, y, q, kind=kind, reduction="batchmean")
        else:
            qce_loss_joint = np.nan

        if nll_loss_joint.size > 1:
            nll_loss_joint = np.sum(nll_loss_joint)

        if isinstance(qce_loss_joint, np.ndarray) and qce_loss_joint.size > 1:
            qce_loss_joint = np.mean(qce_loss_joint)

        metric_dict = {'nll_joint': nll_loss_joint, 'qce_joint': qce_loss_joint}
        metric_dict.update({'pinball_%02d' % dim: loss for dim, loss in enumerate(pinball_loss)})
        metric_dict.update({'uce_%02d' % dim: loss for dim, loss in enumerate(pinball_loss)})
        metric_dict.update({'ence_%02d' % dim: loss for dim, loss in enumerate(pinball_loss)})
        metric_dict.update({'nll_dim_%02d' % dim: loss for dim, loss in enumerate(pinball_loss)})
        metric_dict.update({'qce_dim_%02d' % dim: loss for dim, loss in enumerate(pinball_loss)})

        tensorboard.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=method)

        # -------------------------------------------------
        # visualization

        fig_reliability_independent = reliability.plot(univariate, y, kind=kind, title_suffix=method)
        fig_qce_independent = reliability_qce.plot(univariate, y, q, kind=kind, title_suffix=method)

        # save figure in tensorboard
        tensorboard.add_figure("multivariate/reliability/%s_independent" % method, fig_reliability_independent, close=False)
        tensorboard.add_figure("multivariate/reliability_qce/%s_independent" % method, fig_qce_independent, close=False)

        figures.append(fig_reliability_independent)
        figures.append(fig_qce_independent)

        if kind == "meanstd":
            # plot diagrams with full covariance matrices
            fig_reliability_joint = reliability.plot(multivariate, y, kind=kind, title_suffix=method)
            fig_qce_joint = reliability_qce.plot(multivariate, y, q, kind=kind, title_suffix=method)

            tensorboard.add_figure("multivariate/reliability/%s_joint" % method, fig_reliability_joint, close=False)
            tensorboard.add_figure("multivariate/reliability_qce/%s_joint" % method, fig_qce_joint, close=False)

            figures.append(fig_reliability_joint)
            figures.append(fig_qce_joint)

    plt.show()
