# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from netcal import manual_seed, cumulative_moments
from netcal.regression import IsotonicRegression, VarianceScaling, GPBeta, GPNormal, GPCauchy
from netcal.metrics import NLL, ENCE, UCE, QCE, PinballLoss
from netcal.presentation import ReliabilityRegression

from examples.regression.artificial import generate_variance_dependent, generate_mean_dependent, draw_distributions


if __name__ == '__main__':

    # parameter for regression calibration
    mean_dependent = True  # set to true to sample data with mean dependency
    n_samples = 2000  # number of training/evaluation samples
    seed = 0
    bins = 20  # used for evaluation metrics
    quantiles = np.linspace(0.05, 0.95, 19)  # quantile levels, used for evaluation metrics

    # the following parameters are used during Gaussian process optimization
    n_inducing_points = 12  # number of inducing points used within the GP
    n_random_samples = 256  # number of random samples used for GP training/inference
    n_epochs = 200  # optimization epochs
    use_cuda = False  # or: "cuda:0", "cuda:1", etc.

    # number of samples to describe a cumulative distribution
    t = 256

    # initialize tensorboard logging
    tensorboard = SummaryWriter("logs/netcal/regression/artificial")
    reliability = ReliabilityRegression(quantiles=bins+1)
    figures = []

    # use fixed random seed
    with manual_seed(seed):

        # sample from artificial distributions
        if mean_dependent:
            x, y, ymean, ystd = generate_mean_dependent(n_samples=n_samples)
        else:
            x, y, ymean, ystd = generate_variance_dependent(n_samples=n_samples)

        # -------------------------------------------------
        # fit isotonic regression model
        isotonic = IsotonicRegression()
        isotonic.fit((ymean, ystd), y, tensorboard=tensorboard)
        t_iso, s_iso, q_iso = isotonic.transform((ymean, ystd), t=t)

        # squeeze t, pdf and cdf and get distribution moments
        t_iso, s_iso, q_iso = t_iso[..., 0], s_iso[..., 0], q_iso[..., 0]
        ymean_iso, yvar_iso = cumulative_moments(t_iso, q_iso)
        ystd_iso = np.sqrt(yvar_iso)

        # -------------------------------------------------
        # fit VarianceScaling model
        varscaling = VarianceScaling()
        varscaling.fit((ymean, ystd), y, tensorboard=tensorboard)
        ystd_varscaling = varscaling.transform((ymean, ystd))

        # squeeze calibrated stddev
        ymean_varscaling = ymean
        ystd_varscaling = ystd_varscaling[..., 0]

        # # -------------------------------------------------
        # fit GPBeta model
        gpbeta = GPBeta(n_inducing_points=n_inducing_points, n_random_samples=n_random_samples, n_epochs=n_epochs, use_cuda=use_cuda)
        gpbeta.fit((ymean, ystd), y, tensorboard=tensorboard)
        t_gpbeta, s_gpbeta, q_gpbeta = gpbeta.transform((ymean, ystd), t=t)

        # squeeze t, pdf and cdf and get distribution moments
        t_gpbeta, s_gpbeta, q_gpbeta = t_gpbeta[..., 0], s_gpbeta[..., 0], q_gpbeta[..., 0]
        ymean_gpbeta, yvar_gpbeta = cumulative_moments(t_gpbeta, q_gpbeta)
        ystd_gpbeta = np.sqrt(yvar_gpbeta)

        # -------------------------------------------------
        # fit GPNormal model
        gpnormal = GPNormal(n_inducing_points=n_inducing_points, n_random_samples=n_random_samples, n_epochs=n_epochs, use_cuda=use_cuda)
        gpnormal.fit((ymean, ystd), y, tensorboard=tensorboard)
        ystd_gpnormal = gpnormal.transform((ymean, ystd))

        # squeeze calibrated stddev
        ymean_gpnormal = ymean
        ystd_gpnormal = ystd_gpnormal[..., 0]

        # -------------------------------------------------
        # fit GPCauchy model
        gpcauchy = GPCauchy(n_inducing_points=n_inducing_points, n_random_samples=n_random_samples, n_epochs=n_epochs, use_cuda=use_cuda)
        gpcauchy.fit((ymean, ystd), y, tensorboard=tensorboard)
        ysscale_gpcauchy = gpcauchy.transform((ymean, ystd))

        ymode_gpcauchy = ymean
        yscale_gpcauchy = ysscale_gpcauchy[..., 0]

    # store all methods in a single dict
    methods = {
        "Uncalibrated": {"mean": ymean, "std": ystd},
        "Isotonic Regression": {"t": t_iso, "cdf": q_iso, "mean": ymean_iso, "std": ystd_iso},
        "VarianceScaling": {"mean": ymean_varscaling, "std": ystd_varscaling},
        "GPBeta": {"t": t_gpbeta, "cdf": q_gpbeta, "mean": ymean_gpbeta, "std": ystd_gpbeta},
        "GPNormal": {"mean": ymean_gpnormal, "std": ystd_gpnormal},
        "GPCauchy": {"mode": ymode_gpcauchy, "scale": yscale_gpcauchy},
    }

    # draw data distribution and log to tensorboard
    dist_fig = draw_distributions(x, y, methods, 0.9, bins=bins, quantiles=quantiles)

    nll = NLL()
    pinball = PinballLoss()
    uce = UCE(bins=bins)
    ence = ENCE(bins=bins)
    qce = QCE(bins=bins, marginal=False)

    print("{method: <25}NLL            Pinball        UCE            ENCE           QCE".format(method="Method"))

    # iterate over methods and measure miscalibration/visualize reliability plots
    for method, values in methods.items():

        if "t" in values and "cdf" in values:
            univariate = (values["t"], values["cdf"])
            kind = "cumulative"
        elif "mode" in values and "scale" in values:
            univariate = (values["mode"], values["scale"])
            kind = "cauchy"
        else:
            univariate = (values["mean"], values["std"])
            kind = "meanstd"

        nll_loss = nll.measure(univariate, y, kind=kind)
        pinball_loss = pinball.measure(univariate, y, q=quantiles, kind=kind)
        uce_loss = uce.measure(univariate, y, kind=kind)
        ence_loss = ence.measure(univariate, y, kind=kind)
        qce_loss = qce.measure(univariate, y, q=quantiles, kind=kind)

        fig = reliability.plot(univariate, y, kind=kind, title_suffix=method)

        hparam_dict = {"method": method, "bins": bins}
        metric_dict = {"nll": nll_loss, "pinball": pinball_loss, "uce": uce_loss, "ence": ence_loss, "qce": qce_loss}

        # save figure in tensorboard
        tensorboard.add_figure("artificial/reliability/%s" % method, fig, close=False)
        tensorboard.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, run_name=method)
        figures.append(fig)

        print(
            "{method: <25}{nll:10.4f}     {pinball:10.4f}     {uce:10.4f}     {ence:10.4f}     {qce:10.4f}".format(
                method=method,
                nll=float(nll_loss),
                pinball=float(pinball_loss),
                uce=float(uce_loss),
                ence=float(ence_loss),
                qce=float(qce_loss)
            )
        )

    plt.show()
