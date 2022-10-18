# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Dict, List
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from netcal.metrics import PinballLoss, ENCE, UCE, QCE, NLL


def draw_distributions(
        x: np.ndarray,
        y: np.ndarray,
        methods: Dict,
        quantile: float = 0.9,
        bins: int = 20,
        quantiles: np.ndarray = np.linspace(0.05, 0.95, 19),
) -> plt.Figure:
    """
    Draw the training/evaluation data distributions based on their estimated moments.

    Parameters
    ----------
    x : np.ndarray, shape: (n, [d])
        Input scores on the base x-axis.
    y : np.ndarray, shape: (n, [d])
        Ground-truth target scores on the y-axis.
    methods : dict
        Distributional estimates of y for several calibration methods.
    quantile : float, default: 0.9
        Width of the quantile boundaries used to visualize the base estimators.
    bins : int, default: 20
        Number of quantiles/bins used to estimate the ENCE, UCE and P-NEES metrics.
    quantiles : np.ndarray, shape: (q,)
        Quantile levels used for the evaluation of Pinball loss and Quantile Calibration Error (QCE).

    Returns
    -------
    matplotlib.pyplot.Figure
    """

    # visualize dataset
    axes = []
    fig = plt.figure(figsize=(4 * len(methods), 10))

    gs = fig.add_gridspec(2, 3)
    axes.append(fig.add_subplot(gs[:, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[0, 2]))
    axes.append(fig.add_subplot(gs[1, 1]))
    axes.append(fig.add_subplot(gs[1, 2]))

    zscore = norm.ppf(1 - ((1-quantile) / 2))
    bbox_probs = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # initialize miscalibration metrics
    nll = NLL()
    pinball = PinballLoss()
    uce = UCE(bins=bins)
    ence = ENCE(bins=bins)
    qce = QCE(bins=bins, marginal=False)

    # iterate over data / calibration methods
    for ax, (method, values), letter in zip(axes, methods.items(), ["a", "b", "c", "d", "e"]):

        # get mean and stddev estimates
        mean = values["mean"]
        std = values["std"]

        # plot ground truth and mean estimate of the base estimator
        ax.plot(x, y, "x", color="green", alpha=0.5)
        ax.plot(x, mean, "-", color="blue")

        # prefer 'raw' cumulative over mean/stddev moments if cumulative is given
        if "t" in values and "cdf" in values:

            t = values["t"]
            cdf = values["cdf"]

            # get bounds idx of cumulative
            lb_idx = np.argmax(cdf > (1. - quantile) / 2, axis=0)
            ub_idx = np.argmax(cdf > 1 - ((1. - quantile) / 2), axis=0)

            idx = np.arange(cdf.shape[1])
            lb = t[lb_idx, idx]
            ub = t[ub_idx, idx]

            ax.fill_between(x, lb, ub, alpha=0.9)

            # use cumulative to calculate the metrics
            nll_loss = nll.measure((values["t"], values["cdf"]), y, kind="cumulative")
            pinball_loss = pinball.measure((values["t"], values["cdf"]), y, q=quantiles, kind="cumulative")
            uce_loss = uce.measure((values["t"], values["cdf"]), y, kind="cumulative")
            ence_loss = ence.measure((values["t"], values["cdf"]), y, kind="cumulative")
            qce_loss = qce.measure((values["t"], values["cdf"]), y, q=quantiles, kind="cumulative")

        # otherwise, assert normal distribution
        else:

            # use prediction interval obtained by a normal distribution
            ax.fill_between(x, mean - zscore * std, mean + zscore * std, alpha=0.9)

            # use mean/stddev to calculate the metrics
            nll_loss = nll.measure((values["mean"], values["std"]), y, kind="meanstd")
            pinball_loss = pinball.measure((values["mean"], values["std"]), y, q=quantiles, kind="meanstd")
            uce_loss = uce.measure((values["mean"], values["std"]), y, kind="meanstd")
            ence_loss = ence.measure((values["mean"], values["std"]), y, kind="meanstd")
            qce_loss = qce.measure((values["mean"], values["std"]), y, q=quantiles, kind="meanstd")

        # display metrics in plot
        ax.text(
           0.05, 0.95,
           'NLL: %.4f\nQCE: %.4f\nPinnball: %.4f\nENCE: %.4f\nUCE: %.4f' % (nll_loss, qce_loss, pinball_loss, ence_loss, uce_loss),
           transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=bbox_probs
        )

        ax.set_title("%s) %s" % (letter, method))
        ax.grid(True)
        ax.set_xlim([-5., 5.])
        ax.set_ylim([-3.5, 4.5])

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        fig.tight_layout()

    return fig
