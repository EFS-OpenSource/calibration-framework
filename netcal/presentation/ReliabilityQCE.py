# Copyright (C) 2021-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union, Iterable, Tuple, List
import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib

from netcal.metrics.regression import QCE


class ReliabilityQCE(object):
    """
    Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    for probabilistic regression models.
    See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    as multiple independent Gaussians.
    This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    with covariance matrices.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used by the C-QCE binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).

    References
    ----------
    .. [1] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__
    """

    eps = np.finfo(np.float32).eps

    def __init__(self, bins: Union[int, Iterable[float], np.ndarray] = 10):
        """ Constructor. For detailed parameter documentation view classdocs. """

        self.qce = QCE(bins=bins, marginal=False)

    def plot(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            q: Union[float, Iterable[float], np.ndarray],
            *,
            kind: str = 'meanstd',
            range_: List[Tuple[float, float]] = None,
            filename: str = None,
            tikz: bool = False,
            title_suffix: str = None,
            **save_args
    ) -> Union[plt.Figure, str]:
        """
        Visualizes the C-QCE as a bar chart either for multiple univariate data (if standard deviations are given as
        input) or for a joint multivariate distribution (if covariance matrices are given as input).
        See parameter "kind" for a detailed description of the input format.

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        q : np.ndarray of shape (q,)
            Quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                       first array as mean and the second one as the according stddev predictions for d dimensions.
                       If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                       for each sample. In this case, the NLL is calculated for multivariate distributions.
                       If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                       inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                       case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                          for sample n (and optionally d dimensions).
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the standard deviation for each dimension separately.
            For example, if input data is given with only a few samples having high standard deviations,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the C-QCE either as Matplotlib figure or as string with tikz code.
        """

        # measure QCE and return a miscalibration map
        _, qce_map, num_samples_hist = self.qce.measure(
            X=X,
            y=y,
            q=q,
            kind=kind,
            reduction="none",
            range_=range_,
            return_map=True,
            return_num_samples=True
        )  # (q, b)

        # get number of dimensions
        ndims = len(qce_map)

        # catch if bin_edges is None
        assert len(self.qce._bin_edges) != 0, "Fatal error: could not compute bin_edges for ReliabilityQCE."

        # initialize plot and create an own chart for each dimension
        fig, axes = plt.subplots(nrows=2, ncols=ndims, figsize=(7 * ndims, 6), squeeze=False)
        for dim in range(ndims):

            # convert absolute number of samples to relative amount
            n_samples_hist = np.divide(num_samples_hist[dim], np.sum(num_samples_hist[dim]))

            # compute mean over all quantiles as well as mean over all bins separately
            mean_over_quantiles = np.mean(qce_map[dim], axis=0)  # (b,)

            # get binning boundaries for actual dimension
            bounds = self.qce._bin_edges[dim]  # (b+1)

            for ax, metric, title, ylabel in zip(
                [axes[0][dim], axes[1][dim]],
                [n_samples_hist, mean_over_quantiles],
                ["Sample Histogram", "QCE mean over quantiles"],
                ["% of Samples", "Quantile Calibration Error (QCE)"],
            ):

                # draw bar chart with given edges and metrics
                ax.bar(bounds[:-1], height=metric, width=np.diff(bounds), align='edge', edgecolor='black')

                # set axes edges
                ax.set_xlim((bounds[0], bounds[-1]))
                ax.set_ylim((0., 1.))

                # labels and grid
                if self.qce._is_cov:
                    title = title + ' - multivariate'
                    ax.set_xlabel("sqrt( Standardized Generalized Variance (SGV) )")
                else:
                    title = title + ' - dim %02d' % dim
                    ax.set_xlabel("Standard Deviation")

                ax.set_ylabel(ylabel)
                ax.grid(True)

                # set axis title
                if title_suffix is not None:
                    title = title + ' - ' + title_suffix

                ax.set_title(title)

        fig.tight_layout()

        # if tikz is true, create tikz code from matplotlib figure
        if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            plt.close(fig)
            fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig
