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

from netcal import is_in_quantile


class ReliabilityRegression(object):
    """
    Reliability diagram in the scope of regression calibration for probabilistic regression models.
    This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    coverage scores above the desired quantile levels.
    In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    is given as multiple independent Gaussians.
    This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    input is given with covariance matrices.

    Parameters
    ----------
    quantiles : int or iterable, default: 11
        Quantile levels that are used for the visualization of the regression reliability diagram.
        If int, use NumPy's linspace method to get the quantile levels.
        If iterable, use the specified quantiles for visualization.
    """

    eps = np.finfo(np.float32).eps

    def __init__(self, quantiles: Union[int, Iterable[float], np.ndarray] = 11):
        """ Constructor. For detailed parameter documentation view classdocs. """

        # init list of quantiles if input type is int
        if isinstance(quantiles, int):
            self.quantiles = np.clip(np.linspace(0., 1., quantiles), self.eps, 1.-self.eps)

        # use input list or array as quantile list
        elif isinstance(quantiles, (list, np.ndarray)):

            # at this point, allow for 0 and 1 quantile to be aligned on the miscalibration curve
            assert (quantiles >= 0).all(), "Found quantiles <= 0."
            assert (quantiles <= 1).all(), "Found quantiles >= 1."
            self.quantiles = np.clip(np.array(quantiles), self.eps, 1.-self.eps)

        else:
            raise AttributeError("Unknown type \'%s\' for param \'quantiles\'." % type(quantiles))

    def plot(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            *,
            kind: str = 'meanstd',
            filename: str = None,
            tikz: bool = False,
            title_suffix: str = None,
            feature_names: List[str] = None,
            **save_args
    ) -> Union[plt.Figure, str]:
        """
        Reliability diagram for regression calibration to visualize the predicted quantile levels vs. the actually
        observed quantile coverage probability.
        This method is able to visualize the reliability diagram in terms of multiple univariate distributions if the
        input is given as multiple independent Gaussians.
        This method is also able to visualize the joint multivariate quantile calibration for a multivariate Gaussian
        if the input is given with covariance matrices (see parameter "kind" for a detailed description of the input
        format).

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
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
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        feature_names : list, optional, default: None
            Names of the additional features that are attached to the axes of a reliability diagram.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the quantile calibration either as Matplotlib figure or as string with tikz code.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'

        # get quantile coverage of input
        in_quantile, _, _, _, _ = is_in_quantile(X, y, self.quantiles, kind)  # (q, n, [d]), (q, n, d), (n, d), (n, d, [d])

        # get the frequency of which y is within the quantile bounds
        frequency = np.mean(in_quantile, axis=1)  # (q, [d])

        # make frequency array at least 2d
        if frequency.ndim == 1:
            frequency = np.expand_dims(frequency, axis=1)  # (q, d) or (q, 1)

        n_dims = frequency.shape[-1]

        # check feature names parameter
        if feature_names is not None:
            assert isinstance(feature_names, (list, tuple)), "Parameter \'feature_names\' must be tuple or list."
            assert len(feature_names) == n_dims, "Length of parameter \'feature_names\' must be equal to the amount " \
                                                 "of dimensions. Input with full covariance matrices is interpreted " \
                                                 "as n_features=1."

        # initialize plot and create an own chart for each dimension
        fig, axes = plt.subplots(nrows=n_dims, figsize=(7, 3 * n_dims), squeeze=False)
        for dim, ax in enumerate(axes):

            # ax object also has an extra dim for columns
            ax = ax[0]

            ax.plot(self.quantiles, frequency[:, dim], "o-")

            # draw diagonal as perfect calibration line
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))

            # labels and legend of second plot
            ax.set_xlabel('Expected quantile')
            ax.set_ylabel('Observed frequency')
            ax.legend(['Output', 'Perfect Calibration'])
            ax.grid(True)

            # set axis title
            title = 'Reliability Regression Diagram'
            if title_suffix is not None:
                title = title + ' - ' + title_suffix
            if feature_names is not None:
                title = title + ' - ' + feature_names[dim]
            else:
                title = title + ' - dim %02d' % dim

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
