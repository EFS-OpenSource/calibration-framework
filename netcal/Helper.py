# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union, Iterable, Tuple
import multiprocessing as mp
import numpy as np

from scipy.stats import norm, cauchy, chi2
from scipy.interpolate import interp1d

from .Decorator import global_accepts, global_dimensions
from .Stats import hpdi


@global_accepts((tuple, list))
def _extract_input(X: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Extract input tuple or list. Assert that the input consists of two Numpy arrays containing mean and stddev.

    Parameters
    ----------
    X : tuple of two np.ndarray, each of shape (n,)
        Input data obtained by a model that performs inference with uncertainty.
        Assert X as a tuple of two NumPy arrays with shape (n,) for each array, where the
        first array is interpreted as mean and the second one as the according stddev predictions.
    cov : bool, optional, default: False
        If True, extract covariance matrix on input instead of single independent variances.

    Returns
    -------
    tuple of two np.ndarray and bool
        Return unpacked mean and stddev/cov and flag if covariance matrix is returned.
    """

    # assert input data and std dev as input
    assert len(X) == 2, "If \'X\' is provided with type of {list, tuple}, this methods asserts a length of 2 " \
                        "with X[0] as the predicted mean and X[1] as the predicted stddev."

    # extract mean and stddev of parameter X
    mean, stddev = X[0], X[1]

    # if both are passed as lists, convert to numpy arrays
    mean = squeeze_generic(np.array(mean), axes_to_keep=0)
    stddev = squeeze_generic(np.array(stddev), axes_to_keep=0)

    assert len(mean) == len(stddev), "Different amount of samples passed for mean and stddev."

    cov = mean.ndim == stddev.ndim - 1
    if not cov:
        assert (stddev > 0).all(), "Found zero or negative stddev."
    else:
        assert mean.ndim == stddev.ndim - 1, "Asserted covariance matrix, but invalid amount of dimensions passed for mean and cov."
        assert (np.diagonal(stddev, axis1=-2, axis2=-1) > 0).all(), "Asserted covariance matrix, but found zero or negative variances in cov."

    return mean, stddev, cov


def squeeze_generic(a: np.ndarray, axes_to_keep: Union[int, Iterable[int]]) -> np.ndarray:
    """
    Squeeze input array a but keep axes defined by parameter 'axes_to_keep' even if the dimension is
    of size 1.

    Parameters
    ----------
    a : np.ndarray
        NumPy array that should be squeezed.
    axes_to_keep : int or iterable
        Axes that should be kept even if they have a size of 1.

    Returns
    -------
    np.ndarray
        Squeezed array.
    """

    # if type is int, convert to iterable
    if type(axes_to_keep) == int:
        axes_to_keep = (axes_to_keep, )

    # iterate over all axes in a and check if dimension is in 'axes_to_keep' or of size 1
    out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
    return a.reshape(out_s)


def meanvar(
        X: Union[Iterable[np.ndarray], np.ndarray],
        y: np.ndarray = None
) -> Union[Tuple[np.ndarray, np.ndarray, bool], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, bool]]:
    """
    Convert input X to mean and variance. The input might be given either as tuple consisting of NumPy arrays
    containing mean and stddev. Alternatively, the input might also consist of multiple stochastic samples obtained
    by a sampling algorithm (e.g. MC dropout).

    It is also possible to convert target y scores in {0, 1} to observed frequencies in [0, 1] to convert a
    classification to a regression task. For this purpose, the parameters 'classification' and 'bins' control the
    conversion of the target scores.

    Parameters
    ----------
    X : np.ndarray of shape (r, n, [d]) or Tuple of two np.ndarray, each of shape (n, [d])
        Input data for calibration regression obtained by a model that performs inference with uncertainty.
        Depending on the input format, this method handles the input differently:
        If X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
        first array as mean and the second one as the according stddev predictions with d dimensions (optionally).
        If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
        inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
        case, the mean and stddev is computed automatically.
    y : np.ndarray of shape (n, [d])
        Target scores for each prediction estimate in X.

    Returns
    -------
    tuple of np.ndarray (mean, var) if y=None or (mean, var), y if y is given, with additional bool
        Mean and variance obtained by X. If y is given, also return the (probably) converted target scores.
        Also, always return boolean flag if covariance matrix has been found in input data.
    """

    # extract input if X is given as iterable
    if isinstance(X, (tuple, list)):
        mean, stddev, cov = _extract_input(X)

        # if covariance matrix estimation is enabled, do not square matrix
        if cov:
            variance = stddev
        else:
            variance = np.square(stddev)

    elif isinstance(X, np.ndarray):
        # assert input data of multiple stochastic forward passes per sample (e.g. by MC dropout)
        assert X.ndim >= 2, "If \'X\' is provided as NumPy array, this methods asserts values obtained by " \
                            "multiple stochastic forward passes (e.g. by MC dropout) of shape (R, N, [D]) with " \
                            "R forward passes, N number of samples and D dimensions (optional)."

        # TODO: in general, it should be possible to capture covariance matrices here!
        mean = np.mean(X, axis=0)  # (n, [d])
        variance = np.var(X, axis=0)  # (n, [d])
        cov = False

    else:
        raise ValueError("Parameter \'X\' must be either of type {list, tuple} with mean and stddev passed or "
                         "of NumPy array of shape (R, N, [D]) with N number of samples, R stochastic forward "
                         "passes (e.g. by MC dropout) and D dimensions (optional).")

    # remove data dim if d=1
    mean = squeeze_generic(mean, axes_to_keep=0)
    variance = squeeze_generic(variance, axes_to_keep=0)
    if y is not None:
        y = squeeze_generic(y, axes_to_keep=0)
        return (mean, variance), y, cov
    else:
        return mean, variance, cov


@global_dimensions(None, (2, 3))
def cumulative(X: Union[Iterable[np.ndarray], np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get cumulative distribution function at y using the moments passed in X. The input X might be given either as tuple
    consisting of NumPy arrays containing mean and stddev. Alternatively, the input might also consist of multiple
    stochastic samples obtained by a sampling algorithm (e.g. MC dropout).

    It is also possible to convert target y scores in {0, 1} to observed frequencies in [0, 1] to convert a
    classification to a regression task. For this purpose, the parameters 'classification' and 'bins' control the
    conversion of the target scores.

    Parameters
    ----------
    X : np.ndarray of shape (r, n, [d]) or Tuple of two np.ndarray, each of shape (n, [d])
        Input data obtained by a model that performs inference with uncertainty.
        Depending on the input format, this method handles the input differently:
        If X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
        first array as mean and the second one as the according stddev predictions for d dimensions.
        If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
        inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
        case, the mean and stddev is computed automatically.
    y : np.ndarray of shape (t, n, [d])
        Target scores for each prediction estimate in X. If y has dim t, interpret this dimension as multiple
        points on the CDF curve for each sample.

    Returns
    -------
    tuple of np.ndarray (t/y, n, d)
        Cumulative distribution function of shape (t, n, d). Also return the (probably) converted target scores y.
    """

    # make y at least 3d (with feature dim if not given)
    y = np.expand_dims(y, axis=2) if y.ndim == 2 else y  # (t, n, d)

    # extract input if X is given as iterable
    if isinstance(X, (tuple, list)):
        mean, stddev, cov = _extract_input(X)

        # catch covariance input
        if cov:
            raise RuntimeError("Covariance input is currently not supported for quantile computation.")

        # make mean and stddev at least 2d
        mean = mean.reshape((-1, 1)) if mean.ndim == 1 else mean  # (n, d)
        stddev = stddev.reshape((-1, 1)) if stddev.ndim == 1 else stddev  # (n, d)

        # get CDF for the corresponding target scores
        ret = norm.cdf(y, loc=mean[None, ...], scale=stddev[None, ...])  # (t, n, d)

    elif isinstance(X, np.ndarray):
        # assert input data of multiple stochastic forward passes per sample (e.g. by MC dropout)
        assert X.ndim >= 2, "If \'X\' is provided as NumPy array, this methods asserts values obtained by " \
                            "multiple stochastic forward passes (e.g. by MC dropout) of shape (R, N, [D]) with " \
                            "R forward passes, N number of samples and D dimensions (optional)."

        # make X at least 3d
        X = np.expand_dims(X, axis=2) if X.ndim == 2 else X  # (r, n, d)

        # get CDF of each distribution by measuring the fraction of samples below target y
        ret = np.mean(
            X[None, ...] <= y[:, None, ...],  # (t, r, n, d)
            axis=1,
        )  # (t, n, d)

    else:
        raise ValueError("Parameter \'X\' must be either of type {list, tuple} with mean and stddev passed or "
                         "of NumPy array of shape (R, N, [D]) with N number of samples, R stochastic forward "
                         "passes (e.g. by MC dropout) and D dimensions (optional).")

    return ret, y


def _get_density(t, t_mid, pdf):
    """
    Helper function for "density_from_cumulative" that performs interpolation for a single sample in a single dim.
    """

    return interp1d(
        t_mid,
        pdf,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )(t)  # (t,)


def density_from_cumulative(t: np.ndarray, cdf: np.ndarray):
    """
    Return the probability density function (PDF) given the respective cumulative (CDF) function.

    Parameters
    ----------
    t : np.ndarray of shape (t, n, [d])
        NumPy array that defines the base points (x-axis) of the cumulative distribution function with
        t sampling points, n samples and d dimensions.
    cdf : np.ndarray of shape (t, n, [d])
        NumPy array with actual cumulative scores for each t with t sampling points, n samples and d dimensions.

    Returns
    -------
    np.ndarray of shape (t, n, d)
        Probability density function at points t.
    """

    t = np.expand_dims(t, axis=2) if t.ndim == 2 else t
    cdf = np.expand_dims(cdf, axis=2) if cdf.ndim == 2 else cdf

    _, n_samples, n_dims = t.shape

    # instead of using diffs built-in prepend method, we manually repeat the first PDF score to obtain
    # the same dimension as for CDF
    delta_t = np.clip(np.diff(t, axis=0), a_min=np.finfo(np.float32).eps, a_max=None)  # (t-1, n, d)
    pdf = np.diff(cdf, axis=0) / delta_t  # (t-1, n, d)

    # at this point, the density is defined between the sampling points. Use interpolation to scale them back
    t_mid = (t[:-1, ...] + t[1:, ...]) / 2  # (t-1, n, d)

    density = []
    for d in range(n_dims):
        results = [_get_density(t[:, n, d], t_mid[:, n, d], pdf[:, n, d]) for n in range(n_samples)]

        results = np.stack(results, axis=1)  # (t, n)
        density.append(results)

    density = np.stack(density, axis=2)  # (t, n, d)

    return density


def cumulative_moments(t: np.ndarray, cdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mean and variance of a probability distribution that is defined by a cumulative distribution.

    Parameters
    ----------
    t : np.ndarray of shape (t, n, [d])
        NumPy array that defines the base points (x-axis) of the cumulative distribution function with
        t sampling points, n samples and d dimensions.
    cdf : np.ndarray of shape (t, n, [d])
        NumPy array with actual cumulative scores for each t with t sampling points, n samples and d dimensions.

    Returns
    -------
    Tuple of 2 np.ndarray, both of shape (n, d)
        Mean and variance for each input sample and for each dimension.
    """

    t = np.expand_dims(t, axis=2) if t.ndim == 2 else t
    cdf = np.expand_dims(cdf, axis=2) if cdf.ndim == 2 else cdf

    delta_cdf = np.diff(cdf, axis=0)  # (t-1, n, [d])
    t = (t[:-1, ...] + t[1:, ...]) / 2  # (t-1, n, [d])

    mean = np.sum(t * delta_cdf, axis=0)
    var = np.sum(np.square(t) * delta_cdf, axis=0) - np.square(mean)

    return mean, var


def is_in_quantile(
        X: Union[Iterable[np.ndarray], np.ndarray],
        y: np.ndarray,
        q: Union[float, Iterable[float], np.ndarray],
        kind: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Determine if the ground-truth information :math:`Y` is in the prediction interval of a predicted probabilitiy
    distribution specified by :math:`X` for q certain quantile level :math:`\\tau`.

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
        - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) for each array, this method asserts the
                   first array as mean and the second one as the according stddev predictions for d dimensions.
                   If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                   inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes.
                   In this case, the mean and stddev is computed automatically.
        - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                      for sample n (and optionally d dimensions).

    Returns
    -------
    tuple of 4 np.ndarray
        Tuple with four NumPy arras:
        (1) Boolean flag for each input sample if ground-truth y is in a certain prediction interval
            specified by quantile level q.
        (2) Quantile boundaries for each quantile q w.r.t. distributions given by X. Note that we're working with
            two-sided quantiles. Thus, we return either the lower or the upper quantile boundary, depending on its
            distance to the target score given by Y.
        (3) Tuple with lower and upper quantile boundaries.
        (4) Parametric mean.
        (5) Parametric variance/covariance.
    """
    assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'

    # if quantiles are given as list or float, convert to numpy array
    q = squeeze_generic(np.array(q), axes_to_keep=0).reshape((-1,))  # (q,)
    alpha = 1. - q  # (q,)

    # make y at least 2d
    y = np.expand_dims(y, axis=1) if y.ndim == 1 else y  # (n, d)

    if kind == 'cumulative':

        t, cdf = X[0], X[1]  # (t, n, [d])

        # make CDF at least 3d
        t = np.expand_dims(t, axis=2) if t.ndim == 2 else t  # (t, n, d)
        cdf = np.expand_dims(cdf, axis=2) if cdf.ndim == 2 else cdf  # (t, n, d)

        alpha = alpha.reshape((-1, 1, 1, 1))  # (q, 1, 1, 1)

        # for the moment, use the nearest point of the curve
        # cdf has shape (t, n), q has shape (q,)
        nearest_min = np.argmin(
            np.abs(cdf[None, ...] - (alpha / 2.)),  # (q, t, n, d)
            axis=1
        )  # (q, n, d)

        nearest_max = np.argmin(
            np.abs(cdf[None, ...] - (1. - (alpha / 2.))),  # (q, t, n, d)
            axis=1
        )  # (q, n, d)

        # extract the interval boundaries and check if y is within prediction interval
        qbounds_min = np.take_along_axis(t, nearest_min, axis=0)  # (q, n, d)
        qbounds_max = np.take_along_axis(t, nearest_max, axis=0)  # (q, n, d)

        in_quantile = (y[None, ...] >= qbounds_min) & (y[None, ...] <= qbounds_max)  # (q, n, d)

        # now choose the quantile boundaries that are closer to our ground-truth Y
        qbounds_next = np.where(
            np.abs(qbounds_min - y[None, ...]) < np.abs(qbounds_max - y[None, ...]),
            qbounds_min,
            qbounds_max
        )  # (q, n, d)

        qbounds = (qbounds_min, qbounds_max)

        # calculate mean and variance from cumulative
        mean, var = cumulative_moments(t, cdf)

    # Cauchy distribution is given - check if samples are within two-sided quantile boundaries
    elif kind == 'cauchy':

        mode, scale = X[0], X[1]  # (t, n, [d])

        # catch if multivariate Cauchy with correlations is given
        if scale.ndim > mode.ndim:
            raise RuntimeError("Multivariate Cauchy with correlations is currently not supported.")

        # make mode and scale at least 2d
        mode = mode.reshape((-1, 1)) if mode.ndim == 1 else mode  # (n, d)
        scale = scale.reshape((-1, 1)) if scale.ndim == 1 else scale  # (n, d)

        alpha = alpha.reshape((-1, 1, 1))  # (q, 1, 1)
        qbounds_min = cauchy.ppf(alpha / 2., loc=mode[None, ...], scale=scale[None, ...])  # (q, n, d)
        qbounds_max = cauchy.ppf(1. - (alpha / 2.), loc=mode[None, ...], scale=scale[None, ...])  # (q, n, d)

        in_quantile = (y[None, ...] >= qbounds_min) & (y[None, ...] <= qbounds_max)  # (q, n, d)

        # now choose the quantile boundaries that are closer to our ground-truth Y
        qbounds_next = np.where(
            np.abs(qbounds_min - y[None, ...]) < np.abs(qbounds_max - y[None, ...]),
            qbounds_min,
            qbounds_max
        )  # (q, n, d)

        qbounds = (qbounds_min, qbounds_max)

        # TODO: surrogate fix, edit names!
        mean = mode
        var = scale ** 2

    else:

        # get quantile bounds and convert q to numpy array (if not already given)
        # extract input if X is given as iterable
        if isinstance(X, (tuple, list)):
            mean, stddev, cov = _extract_input(X)

            # make mean and stddev at least 2d
            mean = mean.reshape((-1, 1)) if mean.ndim == 1 else mean  # (n, d)

            # if covariance matrix is given in input, compute the multivariate HDR using the
            # Mahalanobis distance and the chi2 test for multivariate normal
            if cov:

                # get number of dimensions (necessary for chi2 distribution)
                ndims = mean.shape[-1]

                # reshape cov to the right dimensions
                cov = np.expand_dims(stddev, axis=-1) if stddev.ndim == 2 else stddev  # (n, d, d)
                var = cov

                # try matrix inversion and get eigenvalues and eigenvectors to determine the isolines
                # of multivariate normal
                try:
                    invcov = np.linalg.inv(cov)  # (n, d, d)
                except np.linalg.LinAlgError:
                    raise RuntimeError("Input covariance matrices are not positive semidefinite.")

                # reshape diff to the right dimensions and compute Mahalanobis distance
                diff = np.expand_dims(y - mean, axis=-1)  # (n, d, 1)
                mahalanobis = np.transpose(diff, axes=(0, 2, 1)) @ invcov @ diff  # (n, 1, 1)
                mahalanobis = np.squeeze(mahalanobis, axis=(1, 2))  # (n,)

                # currently, qbounds is not defined/implemented for multivariate normal distributions
                qbounds_next = None
                qbounds = (None, None)

            # in the independent case, we also use the Mahalanobis distance but for each dimension separately
            else:

                ndims = 1
                stddev = stddev.reshape((-1, 1)) if stddev.ndim == 1 else stddev  # (n, d)

                # use the variance to compute the Mahalanobis distance
                # in the _extract_input function, we already guarantee nonzero variances
                var = np.square(stddev)  # (n, d)
                mahalanobis = np.divide(np.square(y - mean), var)  # (n, d)

                # since we're working with two-sided quantiles, we choose the quantile boundaries that
                # are closer to our ground-truth target Y
                qbounds_min = norm.ppf(
                    np.expand_dims(alpha / 2., axis=(1, 2)),  # (q, 1, 1)
                    loc=mean[None, ...],  # (1, n, d)
                    scale=stddev[None, ...]  # (1, n, d)
                )  # (q, n, d)

                qbounds_max = norm.ppf(
                    np.expand_dims(1. - (alpha / 2.), axis=(1, 2)),  # (q, 1, 1)
                    loc=mean[None, ...],  # (1, n, d)
                    scale=stddev[None, ...]  # (1, n, d)
                )  # (q, n, d)

                # now choose the quantile boundaries that are closer to our ground-truth Y
                qbounds_next = np.where(
                    np.abs(qbounds_min - y[None, ...]) < np.abs(qbounds_max - y[None, ...]),
                    qbounds_min,
                    qbounds_max
                )

                qbounds = (qbounds_min, qbounds_max)

            # to test for a two-sided quantile coverage, use the (single-sided) chi2 quantile
            # and compare with Mahalanobis distance
            chi2_bounds = chi2.ppf(q, df=ndims)  # (q,)
            chi2_bounds = np.expand_dims(chi2_bounds, axis=list(np.array(range(mahalanobis.ndim)) + 1))  # (q, 1, [1])

            # finally, check if mahalanobis distance is below requested chi2 quantile boundary
            # this is valid for two-sided Gaussian quantiles
            in_quantile = mahalanobis[None, ...] <= chi2_bounds  # (q, n, [d])

        elif isinstance(X, np.ndarray):

            # assert input data of multiple stochastic forward passes per sample (e.g. by MC dropout)
            assert X.ndim >= 2, "If \'X\' is provided as NumPy array, this methods asserts values obtained by " \
                                "multiple stochastic forward passes (e.g. by MC dropout) of shape (R, N, [D]) with " \
                                "R forward passes, N number of samples and D dimensions (optional)."

            # make X at least 3d
            X = np.expand_dims(X, axis=2) if X.ndim == 2 else X  # (r, n, d)

            # on inference mode, get two-sided quantile bounds for remapped CDF scores
            # use HPDI to get the quantile bounds
            qbounds_min, qbounds_max = [], []
            for prob in q:
                qbounds_next = hpdi(X, prob=prob, axis=0)  # (2, n, d)
                qbounds_min.append(qbounds_next[0])
                qbounds_max.append(qbounds_next[1])

            qbounds_min = np.stack(qbounds_min, axis=0)  # (q, n, d)
            qbounds_max = np.stack(qbounds_max, axis=0)  # (q, n, d)

            in_quantile = (y[None, ...] >= qbounds_min) & (y[None, ...] <= qbounds_max)  # (q, n, d)

            # now choose the quantile boundaries that are closer to our ground-truth Y
            qbounds_next = np.where(
                np.abs(qbounds_min - y[None, ...]) < np.abs(qbounds_max - y[None, ...]),
                qbounds_min,
                qbounds_max
            )

            qbounds = (qbounds_min, qbounds_max)

            mean = np.mean(X, axis=0)  # (n, d)
            var = np.var(X, axis=0)  # (n, d)

        else:
            raise ValueError("Parameter \'X\' must be either of type {list, tuple} with mean and stddev passed or "
                             "of NumPy array of shape (R, N, [D]) with N number of samples, R stochastic forward "
                             "passes (e.g. by MC dropout) and D dimensions (optional).")

    return in_quantile, qbounds_next, qbounds, mean, var
