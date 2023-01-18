# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Iterable, Tuple, List
from typing import Union
import numpy as np

from netcal import is_in_quantile
from netcal.metrics.Miscalibration import _Miscalibration


# annotation hint
QCE_METRIC = Union[float, np.ndarray]
QCE_MAP = Union[List[np.ndarray], None]


class QCE(_Miscalibration):
    """
    Marginal Quantile Calibration Error (M-QCE) and Conditional Quantile Calibration Error (C-QCE) which both measure
    the gap between predicted quantiles and observed quantile coverage also for multivariate distributions.
    The M-QCE and C-QCE have originally been proposed by [1]_.
    The derivation of both metrics are based on
    the Normalized Estimation Error Squared (NEES) known from object tracking [2]_.
    The derivation of both metrics is shown in the following.

    **Definition of standard NEES:**
    Given mean prediction :math:`\\hat{\\boldsymbol{y}} \\in \\mathbb{R}^M`, ground-truth
    :math:`\\boldsymbol{y} \\in \\mathbb{R}^M`, and
    estimated covariance matrix :math:`\\hat{\\boldsymbol{\\Sigma}} \\in \\mathbb{R}^{M \\times M}` using
    :math:`M` dimensions, the NEES is defined as

    .. math::
        \\epsilon = (\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top \\hat{\\boldsymbol{\\Sigma}}^{-1}
        (\\boldsymbol{y} - \\hat{\\boldsymbol{y}}) .

    The average NEES is defined as the mean error over :math:`N` trials in a Monte-Carlo simulation for
    Kalman-Filter testing, so that

    .. math::
        \\bar{\\epsilon} = \\frac{1}{N} \\sum^N_{i=1} \\epsilon_i .

    Under the condition, that :math:`\\mathbb{E}[\\boldsymbol{y} - \\hat{\\boldsymbol{y}}] = \\boldsymbol{0}` (zero mean),
    a :math:`\\chi^2`-test is performed to evaluate the estimated uncertainty. This test is accepted, if

    .. math::
        \\bar{\\epsilon} \\leq \\chi^2_M(\\tau),

    where :math:`\\chi^2_M(\\tau)` is the PPF score obtained by a :math:`\\chi^2` distribution
    with :math:`M` degrees of freedom, for a certain quantile level :math:`\\tau \\in [0, 1]`.

    **Marginal Quantile Calibration Error (M-QCE):**
    In the case of regression calibration testing, we are interested in the gap between predicted quantile levels and
    observed quantile coverage probability for a certain set of quantile levels. We assert :math:`N` observations of our
    test set that are used to estimate the NEES, so that we can compute the expected deviation between predicted
    quantile level and observed quantile coverage by

    .. math::
        \\text{M-QCE}(\\tau) := \\mathbb{E} \\Big[ \\big| \\mathbb{P} \\big( \\epsilon \\leq \\chi^2_M(\\tau) \\big) - \\tau \\big| \\Big] ,

    which is the definition of the Marginal Quantile Calibration Error (M-QCE) [1]_.
    The M-QCE is calculated by

    .. math::
        \\text{M-QCE}(\\tau) = \\Bigg| \\frac{1}{N} \\sum^N_{n=1} \\mathbb{1} \\big( \\epsilon_n \\leq \\chi^2_M(\\tau) \\big) - \\tau \\Bigg|

    **Conditional Quantile Calibration Error (C-QCE):**
    The M-QCE measures the marginal calibration error which is more suitable to test for *quantile calibration*.
    However, similar to :class:`netcal.metrics.regression.UCE` and :class:`netcal.metrics.regression.ENCE`,
    we want to induce a dependency on the estimated covariance, since we require
    that

    .. math::
        &\\mathbb{E}[(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top |
        \\hat{\\boldsymbol{\\Sigma}} = \\boldsymbol{\\Sigma}] = \\boldsymbol{\\Sigma},

        &\\forall \\boldsymbol{\\Sigma} \\in \\mathbb{R}^{M \\times M}, \\boldsymbol{\\Sigma} \\succcurlyeq 0,
        \\boldsymbol{\\Sigma}^\\top = \\boldsymbol{\\Sigma} .

    To estimate the a *covariance* dependent QCE, we apply a binning scheme (similar to the
    :class:`netcal.metrics.confidence.ECE`) over the square root of the *standardized generalized variance* (SGV) [3]_,
    that is defined as

    .. math::
        \\sigma_G = \\sqrt{\\text{det}(\\hat{\\boldsymbol{\\Sigma}})^{\\frac{1}{M}}} .

    Using the generalized standard deviation, it is possible to get a summarized statistic across different
    combinations of correlations to denote the distribution's dispersion. Thus, the Conditional Quantile Calibration
    Error (C-QCE) [1]_ is defined by

    .. math::
        \\text{C-QCE}(\\tau) := \\mathbb{E}_{\\sigma_G, X}\\Big[\\Big|\\mathbb{P}\\big(\\epsilon \\leq \\chi^2_M(\\tau) | \\sigma_G\\big) - \\tau \\Big|\\Big] ,

    To approximate the expectation over the generalized standard deviation, we use a binning scheme with :math:`B` bins
    (similar to the ECE) and :math:`N_b` samples per bin to compute the weighted sum across all bins, so that

    .. math::
        \\text{C-QCE}(\\tau) \\approx \\sum^B_{b=1} \\frac{N_b}{N} | \\text{freq}(b) - \\tau |

    where :math:`\\text{freq}(b)` is the coverage frequency within bin :math:`b` and given by

    .. math::
        \\text{freq}(b) = \\frac{1}{N_b} \\sum_{n \\in \\mathcal{M}_b} \\mathbb{1}\\big(\\epsilon_i \\leq \\chi^2_M(\\tau)\\big) ,

    with :math:`\\mathcal{M}_b` as the set of indices within bin :math:`b`.

    Parameters
    ----------
    bins : int or iterable, default: 10
        Number of bins used for the internal binning.
        If iterable, use different amount of bins for each dimension (nx1, nx2, ... = bins).
    marginal : bool, optional, default: False
        If True, compute the M-QCE. This is the marginal probability over all samples falling into the desired
        quantiles. If False, use the C-QCE.
        The C-QCE uses a binning scheme by the gerneralized standard deviation to measure the conditional probability
        over all samples falling into the desired quantiles w.r.t. the generalized standard deviation.
    sample_threshold : int, optional, default: 1
        Bins with an amount of samples below this threshold are not included into the miscalibration metrics.

    References
    ----------
    .. [1] Küppers, Fabian, Schneider, Jonas, and Haselhoff, Anselm:
       "Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection."
       European Conference on Computer Vision (ECCV) Workshops, 2022.
       `Get source online <https://arxiv.org/pdf/2207.01242.pdf>`__

    .. [2] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan:
       "Estimation with applications to tracking and navigation."
       Theory algorithms and software. John Wiley & Sons, 2004.

    .. [3] A. SenGupta:
       “Tests for standardized generalized variances of multivariate normal populations of possibly different dimensions.”
       Journal of Multivariate Analysis, vol. 23, no. 2, pp. 209–219, 1987.
       `Get source online <https://pdf.sciencedirectassets.com/272481/1-s2.0-S0047259X87X8001X/1-s2.0-0047259X87901539/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEB0aCXVzLWVhc3QtMSJHMEUCIAo6pl1vIu8%2F8ZEnEt6OKsA92ajSFy8%2BgjC8Tt9Tr360AiEAiVljbWcKBWAkrSpyT3LCCiX1Wug9gDf9WqxygiF%2B1G8q0gQIRhAEGgwwNTkwMDM1NDY4NjUiDLFU4Fzvuej7%2FpkSSCqvBNybwrdLhAP1yVFKRn0AQPK3G5t2wOY5ssTpqbS1HQzoz1EvOo%2BvWNeXPrWTo%2FhUvw8knQEWp9wA7a5wGdsbnmaBmCAWUFStKxSib9htTLAX1Ij5NCh%2ByA%2BiJwvPk4SXcRAJx4zQfJ8N%2FM2jcA3Lji7VPDYR8jNu7FfMd54veXYuqzUDoIdhVWIF4Czx0fwqngck9errYuNVSQCGSaW28m3%2BCr3%2FiePEQrXJcX0K7Q3z5D0%2FHGJvMsw%2FDqugg9gjWAFCquBO6EL9ircGBmzVsYnx8pKPZfa84aHZgGJPwvJp01o6qvUEdBZXbC5pmvDNAwIkguMahAHfNM3ffVDfyIVzRq1WrZIAKLYfgqFiGpNd3kGEUX5dGbSggAvxNXP7JX%2BifNsde9cIhzhqpjS0Z8BoOW8AGbXq9n15qEK%2F6xKJttpIhouxiadU6DtW%2BSDDFXDgdMVs1xU%2FE6dnWtl%2BQeA5VX4ITvTQMAjHOqu22lXEmOpHOmZ8fGKG5QTmz4iShixLGsgxpUMp2Wz%2F2OlCwt64m0sbI7zXr3YkUpyR019fs6IHJ28EVbeHsvp9xLrc5qltKHN3aoBmK4ZK0SFi%2FqHLhuz1HmWzCRQ0%2FkShcOXz80ttPlaHuX54VdVptvSy%2F1obsurjYuaR2HG9rO3lIneMgwKwwodZcKlu%2Fyo4WUCIk%2FhfKZrOWqVfAqqFPjWomg4CX787OIoqP%2FTz8H6sueGWk7HoeJKgSIW4yEISjNkwpdH7lQY6qQHS3GFk4XwFQEzwimYc%2B59DxQowARFI3PgPN%2F4FzUV5OHhm5G8455Dzm4T6R6E1%2F%2B86lSLdpH9JCkbDXBPJndiHTqXwBUej4wSWPASAvqQ6neC1lcJAqimCB7zn17je6g1dCi80xg1rZ6qU0c5BGtRqsCAB2NZlNn8jul1%2BM%2BiBF8SLE4xoNnT5kwEI%2FnnY81jvJw1kf4DspI9S8sAI2Z3efRca%2FBTIwgVw&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220701T132912Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZE6GY4EQ%2F20220701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=416cca57ae51cdc7c137e60fa5eba2ba651152e29e6b3485feeeb45f66aaeebb&hash=f03a242d7d2e0652dfaa7c5e3fb713736b321247242986106eaa5412f136af12&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0047259X87901539&tid=spdf-4216aa85-a41c-4358-8d5e-3e877785fe29&sid=4e9c65da63ee0949f79bfc42fc03dd438d5egxrqb&type=client&ua=4d545d0256065c055303&rr=723f7d3c3edc2199>`__
    """

    def __init__(
            self,
            bins: int = 10,
            *,
            marginal: bool = True,
            sample_threshold: int = 1
    ):
        """ Constructor. For detailed parameter description, see class docs. """

        super().__init__(bins=bins, equal_intervals=True, detection=False, sample_threshold=sample_threshold)
        self.marginal = marginal

        # these fields only act as a buffer to reuse within QCE presentation
        self._bin_edges = []
        self._is_cov = False

    def measure(
            self,
            X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            y: np.ndarray,
            q: Union[float, Iterable[float], np.ndarray],
            *,
            kind: str = 'meanstd',
            reduction: str = 'batchmean',
            range_: List[Tuple[float, float]] = None,
            return_map: bool = False,
            return_num_samples: bool = False,
    ) -> Union[QCE_METRIC, Tuple[QCE_METRIC, QCE_MAP], Tuple[QCE_METRIC, QCE_MAP, QCE_MAP]]:
        """
        Measure quantile loss for given input data either as tuple consisting of mean and stddev estimates or as
        NumPy array consisting of a sample distribution. The loss is computed for several quantiles
        given by parameter q.

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
            - confidence: asserts X as a single NumPy array of shape (t, n, [1]) with t stochastic forward passes with
                          scores in [0, 1] that represent confidence scores obtained e.g. by Monte-Carlo sampling.
                          Furthermore, this mode asserts the ground-truth labels 'y' in the {0, 1} set and converts
                          them to continuous [0, 1] scores by binning. Thus, it is possible to evaluate the
                          confidence uncertainty with binned labels.
        reduction : str, one of 'none', 'mean' or 'batchmean', default: 'batchmean'
            Specifies the reduction to apply to the output:
            - none : no reduction is performed. Return QCE for each sample and for each dim separately.
            - mean : calculate mean over all quantiles and all dimensions.
            - batchmean : calculate mean over all quantiles but for each dim separately.
                          If input has covariance matrices, 'batchmean' is the same as 'mean'.
        range_ : list of length d with tuples (lower_bound: float, upper_bound: float)
            List of tuples that define the binning range of the standard deviation for each dimension separately.
            For example, if input data is given with only a few samples having high standard deviations,
            this might distort the calculations as the binning scheme commonly takes the (min, max) as the range
            for the binning, yielding a high amount of empty bins.
        return_map: bool, optional, default: False
            If True, return miscalibration score for each quantile and each bin separately. Otherwise, compute mean
            over all quantiles and all bins.
        return_num_samples : bool, optional, default: False
            If True, also return the number of samples in each bin.

        Returns
        -------
        float or np.ndarray or tuple of (float, List[np.ndarray], [List[np.ndarray]])
            Always returns miscalibration metric. See parameter "reduction" for a detailed description.
            If 'return_map' is True, return tuple and append miscalibration map over all bins.
            If 'return_num_samples' is True, return tuple and append the number of samples in each bin (excluding confidence dimension).
        """

        # clear bin_edges buffer list and is_cov buffer
        self._bin_edges = []
        self._is_cov = False

        # kind 'confidence' induces a different treatment - the target scores given by 'y' are assumed
        # to be in the {0, 1} set and are converted to continuous [0, 1] scores by binning the samples.
        # Thus, it is possible to evaluate the confidence uncertainty
        if kind == "confidence":
            result = self.frequency(X=X, y=y, batched=False, uncertainty="mean")

            # frequency is 2nd entry in return tuple and within first batch
            X = X[..., :1]  # (t, n, 1)
            y = np.reshape(result[1][0], (-1, 1))  # (n, 1)

            # kind 'meanstd' will result in a HPDI computation if sample distribution is given
            kind = "meanstd"

        in_quantile, _, _, mean, var = is_in_quantile(X, y, q, kind)  # (q, n, [d]), (q, n, d), (n, d), (n, d, [d])
        n_samples, n_dims = mean.shape

        # if marginal, do not use a binning scheme but rather compute the marginal
        # probabilities for each quantile level
        if self.marginal:
            frequency = np.mean(in_quantile, axis=1)  # (q, [d])

            # expand q if frequency ndim = 2
            q = np.expand_dims(q, axis=1) if frequency.ndim == 2 else q

            # QCE is difference between expected quantile level and actual frequency
            qce = np.abs(frequency - q)  # (q, [d])

            qce_map = None
            num_samples_hist = None

        # if conditional, apply a binning scheme
        else:

            # independent variables: compute histogram edges for each dimensions independently
            if var.ndim == 2:

                self._is_cov = False
                qce, qce_map, num_samples_hist = [], [], []
                std = np.sqrt(var)

                # prepare binning boundaries for regression
                bin_bounds = self._prepare_bins_regression(std, n_dims=n_dims, range_=range_)

                for dim in range(n_dims):

                    # split up arrays for each quantile
                    splitted = [x[:, dim] for x in in_quantile]  # [(n,), (n,), ...] of length q

                    # perform binning over 1D variance
                    freq_hist, n_samples_hist, bounds, _ = self.binning([bin_bounds[dim]], std[:, dim], *splitted)
                    freq_hist, n_samples_hist = np.array(freq_hist), np.array(n_samples_hist)  # (q, b) , (b,)

                    # calculate the difference between the expected quantile level and the observed frequency
                    # compute the weighted sum
                    qce_map_dim = np.abs(freq_hist - q[:, None])  # (q, b)
                    qce_dim = np.average(qce_map_dim, axis=1, weights=n_samples_hist / n_samples)  # (q,)

                    # add dimension results to overal result lists
                    qce.append(qce_dim)
                    qce_map.append(qce_map_dim)
                    num_samples_hist.append(n_samples_hist)
                    self._bin_edges.extend(bounds)

                qce = np.stack(qce, axis=1)  # (q, d)

            # multivariate (dependent case) - compute joint histogram bounds
            # based on standardized generalized variance
            else:

                self._is_cov = True

                # matrix determinant raise an exception - catch this to output a more meaningful error
                try:
                    determinant = np.linalg.det(var)  # (n,)
                except np.linalg.LinAlgError:
                    raise RuntimeError("QCE: found invalid input covariance matrices.")

                # calculate standardized generalized variance
                sgv = np.power(determinant, 1. / (n_dims))  # (n,)
                sg_std = np.sqrt(sgv)

                # prepare binning boundaries for regression
                bin_bounds = self._prepare_bins_regression(sg_std, n_dims=1, range_=range_)

                # split up arrays for each quantile
                splitted = [x for x in in_quantile]  # [(n,), (n,), ...] of length q

                # perform binning over SGV
                frequency_hist, num_samples_hist, edges, _ = self.binning(bin_bounds, sg_std, *splitted)
                frequency_hist, num_samples_hist = np.array(frequency_hist), np.array(num_samples_hist)  # (q, b) , (b,)

                # calculate the difference between the expected quantile level and the observed frequency
                # compute the weighted sum
                qce_map = np.abs(frequency_hist - q[:, None])  # (q, b)
                qce = np.average(qce_map, axis=1, weights=num_samples_hist / n_samples)  # (q,)

                qce_map = [qce_map]
                num_samples_hist = [num_samples_hist]

                self._bin_edges.extend(edges)

        # no reduction is applied
        if reduction is None or reduction == 'none':
            metric = qce

        # 'mean' is mean over all quantiles and all dimensions
        elif reduction == "mean":
            metric = float(np.mean(qce))

        # 'batchmean' is mean over all quantiles but for each dim separately.
        # If input has covariance matrices, 'batchmean' is the same as 'mean'.
        elif reduction == "batchmean":
            metric = np.mean(qce, axis=0)  # (d,)

        # unknown reduction method
        else:
            raise RuntimeError("Unknown reduction: \'%s\'" % reduction)

        # build output structure w.r.t. user input
        if not return_map and not return_num_samples:
            return metric

        return_value = (metric,)
        if return_map and not self.marginal:
            return_value = return_value + (qce_map,)

        if return_num_samples and not self.marginal:
            return_value = return_value + (num_samples_hist,)

        return return_value
