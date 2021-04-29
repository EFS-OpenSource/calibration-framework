# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Use this file to create an artificial dataset to demonstrate the effect of position-dependent
confidence calibration used for object detection calibration.
"""

import numpy as np
from scipy.stats import truncnorm, multivariate_normal


def create_bivariate_normal(num_samples: int = 1000,
                            mean: np.ndarray = None, cov: np.ndarray = None,
                            accuracy: float = 0.7, calibration_error: float = 0.0) -> tuple:
    """
    Create bivariate normal distribution with covariances and commanded calibration error.

    Parameters
    ----------
    num_samples : int, default: 1000
        Number of samples to generate
    mean : np.ndarray of shape=(n_features,), optional, default: [0.5, 0.5]
        Mean of bivariate normal distribution.
    cov : np.ndarray of shape=(n_features, n_features), optional, default: [[0.01, 0], [0, 0.01]]
        Covariance matrix of bivariate normal distribution.
    accuracy : float, optional, default: 0.7
        Commanded accuracy.
    calibration_error : float, optional, default: 0
        Commanded calibration error

    Returns
    -------
    tuple of size 3
        Returns 2D NumPy array with (x, y) position, matched values and confidence values for each sample.
    """

    # default mean
    if mean is None:
        mean = np.array([.5, .5])

    # default covariance matrix
    if cov is None:
        cov = np.array([[.01, .0], [.0, .01]])

    # sample from multivariate normal distribution as long as number of samples limit is reached
    samples = []
    i = 0
    while i<num_samples:

        # sample from normal
        sample = np.random.multivariate_normal(mean=mean, cov=cov, size=1)

        # discard samples from out of scope (out of interval [0,1])
        if (sample > 1.0).any() or (sample < 0.0).any():
            continue
        else:
            samples.append(sample)
            i += 1

    samples = np.array(samples).squeeze()

    # number of bins, bin bounds and bin median for matches and confidences
    n_bins = 10
    bounds = np.linspace(0.0, 1.0, n_bins + 1)
    bin_median = (bounds[:-1] + bounds[1:]) / 2.

    # some magic numbers to scale up covariance values
    # those scale factors control the strength of "value decay" to the boundaries
    # the higher the number, the more constant are the values over the distribution
    shift_scale = 3.0
    shift_cov = cov * shift_scale

    conf_scale = 10.0
    conf_cov = cov * conf_scale

    # sample a normalizer and other samples
    # this guarantees values in the interval [0, 1] different from 1
    # 1st part: probabilities for the accuracy in each region ('matched/not matched' flag)
    normalizer = multivariate_normal.pdf(mean, mean=mean, cov=shift_cov)
    sample_scaling = multivariate_normal.pdf(samples, mean=mean, cov=shift_cov) / normalizer

    # 2nd part: scaling for confidence estimates in each region
    normalizer = multivariate_normal.pdf(mean, mean=mean, cov=conf_cov)
    conf_scaling = multivariate_normal.pdf(samples, mean=mean, cov=conf_cov) / normalizer

    # now sample the "ground-truth" information according to the probabilities obtained beforehand
    matched = []
    for scale in sample_scaling:

        # clip probabilitiy to [0, 1] interval and sample from {0, 1}
        prob = np.clip(accuracy * scale, 0.0, 1.0)
        matched.append(np.random.choice([0, 1], size=1, replace=True, p=[1. - prob, prob]))

    matched = np.array(matched).squeeze()

    # some more magic numbers: loc is the mean accuracy, scale the stddev of the accuracy
    loc = accuracy+0.0
    scale = 0.15

    # upper and lower bounds for truncated normal
    a, b = (0 - loc) / scale, (1 - loc) / scale

    # now we need to distribute the samples along the confidence bins
    # thus, we need a probability of sample occurrence over the confidence bins with a peak on the desired
    # mean accuracy

    # use a CDF and calculate the n-th discrete difference along the second axis
    # this guarantees, that the occurrence probability sums up to 1
    cumulated = np.array(
        [truncnorm.cdf([bounds[i], bounds[i + 1]], a, b, loc=loc, scale=scale) for i in range(n_bins)]
    )
    bin_props = np.diff(cumulated, axis=1)

    # convert the probabilities into absolute sample values and check if the number of desired samples still matches
    samples_per_bin = np.around(bin_props * num_samples).astype(np.int)
    if np.sum(samples_per_bin) != num_samples:
        samples_per_bin[np.argmax(samples_per_bin)] -= np.sum(samples_per_bin) - num_samples

    # lastly, get confidence values over all bins
    # for this purpose, use a truncated normal distribution in each bin
    confidences = []
    for i in range(n_bins):

        # use some more magic numbers to rescale confidence estimates in order to get the desired calibration error
        loc = bin_median[i]
        scale = 100 * calibration_error / n_bins

        # bounds of truncated normal
        a = (np.clip(bounds[i] - calibration_error, 0., 1.) - loc) / scale
        b = (np.clip(bounds[i + 1] + calibration_error, 0., 1.) - loc) / scale

        confidences.append(truncnorm.rvs(a, b, loc=loc, scale=scale, size=samples_per_bin[i]))

    # rescale confidences according to the "scale PDF" obtained beforehand
    confidences = np.concatenate(confidences)
    confidences = confidences * conf_scaling

    return samples, matched, confidences


if __name__ == '__main__':

    num_samples = 10000

    # those values can be set to control the strength of miscalibration
    # however, you can also tweak the remaining "magic numbers" in the function call above
    commanded_calibration_error = 0.01
    commanded_accuracy = 0.7

    mean = np.array([.5, .5])

    # you can use either uncorrelated or correlated covariance matrices
    # correlated covariance matrices are more interesting in our case
    cov_uncorrelated = np.array([[.01, 0.], [0., .01]])
    cov_correlated = np.array([[.09, 0.06], [0.06, .09]])

    #cov = cov_uncorrelated
    cov = cov_correlated

    # set seed for reproducibility
    seed = None
    np.random.seed(seed)

    samples, matched, confidences = create_bivariate_normal(num_samples=num_samples, mean=mean, cov=cov,
                                                            accuracy=commanded_accuracy,
                                                            calibration_error=commanded_calibration_error)

    # save in NumPy format
    with open("artificial_dataset.npz", "wb") as open_file:
        np.savez_compressed(open_file, matched=matched, confidences=confidences, cx=samples[..., 0], cy=samples[..., 1])
