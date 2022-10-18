import numpy as np
from scipy.stats import norm
from scipy.special import expit


def generate_mean_dependent(n_samples):
    """
    Generate samples of an artificial base estimator that are consistently overconfident.
    Introduce a dependency on the mean.
    """

    # step 1: generate some y ground-truth and add a hypothetical regressor that fits its mean predictions
    # to the ground-truth
    x = np.linspace(-5, 5, n_samples)
    y = np.cos(x)
    ymean = np.array(y)

    # step 2: corrupt the ground-truth with gaussian noise based on the mean estimate
    y = norm.rvs(loc=y, scale=np.ones_like(y) * expit(y*1.5) * 1.5 + .125)

    # step 3: sample the stddev that is also predicted by a hypothetical regressor that however consistently
    # underestimates the true variance
    ystd = np.random.uniform(low=0.25, high=0.5, size=(n_samples,))

    return x, y, ymean, ystd


def generate_variance_dependent(n_samples):
    """
    Generate samples of an artificial base estimator that are consistently overconfident.
    Introduce a dependency on the variance.
    """

    # step 1: generate some y ground-truth and add a hypothetical regressor that fits its mean predictions
    # to the ground-truth
    x = np.linspace(-5, 5, n_samples)
    y = np.cos(x)
    ymean = np.array(y)

    # step 2: sample the stddev that is predicted by a hypothetical regressor that however consistently
    # underestimates the true variance
    ystd = np.random.uniform(low=0.1, high=0.25, size=(n_samples,))

    # step 3: corrupt the ground-truth with gaussian noise that bases on the predicted variance
    y = norm.rvs(loc=y, scale=np.ones_like(y) * expit((ystd-0.175) * 2.) * .5 + .25)

    return x, y, ymean, ystd
