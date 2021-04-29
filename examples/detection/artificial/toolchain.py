# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Use this file to demonstrate the effect of position-dependent confidence calibration used for object
detection calibration on an artificial dataset.
"""

import numpy as np
from examples.detection.artificial import create_bivariate_normal
from examples.detection.artificial import calibration
from examples.detection.artificial import measure_miscalibration, plot_results


def chain():
    """
    Grouped function call to all evaluation routines: from creating the artificial dataset over calibration
    and visualization.
    """

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

    calibration_bins = [15, 15, 15]
    measure_bins = [12, 12, 12]
    save_models = True

    # specify methods that should be evaluated
    methods0d = ["hist", "betacal", "lr_calibration"]
    methods2d = ["hist2d", "betacal2d", "betacal_dependent2d", "lr_calibration2d", "lr_calibration_dependent2d"]

    # first step: create data
    coordinates, matched, confidences = create_bivariate_normal(num_samples=num_samples, mean=mean, cov=cov,
                                                                accuracy=commanded_accuracy,
                                                                calibration_error=commanded_calibration_error)

    # collect data in a single dict
    samples = {'matched': matched, 'confidences': confidences, 'cx': coordinates[..., 0], 'cy': coordinates[..., 1]}

    # second step: perform calibration
    results = calibration(samples, calibration_bins, save_models, seed)

    # last step: measure and visualize miscalibration
    measure_miscalibration(measure_bins, results, methods0d, methods2d)
    plot_results(measure_bins, results, methods0d, methods2d)


if __name__ == '__main__':
    chain()
