# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from calibration.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from calibration.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration

from examples.utils import single_example, cross_validation_5_2


def example_calibration(datafile: str) -> int:
    """
    Example of several calibration methods.

    Parameters
    ----------
    datafile : str
        Path to datafile which contains two NumPy arrays with keys 'ground_truth' and 'predictions'.

    Returns
    -------
    int
        0 at success, -1 otherwise.
    """

    bins = 10

    # diagram = None
    diagram = 'diagram'

    # define validation split for test data
    validation_split = 0.7

    # if True, a Pickle-Object will be written out for each calibration model built
    save_models = False

    histogram = HistogramBinning(bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()
    lr_calibration = LogisticCalibration()
    temperature = TemperatureScaling()
    betacal = BetaCalibration()

    models = [("Histogram Binning", histogram),
              ("Isotonic Regression", iso),
              ("BBQ", bbq),
              ("ENIR", enir),
              ("Logistic Calibration", lr_calibration),
              ("Temperature Scaling", temperature),
              ("Beta Calibration", betacal)]

    # see ../utils.py for calibration and its measurement
    success = single_example(models=models, datafile=datafile, bins=bins,
                             diagram=diagram, validation_split=validation_split,
                             save_models=save_models)

    return success


def cross_validation(datafile: str) -> int:
    """
    5x2 cross validation of several calibration methods.

    Parameters
    ----------
    datafile : str
        Path to datafile which contains two NumPy arrays with keys 'ground_truth' and 'predictions'.

    Returns
    -------
    int
        0 at success, -1 otherwise.
    """

    bins = 10

    # if True, a Pickle-Object will be written out for each calibration model built
    save_models = False

    histogram = HistogramBinning(bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()
    lr_calibration = LogisticCalibration()
    temperature = TemperatureScaling()
    betacal = BetaCalibration()

    models = [("Histogram Binning", histogram),
              ("Isotonic Regression", iso),
              ("BBQ", bbq),
              ("ENIR", enir),
              ("Logistic Calibration", lr_calibration),
              ("Temperature Scaling", temperature),
              ("Beta Calibration", betacal)]

    # invoke cross validation function from ../utils.py
    # see ../utils.py for calibration and its measurement
    success = cross_validation_5_2(models=models, datafile=datafile, bins=bins, save_models=save_models)

    return success


if __name__ == '__main__':

    # example on CIFAR-100 with LeNet-5, DenseNet-BC-100 and WideResnet-16-4
    lenet = "lenet-5-cifar-100.npz"
    densenet = "densenet-bc-100-cifar-100.npz"
    wideresnet = "wideresnet-16-4-cifar-100.npz"

    example_calibration(lenet)
    cross_validation(lenet)

    example_calibration(densenet)
    cross_validation(densenet)

    example_calibration(wideresnet)
    cross_validation(wideresnet)
