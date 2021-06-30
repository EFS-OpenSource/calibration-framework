# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Union

from netcal.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration

from examples.classification import single_example, cross_validation_5_2


def example_calibration(datafile: str, domain: str = ".") -> int:
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

    # kwargs for uncertainty mode. Those can also be safely set on MLE
    uncertainty_kwargs = {'mcmc_chains': 1,
                          'mcmc_samples': 300,
                          'mcmc_warmup_steps': 50,
                          'vi_samples': 300,
                          'vi_epochs': 3000}

    if domain == 'examination-mcmc':
        method = 'mcmc'
    elif domain == 'examination-variational':
        method = 'variational'
    else:
        method = 'mle'

    bins = 15
    hist_bins = 20

    # diagram = None
    diagram = 'diagram'

    # define validation split for test data
    validation_split = 0.7

    # if True, a Pickle-Object will be written out for each calibration model built
    save_models = False

    histogram = HistogramBinning(hist_bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()

    lr_calibration = LogisticCalibration(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)
    temperature = TemperatureScaling(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)
    betacal = BetaCalibration(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)

    models = [("hist", histogram),
              ("iso", iso),
              ("bbq", bbq),
              ("enir", enir),
              ("lr", lr_calibration),
              ("temperature", temperature),
              ("beta", betacal)]

    # see ../utils.py for calibration and its measurement
    success = single_example(models=models, datafile=datafile, bins=bins,
                             diagram=diagram, validation_split=validation_split,
                             save_models=save_models, domain=domain)

    return success


def cross_validation(datafile: str, use_cuda: Union[bool, str] = False, domain: str = ".") -> int:
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

    # kwargs for uncertainty mode. Those can also be safely set on MLE
    uncertainty_kwargs = {'mcmc_chains': 1,
                          'mcmc_samples': 300,
                          'mcmc_warmup_steps': 50,
                          'vi_samples': 300,
                          'vi_epochs': 3000}

    hist_bins = 20
    bins = 15

    if domain == 'examination-mcmc':
        method = 'mcmc'
    elif domain == 'examination-variational':
        method = 'variational'
    else:
        method = 'mle'

    # if True, a Pickle-Object will be written out for each calibration model built
    save_models = True

    histogram = HistogramBinning(hist_bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()
    lr_calibration = LogisticCalibration(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)
    temperature = TemperatureScaling(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)
    betacal = BetaCalibration(detection=False, method=method, use_cuda=use_cuda, **uncertainty_kwargs)

    models = [("hist", histogram),
              ("iso", iso),
              ("bbq", bbq),
              ("enir", enir),
              ("lr", lr_calibration),
              ("temperature", temperature),
              ("beta", betacal)]

    # invoke cross validation function from ../utils.py
    # see ../utils.py for calibration and its measurement
    success = cross_validation_5_2(models=models, datafile=datafile, bins=bins, save_models=save_models, domain=domain)

    return success


if __name__ == '__main__':

    use_cuda = 'cuda:0'

    # domain = "examination-map"
    # domain = "examination-mcmc"
    domain = "examination-variational"

    # example on CIFAR-10 with LeNet-5 and WideResnet-16-4
    lenet = "records/cifar10/lenet-5-cifar-10.npz"
    wideresnet = "records/cifar10/wideresnet-16-4-cifar-10.npz"

    cifar10 = [lenet, wideresnet]

    # for each model, perform a single example and a 5x2 cross validation
    for model in cifar10:
       example_calibration(model)
       cross_validation(model)

    # example on CIFAR-100 with LeNet-5, DenseNet-BC-100 and WideResnet-16-4
    lenet = "records/cifar100/lenet-5-cifar-100.npz"
    densenet = "records/cifar100/densenet-bc-100-cifar-100.npz"
    wideresnet = "records/cifar100/wideresnet-16-4-cifar-100.npz"

    cifar100 = [lenet, densenet, wideresnet]

    # for each model, perform a single example and a 5x2 cross validation
    for model in cifar100:
        example_calibration(model)
        cross_validation(model, use_cuda=use_cuda, domain=domain)
