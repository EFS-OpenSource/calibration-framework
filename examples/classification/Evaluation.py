# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import pandas as pd
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

from netcal.metrics import ACE, ECE, MCE
from netcal.presentation import ReliabilityDiagram


def read(base_dir: str, network: str) -> list:
    """
    Read calibrated data set from NumPy archive.
    Returns
    -------
    list
        Test set of artificial data set with default confidence estimates and calibrated ones.
    """

    data = []
    for filename in os.listdir(base_dir):
        if filename.startswith(network):

            filename = "%s/%s" % (base_dir, filename)
            with open(filename, "rb") as open_file:
                print("Read file: %s" % filename)
                npz = np.load(open_file)
                data.append(dict(npz))

    return data


def measure(key: str, ground_truth: list, data: list, uncertainty: str, bins: int):
    """ Measure miscalibration (batched mode) """

    print("Measure: %s" % key)
    try:
        confidence = [x[key] for x in data]
    except KeyError:
        return np.nan

    ece = ECE(bins=bins, detection=False)
    miscalibration = []
    for conf, gt in zip(confidence, ground_truth):

        if conf.ndim == 3:
            if uncertainty == 'mean':
                conf = np.mean(conf, axis=0)
            elif uncertainty == 'flatten':
                gt = np.tile(gt, conf.shape[0]).flatten()
                conf = conf.flatten()
            else:
                raise AttributeError("Unknown type of uncertainty handling: %s." % uncertainty)

        miscalibration.append(ece.measure(conf, gt))

    return np.mean(miscalibration)


def measure_miscalibration(bins: int, methods: list, uncertainty: str,
                           map_data: list = None, mcmc_data: list = None, vi_data: list = None):
    """
    Measure miscalibration and write to stdout.

    Parameters
    ----------
    bins : iterable or int
        Number of bins used by ACE, ECE and MCE.
    data : list
        List with dictionaries containing calibration data.
    methods : list
        List with strings containing the keys for the calibration data (confidence only methods).
    uncertainty : str
        Type how to handle uncertainty quantification. Must be either 'mean' or 'flatten'.
    """

    map_gt = [x['test_gt'] for x in map_data]
    mcmc_gt = [x['test_gt'] for x in mcmc_data]
    vi_gt = [x['test_gt'] for x in vi_data]

    map_kwargs = {'data': map_data, 'ground_truth': map_gt, 'uncertainty': uncertainty, 'bins': bins}
    mcmc_kwargs = {'data': mcmc_data, 'ground_truth': mcmc_gt, 'uncertainty': uncertainty, 'bins': bins}
    vi_kwargs = {'data': vi_data, 'ground_truth': vi_gt, 'uncertainty': uncertainty, 'bins': bins}

    types = []

    column_baseline = []
    columns_methods = [[] for _ in methods]

    if map_data is not None:
        types.append('MAP')
        column_baseline.append(measure('test_scores', **map_kwargs))

        for i, method in enumerate(methods):
            columns_methods[i].append(measure(method, **map_kwargs))

    if mcmc_data is not None:
        types.append('MCMC')
        column_baseline.append(np.nan)

        for i, method in enumerate(methods):
            columns_methods[i].append(measure(method, **mcmc_kwargs))

    if vi_data is not None:
        types.append('VI')
        column_baseline.append(np.nan)

        for i, method in enumerate(methods):
            columns_methods[i].append(measure(method, **vi_kwargs))

    index = pd.Index(types)
    df = pd.DataFrame(data=np.stack((column_baseline, *columns_methods), axis=1), index=index)
    df.columns = ['baseline'] + methods

    print(df)
    return df


if __name__ == '__main__':

    bins = 15
    uncertainty = 'flatten'
    methods = ["hist", "lr", "temperature", "beta"]

    # network = "wideresnet-16-4-cifar-100"
    network = "densenet-bc-100-cifar-100"

    base_dir_map = "examination-map/results/"
    base_dir_mcmc = "examination-mcmc/results/"
    base_dir_vi = "examination-variational/results/"

    map_data = read(base_dir_map, network)
    mcmc_data = read(base_dir_mcmc, network)
    vi_data = read(base_dir_vi, network)

    # mcmc_data = None
    # vi_data = None

    df = measure_miscalibration(bins, methods, uncertainty, map_data, mcmc_data, vi_data)
    df.to_excel("evaluation-%s-%s.xlsx" % (uncertainty, network))
