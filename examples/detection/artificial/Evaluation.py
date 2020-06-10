# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Use this file to evaluate the calibration results on an artificial dataset to
demonstrate the effect of position-dependent confidence calibration used for object detection calibration.
"""

import numpy as np
from typing import Union
from matplotlib import pyplot as plt

from netcal.metrics import ACE, ECE, MCE
from netcal.presentation import ReliabilityDiagram


def read():
    """
    Read calibrated data set from NumPy archive.
    Returns
    -------
    dict
        Test set of artificial data set with default confidence estimates and calibrated ones.
    """

    with open("calibrated_dataset.npz", "rb") as open_file:
        npz = np.load(open_file)
        data = dict(npz)

    return data


def measure_miscalibration(bins: Union[tuple, list, int], data: dict, methods0d: list, methods2d: list):
    """
    Measure miscalibration and write to stdout.

    Parameters
    ----------
    bins : iterable or int
        Number of bins used by ACE, ECE and MCE.
    data : dict
        Dictionary of calibration data.
    methods0d : list
        List with strings containing the keys for the calibration data (confidence only methods).
    methods2d : list
        List with strings containing the keys for the calibration data (2D methods).
    """

    # iterate over 0D and 2D methods
    for i, methods in enumerate([methods0d, methods2d]):

        # insert 'confidence' key to the first place in the list to keep track of default miscalibration
        if i==1:
            methods = ['confidence'] + methods0d + methods2d
        else:
            methods = ['confidence'] + methods

        # on confidence only, use one single value (the first one)
        bins = bins[0] if i == 0 and isinstance(bins, (tuple, list)) else bins

        # create instances for measuring miscalibration
        ace = ACE(bins=bins, detection=True)
        ece = ECE(bins=bins, detection=True)
        mce = MCE(bins=bins, detection=True)

        # initialize empty lists
        ace_list = []
        ece_list = []
        mce_list = []

        # iterate over all methods
        for method in methods:
            data_input = data[method] if i==0 else np.stack((data[method], data['cx'], data['cy']), axis=1)
            ace_list.append(ace.measure(data_input, data['matched']))
            ece_list.append(ece.measure(data_input, data['matched']))
            mce_list.append(mce.measure(data_input, data['matched']))

        # output formatted ECE
        names = [len(x) for x in methods]
        buffer = max(names)

        # write out all miscalibration results in a 'pretty' manner
        for j, method in enumerate(methods):
            fill = (buffer - len(method)) * " "
            print("%s%s ACE: %.5f - ECE: %.5f - MCE: %.5f" % (method, fill, ace_list[j], ece_list[j], mce_list[j]))


def plot_results(bins: Union[tuple, list, int], data: dict, methods0d: list, methods2d: list):
    """
    Plot results as reliability diagrams (either 0D or 2D).

    Parameters
    ----------
    bins : iterable or int
        Number of bins used by ACE, ECE and MCE
    data : dict
        Dictionary of calibration data.
    methods0d : list
        List with strings containing the keys for the calibration data (confidence only methods).
    methods2d : list
        List with strings containing the keys for the calibration data (2D methods).
    """

    for i, methods in enumerate([methods0d, methods2d]):

        # insert 'confidence' key to the first place in the list to keep track of default miscalibration
        methods = ['confidence'] + methods

        # on confidence only, use one single value (the first one)
        bins = bins[0] if i == 0 and isinstance(bins, (tuple, list)) else bins

        # iterate over all calibration models and plot reliability diagram
        for method in methods:
            diagram = ReliabilityDiagram(bins, detection=True, title_suffix=method)
            fig = diagram.plot(data[method], data['matched'])

        # --------------------------------------------
        # second, plot 2D reliability diagrams as heatmaps
        for method in methods:
            data_input = np.stack((data[method], data['cx'], data['cy']), axis=1)

            diagram = ReliabilityDiagram(bins, detection=True, feature_names=['cx', 'cy'],
                                         fmin=0.0, fmax=0.3, title_suffix=method)
            fig = diagram.plot(data_input, data['matched'])

    plt.show()


if __name__ == '__main__':

    bins = 15
    methods0d = ["hist", "betacal", "lr_calibration"]
    methods2d = ["hist2d", "betacal2d", "betacal_dependent2d", "lr_calibration2d", "lr_calibration_dependent2d"]

    data = read()
    measure_miscalibration(bins, data, methods0d, methods2d)
    # plot_results(bins, data, methods0d, methods2d)
