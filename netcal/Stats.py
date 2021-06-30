# Copyright (C) 2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.
#
# Parts of this file have been adapted from NumPyro: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/diagnostics.py

import numpy as np


def hpdi(x, prob=0.90, axis=0):
    """
    Computes "highest posterior density interval" (HPDI) which is the narrowest
    interval with probability mass ``prob``. This method has been adapted from NumPyro:
    `Find NumPyro original implementation <https://github.com/pyro-ppl/numpyro/blob/v0.2.4/numpyro/diagnostics.py#L191>_`.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    prob : float, optional, default: 0.9
        Probability mass of samples within the interval.
    axis : int, optional, default: 0
        The dimension to calculate hpdi.

    Returns
    -------
    np.ndarray
        Quantiles of ``x`` at ``(1 - prob) / 2`` and ``(1 + prob) / 2``.
    """
    x = np.swapaxes(x, axis, 0)
    sorted_x = np.sort(x, axis=0)
    mass = x.shape[0]
    index_length = int(prob * mass)
    intervals_left = sorted_x[:(mass - index_length)]
    intervals_right = sorted_x[index_length:]
    intervals_length = intervals_right - intervals_left
    index_start = intervals_length.argmin(axis=0)
    index_end = index_start + index_length
    hpd_left = np.take_along_axis(sorted_x, index_start[None, ...], axis=0)
    hpd_left = np.swapaxes(hpd_left, axis, 0)
    hpd_right = np.take_along_axis(sorted_x, index_end[None, ...], axis=0)
    hpd_right = np.swapaxes(hpd_right, axis, 0)

    return np.concatenate([hpd_left, hpd_right], axis=axis)
