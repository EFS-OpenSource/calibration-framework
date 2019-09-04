# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from calibration import accepts
from calibration.scaling import LogisticCalibration


class TemperatureScaling(LogisticCalibration):
    """
    Perform Temperature scaling to logits of NN.
    The calibrated probability :math:`\\hat{q}` is computed by

    .. math::

       \\hat{q} = \\sigma_{\\text{SM}} (z / T)

    with :math:`\\sigma_{\\text{SM}}` as the softmax operator (or the sigmoid alternatively),
    :math:`z` as the logits and :math:`T` as the temperature estimated by logistic regression.
    This leds to calibrated confidence estimates.

    Parameters
    ----------
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    References
    ----------
    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger:
    "On Calibration of Modern Neural Networks."
    arXiv (abs/1706.04599), 2017.
    `Get source online <https://arxiv.org/abs/1706.04599>`_
    """

    @accepts(bool)
    def __init__(self, independent_probabilities: bool = False):
        """
        Constructor.

        Parameters
        ----------
        independent_probabilities : bool, default=False
            boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        super().__init__(temperature_only=True, independent_probabilities=independent_probabilities)
