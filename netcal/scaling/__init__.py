# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Scaling methods for confidence calibration.
This package consists of several methods for confidence calibration which use confidence scaling to approximate
confidence estimates to observed accuracy.

Available classes
=================

.. autosummary::
   :toctree: _autosummary_scaling
   :template: custom_class.rst

   LogisticCalibration
   TemperatureScaling
   BetaCalibration
"""


from .LogisticCalibration import LogisticCalibration
from .TemperatureScaling import TemperatureScaling
from .BetaCalibration import BetaCalibration
