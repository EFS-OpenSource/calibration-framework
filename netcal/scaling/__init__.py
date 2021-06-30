# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Scaling methods for confidence calibration.
This package consists of several methods for confidence calibration which use confidence scaling to approximate
confidence estimates to observed accuracy.

Available classes
=================

.. autosummary::
   :toctree: _autosummary_scaling
   :template: custom_class.rst

   AbstractLogisticRegression
   LogisticCalibration
   LogisticCalibrationDependent
   TemperatureScaling
   BetaCalibration
   BetaCalibrationDependent
"""

from .AbstractLogisticRegression import AbstractLogisticRegression

from .LogisticCalibration import LogisticCalibration
from .LogisticCalibrationDependent import LogisticCalibrationDependent
from .TemperatureScaling import TemperatureScaling
from .BetaCalibration import BetaCalibration
from .BetaCalibrationDependent import BetaCalibrationDependent
