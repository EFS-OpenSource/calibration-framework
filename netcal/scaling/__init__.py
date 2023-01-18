# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/scaling/README.md
   :parser: myst_parser.sphinx_

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
