# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Binning methods for confidence calibration.
This package consists of several methods for confidence calibration which use binning to approximate
confidence estimates to the measured accuracy.

Available classes
=================

.. autosummary::
   :toctree: _autosummary_binning
   :template: custom_class.rst

   HistogramBinning
   IsotonicRegression
   BBQ
   ENIR
"""

from .HistogramBinning import HistogramBinning
from .BBQ import BBQ
from .ENIR import ENIR
from .IsotonicRegression import IsotonicRegression
from .NearIsotonicRegression import NearIsotonicRegression
