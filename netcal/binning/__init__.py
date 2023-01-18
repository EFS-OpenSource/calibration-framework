# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/binning/README.md
   :parser: myst_parser.sphinx_

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
