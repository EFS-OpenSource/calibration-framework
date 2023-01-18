# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/metrics/regression/README.md
   :parser: myst_parser.sphinx_

Available classes
=================

.. autosummary::
   :toctree: _autosummary_metric
   :template: custom_class.rst

   QuantileLoss
   PinballLoss
   ENCE
   UCE
   PICP
   QCE
   NLL
"""

from .QuantileLoss import QuantileLoss
from .QuantileLoss import PinballLoss
from .ENCE import ENCE
from .UCE import UCE
from .PICP import PICP
from .QCE import QCE
from .NLL import NLL
