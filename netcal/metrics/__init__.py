# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/metrics/README.md
   :parser: myst_parser.sphinx_

Available classes
=================

.. autosummary::
   :toctree: _autosummary_metric
   :template: custom_class.rst

   ACE
   ECE
   MCE
   MMCE
   QuantileLoss
   PinballLoss
   ENCE
   UCE
   PICP
   QCE
   NLL

Packages
========

.. autosummary::
   :toctree: _autosummary_metric_submodules

   confidence
   regression
"""

from .confidence.ACE import ACE
from .confidence.ECE import ECE
from .confidence.MCE import MCE
from .confidence.MMCE import MMCE

from .regression.QuantileLoss import QuantileLoss
from .regression.QuantileLoss import PinballLoss
from .regression.ENCE import ENCE
from .regression.UCE import UCE
from .regression.PICP import PICP
from .regression.QCE import QCE
from .regression.NLL import NLL
