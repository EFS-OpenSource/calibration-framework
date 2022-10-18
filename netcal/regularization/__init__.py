# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/regularization/README.md
   :parser: myst_parser.sphinx_

Available functions
===================

.. autosummary::
   :toctree: _autosummary_regularization_func

   confidence_penalty
   ConfidencePenalty
   MMCEPenalty
   DCAPenalty
"""


from .ConfidencePenalty import confidence_penalty
from .ConfidencePenalty import ConfidencePenalty
from .MMCEPenalty import MMCEPenalty
from .DCAPenalty import DCAPenalty
