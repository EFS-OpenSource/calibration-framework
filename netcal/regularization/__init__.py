# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Regularization methods which are applied during model training. These methods should achieve a
confidence calibration during model training. For example, the Confidence Penalty
penalizes confident predictions and prohibits over-confident estimates.
Use the functions to obtain the regularization and callback instances with prebuild parameters.

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
