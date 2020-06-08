# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
"""


from .ConfidencePenalty import confidence_penalty
