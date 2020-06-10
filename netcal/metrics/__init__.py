# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Methods for measuring miscalibration. The common methods are given with the
'Average Calibration Error (ACE)', 'Expected Calibration Error (ECE)' and 'Maximum Calibration Error (MCE)'.
Each methods bins the samples by their confidence and measures the accuracy in each bin. The ECE gives the
mean gap between confidence and observed accuracy in each bin weighted by the number of samples.
The MCE returns the highest observed deviation. The ACE is similar to the ECE but weights each bin equally.

Available classes
=================

.. autosummary::
   :toctree: _autosummary_metric
   :template: custom_class.rst

   ACE
   ECE
   MCE
"""

from .ACE import ACE
from .ECE import ECE
from .MCE import MCE
from .Miscalibration import _Miscalibration