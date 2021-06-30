# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Methods for the visualization of miscalibration. This package consists of a Reliability Diagram method.
This method bins similar to the ACE or ECE the samples in equally sized bins by their confidence and
displays the gap between confidence and observed accuracy in each bin.

Available classes
=================

.. autosummary::
   :toctree: _autosummary_presentation
   :template: custom_class.rst

   ReliabilityDiagram
"""


from .ReliabilityDiagram import ReliabilityDiagram

