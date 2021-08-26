# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
API Reference
=============
This is the detailled API reference for the confidence calibration framework. This framework can be used to
obtain well-calibrated confidence estimates from biased estimators like Neural Networks.
The API reference contains a detailled description of all available methods and their parameters. For
miscellaneous examples on how to use these methods, see readme below.

Available packages

.. autosummary::
   :toctree: _autosummary

   binning
   scaling
   regularization
   metrics
   presentation


Each calibration method must inherit the base class :class:`AbstractCalibration`. If you want to write your own method and
include into the framework, include this class as the base class.

.. autosummary::
   :toctree: _autosummary_abstract_calibration
   :template: custom_class.rst

   AbstractCalibration

"""

name = 'calibration'
__version__ = '1.2.1'

from .AbstractCalibration import AbstractCalibration
from .Decorator import accepts, dimensions, global_accepts, global_dimensions
from .Context import manual_seed, redirect
from .Stats import hpdi
