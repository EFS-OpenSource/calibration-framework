# Copyright (C) 2019-2022 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
API Reference of net:cal
========================
This is the detailed API reference for the net:cal calibration framework. This library can be used to
obtain well-calibrated confidence or uncertainty estimates from biased estimators such as neural networks.
The API reference contains a detailed description of all available methods and their parameters. For
miscellaneous examples on how to use these methods, see readme below.

Available packages
------------------

.. autosummary::
   :toctree: _autosummary

   binning
   scaling
   regularization
   metrics
   presentation
   regression


Each calibration method must inherit the base class :class:`AbstractCalibration`. If you want to write your own method and
include into the framework, include this class as the base class.

Base class
----------

.. autosummary::
   :toctree: _autosummary_abstract_calibration
   :template: custom_class.rst

   AbstractCalibration

"""

name = 'netcal'
__version__ = '1.3.3'

from .AbstractCalibration import AbstractCalibration
from .Decorator import accepts, dimensions, global_accepts, global_dimensions
from .Context import manual_seed, redirect
from .Stats import hpdi, mv_cauchy_log_density
from .Helper import squeeze_generic, meanvar, cumulative, is_in_quantile, cumulative_moments, density_from_cumulative
