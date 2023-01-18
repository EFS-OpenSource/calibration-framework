# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/regression/README.md
   :parser: myst_parser.sphinx_

Available classes
=================

.. autosummary::
   :toctree: _autosummary_regression
   :template: custom_class.rst

   IsotonicRegression
   VarianceScaling
   GPBeta
   GPNormal
   GPCauchy

Package for Gaussian process optimization
=========================================

.. autosummary::
   :toctree: _autosummary_regression_gp

   gp
"""

from .IsotonicRegression import IsotonicRegression
from .VarianceScaling import VarianceScaling

from .gp.GPBeta import GPBeta
from .gp.GPNormal import GPNormal
from .gp.GPCauchy import GPCauchy
