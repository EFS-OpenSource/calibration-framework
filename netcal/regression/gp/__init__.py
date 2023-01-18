# Copyright (C) 2021-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/regression/gp/README.md
   :parser: myst_parser.sphinx_

Available classes
=================

.. autosummary::
   :toctree: _autosummary_regression_gp
   :template: custom_class.rst

   AbstractGP
   GPBeta
   GPNormal
   GPCauchy

Packages
========

.. autosummary::
   :toctree: _autosummary_regression_gp_submodules

   kernel
   likelihood
"""

from .AbstractGP import AbstractGP
from .GPBeta import GPBeta
from .GPNormal import GPNormal
from .GPCauchy import GPCauchy
