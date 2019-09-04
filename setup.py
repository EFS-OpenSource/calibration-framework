"""
Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
AND Visteon Electronics Germany GmbH, Kerpen, Germany

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calibration",
    version="1.0",
    author="Fabian Kueppers",
    author_email="fabian.kueppers@hs-ruhrwest.de",
    description="Python Framework to calibrate confidence estimates of classifiers like Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabiankueppers/calibration-framework",
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'sklearn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
