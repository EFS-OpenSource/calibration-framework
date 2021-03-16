"""
Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
AND Visteon Electronics Germany GmbH, Kerpen, Germany

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="netcal",
    version="1.1.3",
    author="Fabian Kueppers",
    author_email="fabian.kueppers@hs-ruhrwest.de",
    description="Python Framework to calibrate confidence estimates of classifiers like Neural Networks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/fabiankueppers/calibration-framework",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.17', 'scipy>=1.3', 'matplotlib>=3.1', 'scikit-learn>=0.20.0', 'torch>=1.1', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
