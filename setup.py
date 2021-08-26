# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerksysteme GmbH, Gaimersheim Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="netcal",
    version="1.2.1",
    author="Fabian Kueppers",
    author_email="fabian.kueppers@hs-ruhrwest.de",
    description="Python Framework to calibrate confidence estimates of classifiers like Neural Networks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/fabiankueppers/calibration-framework",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.17', 'scipy>=1.3', 'matplotlib>=3.1', 'scikit-learn>=0.21', 'torch>=1.4', 'torchvision>=0.5.0', 'tqdm>=4.40', 'pyro-ppl>=1.3', 'tikzplotlib>=0.9.8', 'tensorboard>=2.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
