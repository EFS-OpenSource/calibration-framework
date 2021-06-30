# Copyright (C) 2019-2021 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Elektronische Fahrwerkssysteme, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import contextlib
import numpy as np
import sys
import torch


@contextlib.contextmanager
def manual_seed(seed: int = None):
    """ Context manager to temporally set a fixed RNG seed for NumPy and PyTorch. """

    # store old states
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        deterministic = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark

    # set new states in the current context
    try:
        if seed is not None:

            if has_cuda:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            torch.random.manual_seed(seed)
            np.random.seed(seed)

        yield

    # restore old states after leaving the current context
    finally:
        if seed is not None:

            if has_cuda:
                torch.backends.cudnn.deterministic = deterministic
                torch.backends.cudnn.benchmark = benchmark

            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)


@contextlib.contextmanager
def redirect(fileHandle):
    """ Redirect std output to a logfile given with fileHandle parameter. """

    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = fileHandle, fileHandle
    try:
        yield fileHandle
    finally:
        sys.stdout, sys.stderr = out, err
