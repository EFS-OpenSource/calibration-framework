"""
Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
AND Visteon Electronics Germany GmbH, Kerpen, Germany

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


import numpy as np
from functools import wraps


def accepts(*types):
    """
    Decorator for function arg check
    """
    def check_accepts(f):
        assert len(types)+1 == f.__code__.co_argcount

        @wraps(f)
        def new_f(*args, **kwds):
            for i, (a, t) in enumerate(zip(args[1:], types), start=1):
                if t is None:
                    continue

                if type(t) == tuple:
                    for st in t:
                        if type(a) == st:
                            break
                    else:
                        raise AssertionError("arg \'%s\' does not match one of types %s" % (f.__code__.co_varnames[i], str(t)))
                else:
                    assert isinstance(a, t), "arg \'%s\' does not match %s" % (f.__code__.co_varnames[i],t)
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def dimensions(*dim):
    """
    Decorator for numpy array dimension check
    """
    def check_dim(f):
        assert len(dim)+1 == f.__code__.co_argcount

        @wraps(f)
        def new_f(*args, **kwds):

            for i, (a, d) in enumerate(zip(args[1:], dim), start=1):
                if d is None:
                    continue

                assert isinstance(a, np.ndarray), "arg \'%s\' does not match %s" % (f.__code__.co_varnames[i], np.ndarray)

                if type(d) == tuple:
                    assert len(a.shape) in d, "dimension of arg \'%s\' must match %s but is %d" % (f.__code__.co_varnames[i], str(d), len(a.shape))
                elif type(d) == int:
                    assert len(a.shape) == d, "dimension of arg \'%s\' must match %s but is %d" % (f.__code__.co_varnames[i], str(d), len(a.shape))
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f

    return check_dim


def global_accepts(*types):
    """
    Decorator for global function's arg check
    """
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount

        @wraps(f)
        def new_f(*args, **kwds):
            for i, (a, t) in enumerate(zip(args, types)):
                if t is None:
                    continue

                if type(t) == tuple:
                    for st in t:
                        if type(a) == st:
                            break
                    else:
                        raise AssertionError("arg \'%s\' does not match one of types %s" % (f.__code__.co_varnames[i], str(t)))
                else:
                    assert isinstance(a, t), "arg \'%s\' does not match %s" % (f.__code__.co_varnames[i],t)
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def global_dimensions(*dim):
    """
    Decorator for global function's numpy array dimension check
    """
    def check_dim(f):
        assert len(dim) == f.__code__.co_argcount

        @wraps(f)
        def new_f(*args, **kwds):

            for i, (a, d) in enumerate(zip(args, dim)):
                if d is None:
                    continue

                assert isinstance(a, np.ndarray), "arg \'%s\' does not match %s" % (f.__code__.co_varnames[i], np.ndarray)

                if type(d) == tuple:
                    assert len(a.shape) in d, "dimension of arg \'%s\' must match %s but is %d" % (f.__code__.co_varnames[i], str(d), len(a.shape))
                elif type(d) == int:
                    assert len(a.shape) == d, "dimension of arg \'%s\' must match %s but is %d" % (f.__code__.co_varnames[i], str(d), len(a.shape))
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f

    return check_dim
