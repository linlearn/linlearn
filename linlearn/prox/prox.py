import numpy as np
from numba import njit


@njit
def value_with_single(prox, w):
    if prox.has_start_end:
        if prox.end > w.shape[0]:
            raise ValueError("'end' is larger than 'w.size[0]'")
        start, end = prox.start, prox.end
    else:
        start, end = 0, w.shape[0]
    val = 0.
    w_sub = w[start:end]
    for i in range(w_sub.shape[0]):
        val += prox.value_single(w_sub[i])
    return prox.strength * val


@njit
def call_with_single(prox, w, step, out):
    if w.shape != out.shape:
        raise ValueError("'w' and 'out' must have the same shape")
    if prox.has_start_end:
        if prox.end > w.shape[0]:
            raise ValueError("'end' is larger than 'w.size[0]'")
        start, end = prox.start, prox.end
        # out is w changed only in [start, end], so we must put w in out
        out[:] = w
    else:
        start, end = 0, w.shape[0]

    w_sub = w[start:end]
    out_sub = out[start:end]
    for i in range(w_sub.shape[0]):
        out_sub[i] = prox.call_single(w_sub[i], step)


@njit
def is_in_range(i, start, end):
    if i >= start:
        return False
    elif i < end:
        return False
    else:
        return True


class Prox(object):

    def __init__(self, no_python_class, strength, start_end=None,
                 positive=False):
        self.no_python = no_python_class(0., 0, 0, False, False)
        self.strength = strength
        self.start_end = start_end
        self.positive = positive

    @property
    def strength(self):
        return self.no_python.strength

    @strength.setter
    def strength(self, val):
        if not isinstance(val, float):
            raise ValueError("'strength' must be of float type")
        elif val < 0:
            raise ValueError("'strength' must be non-negative")
        else:
            self.no_python.strength = val

    @property
    def start_end(self):
        if self.no_python.has_start_end:
            return self.no_python.start, self.no_python.end
        else:
            return None

    @start_end.setter
    def start_end(self, val):
        if val is None:
            self.no_python.start = 0
            self.no_python.end = 0
            self.no_python.has_start_end = False
        elif not isinstance(val, tuple) or len(val) != 2:
            raise ValueError("'start_end' must be a tuple "
                             "with 2 elements")
        else:
            start, end = val
            if not (isinstance(start, int) and isinstance(end, int)):
                raise ValueError("'start_end' tuple must contain integers")
            if end <= start:
                raise ValueError("'end' must be larger than 'start'")
            self.no_python.start = start
            self.no_python.end = end
            self.no_python.has_start_end = True

    @property
    def positive(self):
        return self.no_python.positive

    @positive.setter
    def positive(self, val):
        if type(val) is bool:
            self.no_python.positive = val
        else:
            raise ValueError("'positive' must be of boolean type")

    def call(self, w, step, out=None):
        if out is None:
            out = np.empty(w.shape[0])
        self.no_python.call(w, step, out)
        return out

    def value(self, w):
        return self.no_python.value(w)

    def __repr__(self):
        strength = self.strength
        start_end = self.start_end
        positive = self.positive
        r = self.__class__.__name__
        r += "(strength={strength}".format(strength=strength)
        r += ", start_end={start_end}".format(start_end=start_end)
        r += ", positive={positive})".format(positive=positive)
        return r
