import numpy as np


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
        # TODO: checks on w, step and out
        if out is None:
            out = np.empty(w.shape[0])
        self.no_python.call(w, step, out)
        return out

    def value(self, w):
        # TODO: checks on w
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
