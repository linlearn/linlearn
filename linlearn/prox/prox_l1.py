from math import fabs
from numba import jitclass
from numba.types import int64, float64, boolean
from linlearn.prox.prox import Prox
from linlearn.prox.utils import call_with_single, value_with_single


specs = [
    ('strength', float64),
    ('start', int64),
    ('end', int64),
    ('positive', boolean),
    ('has_start_end', boolean)
]
@jitclass(specs)
class ProxL1NoPython(object):

    def __init__(self, strength, start, end, has_start_end, positive):
        self.strength = strength
        self.start = start
        self.end = end
        self.has_start_end = has_start_end
        self.positive = positive

    def call_single(self, x, step):
        thresh = step * self.strength
        if x > 0:
            if x > thresh:
                return x - thresh
            else:
                return 0
        else:
            # If x is negative and we project onto the non-negative half-plane
            # we set it to 0
            if self.positive:
                return 0
            else:
                if x < -thresh:
                    return x + thresh
                else:
                    return 0

    def value_single(self, x):
        return fabs(x)

    def call(self, w, step, out):
        call_with_single(self, w, step, out)

    def value(self, w):
        return value_with_single(self, w)


class ProxL1(Prox):
    """Proximal operator of the L1 norm (Lasso penalization)

    Parameters
    ----------
    strength : `float`
        Level of L1 penalization.

    start_end : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector.

    positive : `bool`, default=`False`
        If True, apply ridge penalization together with a projection
        onto the set of vectors with non-negative entries.
    """
    def __init__(self, strength, start_end=None, positive=False):
        Prox.__init__(self, ProxL1NoPython, strength, start_end, positive)
