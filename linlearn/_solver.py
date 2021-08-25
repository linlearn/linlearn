import numpy as np
from numba import boolean
from numba.experimental import jitclass

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FLOAT, NP_FLOAT


#
# Coordinate gradient descent CGD
#


spec_state_cgd = [
    ("inner_products", FLOAT[::1]),
    ("fit_intercept", boolean)
]


@jitclass(spec_state_cgd)
class StateCGD(object):
    def __init__(self, n_samples, fit_intercept=True):
        self.inner_products = np.zeros((n_samples,), dtype=NP_FLOAT)
        self.fit_intercept = fit_intercept
