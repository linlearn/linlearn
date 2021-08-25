import numpy as np
from numba.experimental import jitclass

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FLOAT, NP_FLOAT


#
# Coordinate gradient descent CGD
#


spec_state_cgd = [
    ("inner_products", FLOAT[::1]),
]


@jitclass(spec_state_cgd)
class StateCGD(object):
    def __init__(self, n_samples):
        self.inner_products = np.zeros((n_samples,), dtype=NP_FLOAT)
