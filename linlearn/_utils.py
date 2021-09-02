# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from numba import jit, void, uintp, float64
import numpy as np
from numpy.random import randint

import numba as nb

# Numba flags applied to all jit decorators
NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
FASTMATH = True


nb_float = nb.float64
np_float = np.float64


def get_type(class_):
    """Gives the numba type of an object if numba.jit decorators are enabled and None
    otherwise. This helps to get correct coverage of the code

    Parameters
    ----------
    class_ : `object`
        A class

    Returns
    -------
    output : `object`
        A numba type of None
    """
    class_type = getattr(class_, "class_type", None)
    if class_type is None:
        return lambda *_: None
    else:
        return class_type.instance_type


@jit(
    void(uintp[:], uintp[:]),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={"n_samples": uintp, "population_size": uintp, "i": uintp, "j": uintp},
)
def sample_without_replacement(pool, out):
    """Samples integers without replacement from pool into out inplace.

    Parameters
    ----------
    pool : ndarray of size population_size
        The array of integers to sample from (it containing [0, ..., n_samples-1]

    out : ndarray of size n_samples
        The sampled subsets of integer
    """
    # We sample n_samples elements from the pool
    n_samples = out.shape[0]
    population_size = pool.shape[0]
    # Initialize the pool
    for i in range(population_size):
        pool[i] = i

    for i in range(n_samples):
        j = randint(population_size - i)
        out[i] = pool[j]
        pool[j] = pool[population_size - i - 1]
