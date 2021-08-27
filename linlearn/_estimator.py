# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from collections import namedtuple
import numpy as np
from numpy.random import shuffle
from numba import jit, uintp
from numba.experimental import jitclass

from ._loss import partial_deriv
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FLOAT, NP_FLOAT


Estimator = namedtuple("Estimator", ["state", "partial_deriv"])


################################################################
# Empirical risk minimizer (ERM)
################################################################


@jitclass([])
class StateERM(object):
    """Nothing happens here, but we need some empty shell to pass to the methods
    """

    def __init__(self):
        pass


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def partial_deriv_erm(deriv_loss, state_loss, j, X, y, state_solver, state_erm):
    deriv_sum = 0.0
    fit_intercept = state_solver.fit_intercept
    inner_products = state_solver.inner_products
    n_samples = inner_products.shape[0]
    for i in range(n_samples):
        deriv_sum += partial_deriv(
            deriv_loss, state_loss, j, y[i], inner_products[i], X[i], fit_intercept
        )
    return deriv_sum / n_samples


################################################################
# Median of means estimator (MOM)
################################################################


spec_state_mom = [
    ("block_size", uintp),
    ("n_blocks", uintp),
    ("last_block_size", uintp),
    ("block_means", FLOAT[::1]),
    ("sample_indices", uintp[::1]),
]


@jitclass(spec_state_mom)
class StateMOM(object):
    def __init__(self, n_samples, block_size):
        self.block_size = block_size
        self.n_blocks = n_samples // block_size
        self.last_block_size = n_samples % block_size
        if self.last_block_size > 0:
            self.n_blocks += 1
        self.block_means = np.empty(self.n_blocks, dtype=NP_FLOAT)
        self.sample_indices = np.empty(n_samples, dtype=np.uintp)


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def partial_deriv_mom(deriv_loss, state_loss, j, X, y, state_solver, state_mom):
    """Computation of the derivative of the loss with respect to a coordinate using the
    median of means (mom) stategy."""
    # TODO: parallel ?
    # TODO: sparse matrix ?
    fit_intercept = state_solver.fit_intercept
    inner_products = state_solver.inner_products
    n_samples = inner_products.shape[0]
    block_size = state_mom.block_size
    n_blocks = state_mom.n_blocks
    last_block_size = state_mom.last_block_size
    sample_indices = state_mom.sample_indices
    block_means = state_mom.block_means
    # TODO: test / benchmark a handmade Fisher Yates with preallocated pool
    # Shuffle the sample indices to get different blocks each time
    for i in range(n_samples):
        sample_indices[i] = i
    shuffle(sample_indices)
    # Cumulative sum in the block
    derivatives_sum_block = 0.0
    # Block counter
    n_block = 0
    for i in sample_indices:
        derivatives_sum_block += partial_deriv(
            deriv_loss, state_loss, j, y[i], inner_products[i], X[i], fit_intercept
        )
        if (i != 0) and ((i + 1) % block_size == 0):
            block_means[n_block] = derivatives_sum_block / block_size
            n_block += 1
            derivatives_sum_block = 0.0
    if last_block_size != 0:
        block_means[n_blocks] = derivatives_sum_block / last_block_size
    return np.median(block_means)


################################################################
# Trimmed means estimator (TMEAN)
################################################################


spec_state_trim = [
    ("n_samples", uintp),
    ("percentage", FLOAT),
    ("n_excluded", uintp),
    ("derivs", FLOAT[::1]),
]


@jitclass(spec_state_trim)
class StateTrim(object):
    def __init__(self, n_samples, percentage=0.1):
        self.n_samples = n_samples
        self.percentage = percentage
        self.n_excluded = int(n_samples * percentage / 2)
        self.derivs = np.empty(n_samples, dtype=NP_FLOAT)


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def partial_deriv_trim(loss_deriv, j, X, y, state_solver, state_trim):
    fit_intercept = state_solver.fit_intercept
    inner_products = state_solver.inner_products
    derivs = state_trim.derivs
    n_excluded = state_trim.n_excluded
    n_samples = inner_products.shape[0]
    for i in range(n_samples):
        derivs[i] = partial_deriv(
            loss_deriv, j, y[i], inner_products[i], X[i], fit_intercept
        )
    derivs.sort()
    return np.mean(derivs[n_excluded:-n_excluded])


def get_estimator(estimator, **kwargs):
    if estimator == "erm":
        return Estimator(state=StateERM(**kwargs), partial_deriv=partial_deriv_erm)
    elif estimator == "mom":
        return Estimator(state=StateMOM(**kwargs), partial_deriv=partial_deriv_mom)
    elif estimator == "trim":
        return Estimator(state=StateTrim(**kwargs), partial_deriv=partial_deriv_trim)
    else:
        raise ValueError()
