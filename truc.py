from collections import namedtuple
from numba import njit, uint32, float64
from numba.experimental import jitclass
import numpy as np


# Estimator = namedtuple("Estimator", ["partial_deriv", "state"])


spec_state_mom = [
    ("n_blocks", uint32),
    ("block_means", float64[::1]),
]


@jitclass(spec_state_mom)
class StateMOM(object):
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.block_means = np.zeros((n_blocks,), dtype=np.float64)


@jitclass([])
class StateERM(object):
    def __init__(self):
        pass


@njit
def partial_deriv_least_squares():
    return 2.78


@njit
def partial_deriv_logistic():
    return 3.14


@njit
def partial_deriv_mom(
    j, partial_deriv_loss, state_solver, state_mom,
):
    state_solver.inner_products[0] = partial_deriv_loss()
    state_mom.block_means[0] = partial_deriv_loss()


@njit
def partial_deriv_erm(
    j, partial_deriv_loss, state_solver, state_erm,
):
    state_solver.inner_products[0] = partial_deriv_loss()
    # state_mom.block_means[0] = partial_deriv_loss()


spec_state_cgd = [
    ("inner_products", float64[::1]),
]


@jitclass(spec_state_cgd)
class StateCGD(object):
    def __init__(self, n_samples):
        self.inner_products = np.zeros((n_samples,), dtype=np.float64)


# mom_state = MOMState(3)
#
# print(mom_state)
#
# cgd_state = CGDState(10)
#
# print(cgd_state)
#

EstimatorMethods = namedtuple("EstimatorMethods", ["partial_deriv"])


@njit
def get_methods_mom():
    return EstimatorMethods(partial_deriv=partial_deriv_mom)


@njit
def get_methods_erm():
    return EstimatorMethods(partial_deriv=partial_deriv_erm)


# @njit
# def factory_mom_estimator(partial_deriv_loss, state_mom):
#     # mom_state = MOMState(n_blocks=n_blocks)
#     # block_means = state_mom.block_means
#     # inner_products = solver_state.inner_products
#
#     @njit
#     def partial_deriv(j, inner_products):
#         return partial_deriv_mom(j, inner_products, partial_deriv_loss, state_mom)
#
#     return Estimator(partial_deriv=partial_deriv)


# @njit
# def partial_deriv_mom(j, inner_products, partial_deriv_loss, estimator_state):
#     inner_products[0] = 3.14
#     block_means[0] = 2.78


# def solve(estimator):
#     partial_deriv = estimator.partial_deriv
#
#     @njit
#     def run():
#         partial_deriv(0)
#
#     run()


# @njit
# def partial_deriv_erm(j, cgd_state, strategy_state):
#     inner_products = cgd_state.inner_products
#     # block_means = strategy_state.block_means
#     inner_products[0] = 42

n_samples = 5
n_blocks = 2

# state_estimator = StateMOM(n_blocks=n_blocks)
state_estimator = StateERM()
state_solver = StateCGD(n_samples=n_samples)
estimator_methods = get_methods_erm()
# estimator_methods = get_methods_mom()


# estimator_methods

# estimator = factory_mom_estimator(partial_deriv_logistic, state_mom)

# estimator.partial_deriv(0, state_cgd.inner_products)


@njit
def solve(
    partial_deriv, state_solver, state_estimator,
):

    partial_deriv(0, partial_deriv_logistic, state_solver, state_estimator)


solve(estimator_methods.partial_deriv, state_solver, state_estimator)


print(state_solver.inner_products)

# print(state_estimator.block_means)

# j, partial_deriv_loss, state_solver, state_mom,
