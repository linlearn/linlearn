import numpy as np
from numpy.random import shuffle
from numba import jit, uint32, uintp
from numba.experimental import jitclass

from ._loss import partial_deriv

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FLOAT, NP_FLOAT


#
# Empirical risk minimizer (ERM)
#


@jitclass([])
class StateERM(object):
    """Nothing happens here, but we need some empty shell to pass to the methods
    """

    def __init__(self):
        pass


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def partial_deriv_erm(loss_deriv, j, X, y, state_solver, state_erm):
    deriv_sum_block = 0.0
    fit_intercept = state_solver.fit_intercept
    inner_products = state_solver.inner_products
    n_samples = inner_products.shape[0]
    for i in range(n_samples):
        deriv_sum_block += partial_deriv(
            loss_deriv, j, y[i], inner_products[i], X[i], fit_intercept
        )
    return deriv_sum_block / n_samples


# @njit
# def grad_coordinate_erm(loss_derivative, j, X, y, inner_products, fit_intercept):
#     """Computation of the derivative of the loss with respect to a coordinate using the
#     empirical risk minimization (erm) stategy."""
#     grad = 0.0
#     # TODO: parallel ?
#     # TODO: sparse matrix ?
#     n_samples = inner_products.shape[0]
#     if fit_intercept:
#         if j == 0:
#             # In this case it's the derivative w.r.t the intercept
#             for i in range(n_samples):
#                 grad += loss_derivative(y[i], inner_products[i])
#         else:
#             for i in range(n_samples):
#                 grad += loss_derivative(y[i], inner_products[i]) * X[i, j - 1]
#     else:
#         # There is no intercept
#         for i in range(n_samples):
#             grad += loss_derivative(y[i], inner_products[i]) * X[i, j]
#     return grad / n_samples
#

#
# Median of means estimator (MOM)
#

spec_state_mom = [
    ("block_size", uintp),
    ("n_blocks", uintp),
    ("last_block_size", uintp),
    ("block_means", FLOAT[::1]),
    ("sample_indices", uintp[::1]),
    # ("pool", uint32[::1]),
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
def partial_deriv_mom(loss_derivative, j, X, y, state_solver, state_mom):
    """Computation of the derivative of the loss with respect to a coordinate using the
    median of means (mom) stategy."""
    # grad = 0.0
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
            loss_derivative, j, y[i], inner_products[i], X[i], fit_intercept
        )
        if (i != 0) and ((i + 1) % block_size == 0):
            block_means[n_block] = derivatives_sum_block / block_size
            n_block += 1
            derivatives_sum_block = 0.0

    if last_block_size != 0:
        block_means[n_blocks] = derivatives_sum_block / last_block_size

    return np.median(block_means)

    # if fit_intercept:
    #     if j == 0:
    #         # It the partial derivative w.r.t the intercept
    #         for i in sample_indices:
    #             # Update current sum in the block
    #             # print(sum_block, "+=", x[i])
    #             derivatives_sum_block += loss_derivative(y[i], inner_products[i])
    #             # sum_block += x[i]
    #             if (i != 0) and ((i + 1) % block_size == 0):
    #                 # End of the block => save its mean
    #                 # print("sum_block: ", sum_block)
    #                 block_means[n_block] = derivatives_sum_block / block_size
    #                 n_block += 1
    #                 derivatives_sum_block = 0.0
    #
    #         if last_block_size != 0:
    #             block_means[n_blocks] = derivatives_sum_block / last_block_size
    #
    #         return np.median(block_means)
    #     else:
    #         for idx in range(n_samples):
    #             i = idx_samples[idx]
    #             # Update current sum in the block
    #             # print(sum_block, "+=", x[i])
    #             derivatives_sum_block += (
    #                 loss_derivative(y[i], inner_products[i]) * X[i, j - 1]
    #             )
    #             # sum_block += x[i]
    #             if (i != 0) and ((i + 1) % n_samples_in_block == 0):
    #                 # It's the end of the block, we need to save its mean
    #                 # print("sum_block: ", sum_block)
    #                 grad_means_in_blocks[n_block] = (
    #                     derivatives_sum_block / n_samples_in_block
    #                 )
    #                 n_block += 1
    #                 derivatives_sum_block = 0.0
    #
    #         if last_block_size != 0:
    #             grad_means_in_blocks[n_blocks] = derivatives_sum_block / last_block_size
    #
    #         grad_mom = np.median(grad_means_in_blocks)
    #         return grad_mom
    # else:
    #     # There is no intercept
    #     for idx in range(n_samples):
    #         i = idx_samples[idx]
    #         # Update current sum in the block
    #         # print(sum_block, "+=", x[i])
    #         derivatives_sum_block += loss_derivative(y[i], inner_products[i]) * X[i, j]
    #         # sum_block += x[i]
    #         if (i != 0) and ((i + 1) % n_samples_in_block == 0):
    #             # It's the end of the block, we need to save its mean
    #             # print("sum_block: ", sum_block)
    #             grad_means_in_blocks[n_block] = (
    #                 derivatives_sum_block / n_samples_in_block
    #             )
    #             n_block += 1
    #             derivatives_sum_block = 0.0
    #
    #     if last_block_size != 0:
    #         grad_means_in_blocks[n_blocks] = derivatives_sum_block / last_block_size
    #
    #     grad_mom = np.median(grad_means_in_blocks)
    #     return grad_mom


# @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
# def median_of_means(x, block_means, block_size):
#     n = x.shape[0]
#     n_blocks = int(n // block_size)
#     last_block_size = n % block_size
#     if last_block_size == 0:
#         block_means = np.empty(n_blocks, dtype=x.dtype)
#     else:
#         block_means = np.empty(n_blocks + 1, dtype=x.dtype)
#
#     # TODO:instanciates in the closure
#     # This shuffle or the indexes to get different blocks each time
#     permuted_indices = permutation(n)
#     sum_block = 0.0
#     n_block = 0
#     for i in range(n):
#         idx = permuted_indices[i]
#         # Update current sum in the block
#         sum_block += x[idx]
#         if (i != 0) and ((i + 1) % block_size == 0):
#             # It's the end of the block, save its mean
#             block_means[n_block] = sum_block / block_size
#             n_block += 1
#             sum_block = 0.0
#
#     if last_block_size != 0:
#         block_means[n_blocks] = sum_block / last_block_size
#
#     mom = np.median(block_means)
#     return mom  # , blocks_means

#
# @njit
# def trimmed_mean(x, delta=0.01):
#     x.sort()
#     n_excluded = ceil(5 * np.log(8 / delta))
#     return np.mean(x[n_excluded:-n_excluded])
#
#
# @njit
# def grad_coordinate_per_sample(
#     loss_derivative, j, X, y, inner_products, fit_intercept,
# ):
#     n_samples = inner_products.shape[0]
#     # TODO: parallel ?
#     # TODO: sparse matrix ?
#
#     place_holder = np.empty(n_samples, dtype=X.dtype)
#     if fit_intercept:
#         if j[0] == 0:
#             # In this case it's the derivative w.r.t the intercept
#             for idx in range(n_samples):
#                 place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1])
#
#         else:
#             for idx in range(n_samples):
#                 place_holder[idx] = (
#                     loss_derivative(y[idx], inner_products[idx], j[1])
#                     * X[idx, j[0] - 1]
#                 )
#
#     else:
#         # There is no intercept
#         for idx in range(n_samples):
#             place_holder[idx] = (
#                 loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0]]
#             )
#     return place_holder
#
#
# @njit
# def grad_coordinate_erm(loss_derivative, j, X, y, inner_products, fit_intercept):
#     """Computation of the derivative of the loss with respect to a coordinate using the
#     empirical risk minimization (erm) stategy."""
#
#     return np.mean(
#         grad_coordinate_per_sample(
#             loss_derivative, j, X, y, inner_products, fit_intercept
#         )
#     )
#
#
# def erm_strategy_factory(loss, fit_intercept, **kwargs):
#     @njit
#     def grad_coordinate(X, y, j, inner_products):
#         return grad_coordinate_erm(
#             loss.derivative, j, X, y, inner_products, fit_intercept
#         )
#
#     return Strategy(
#         grad_coordinate=grad_coordinate, n_samples_in_block=None, name="erm"
#     )
#
#
# # TODO: overlapping blocks in MOM ???
#
#
# @njit
# def grad_coordinate_mom(
#     loss_derivative, j, X, y, inner_products, fit_intercept, n_samples_in_block,
# ):
#     """Computation of the derivative of the loss with respect to a coordinate using the
#     median of means (mom) stategy."""
#     # TODO: parallel ?
#     # TODO: sparse matrix ?
#     return median_of_means(
#         grad_coordinate_per_sample(
#             loss_derivative, j, X, y, inner_products, fit_intercept
#         ),
#         n_samples_in_block,
#     )

# n_samples = inner_products.shape[0]
# n_blocks = n_samples // n_samples_in_block
# last_block_size = n_samples % n_samples_in_block
#
# if n_samples % n_samples_in_block == 0:
#     grad_means_in_blocks = np.empty(n_blocks, dtype=X.dtype)
# else:
#     grad_means_in_blocks = np.empty(n_blocks + 1, dtype=X.dtype)
#
# # TODO:instanciates in the closure
# # This shuffle or the indexes to get different blocks each time
# idx_samples = permutation(n_samples)
# #permutation(idx_samples)
#
# # Cumulative sum in the block
# grad_block = 0.0
# # Block counter
# n_block = 0
#
# #print(j)
# if fit_intercept:
#     if j[0] == 0:
#         # In this case it's the derivative w.r.t the intercept
#         for idx in range(n_samples):
#             i = idx_samples[idx]
#             # Update current sum in the block
#             # print(sum_block, "+=", x[i])
#             grad_block += loss_derivative(y[i], inner_products[i], j[1])
#             # sum_block += x[i]
#             if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
#                 # It's the end of the block, we need to save its mean
#                 # print("sum_block: ", sum_block)
#                 grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
#                 n_block += 1
#                 grad_block = 0.0
#
#         if last_block_size != 0:
#             grad_means_in_blocks[n_blocks] = grad_block / last_block_size
#
#         grad_mom = np.median(grad_means_in_blocks)
#         return grad_mom
#     else:
#         for idx in range(n_samples):
#             i = idx_samples[idx]
#             # Update current sum in the block
#             # print(sum_block, "+=", x[i])
#             grad_block += loss_derivative(y[i], inner_products[i], j[1]) * X[i, j[0] - 1]
#             # sum_block += x[i]
#             if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
#                 # It's the end of the block, we need to save its mean
#                 # print("sum_block: ", sum_block)
#                 grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
#                 n_block += 1
#                 grad_block = 0.0
#
#         if last_block_size != 0:
#             grad_means_in_blocks[n_blocks] = grad_block / last_block_size
#
#         grad_mom = np.median(grad_means_in_blocks)
#         return grad_mom
# else:
#     # There is no intercept
#     for idx in range(n_samples):
#         i = idx_samples[idx]
#         # Update current sum in the block
#         # print(sum_block, "+=", x[i])
#         grad_block += loss_derivative(y[i], inner_products[i], j[1]) * X[i, j[0]]
#         # sum_block += x[i]
#         if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
#             # It's the end of the block, we need to save its mean
#             # print("sum_block: ", sum_block)
#             grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
#             n_block += 1
#             grad_block = 0.0
#
#     if last_block_size != 0:
#         grad_means_in_blocks[n_blocks] = grad_block / last_block_size
#
#     grad_mom = np.median(grad_means_in_blocks)
#     return grad_mom


# def erm_strategy_factory(loss, X, y, fit_intercept):
#     @njit
#     def grad_coordinate(j, inner_products):
#         return grad_coordinate_erm(
#             loss.derivative, j, X, y, inner_products, fit_intercept
#         )
#
#     return Strategy(grad_coordinate=grad_coordinate)


def mom_strategy_factory(loss, fit_intercept, n_samples_in_block, **kwargs):
    @njit
    def grad_coordinate(X, y, j, inner_products):
        return grad_coordinate_mom(
            loss.derivative, j, X, y, inner_products, fit_intercept, n_samples_in_block
        )

    return Strategy(
        grad_coordinate=grad_coordinate,
        n_samples_in_block=n_samples_in_block,
        name="mom",
    )


#
# @njit
# def grad_coordinate_catoni(
#     loss_derivative, j, X, y, inner_products, fit_intercept,
# ):
#     """Computation of the derivative of the loss with respect to a coordinate using the
#     catoni stategy."""
#     return Holland_catoni_estimator(
#         grad_coordinate_per_sample(
#             loss_derivative, j, X, y, inner_products, fit_intercept
#         )
#     )
#
#     # # grad = 0.0
#     # # TODO: parallel ?
#     # # TODO: sparse matrix ?
#     # n_samples = inner_products.shape[0]
#     #
#     # # TODO:instanciates in the closure
#     #
#     # place_holder = np.empty(n_samples, dtype=X.dtype)
#     # if fit_intercept:
#     #     if j[0] == 0:
#     #         # In this case it's the derivative w.r.t the intercept
#     #         for idx in range(n_samples):
#     #             place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1])
#     #
#     #     else:
#     #         for idx in range(n_samples):
#     #             place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0] - 1]
#     #
#     # else:
#     #     # There is no intercept
#     #     for idx in range(n_samples):
#     #         place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0]]
#
#
# def catoni_strategy_factory(loss, fit_intercept, **kwargs):
#     @njit
#     def grad_coordinate(X, y, j, inner_products):
#         return grad_coordinate_catoni(
#             loss.derivative, j, X, y, inner_products, fit_intercept
#         )
#
#     return Strategy(
#         grad_coordinate=grad_coordinate, n_samples_in_block=None, name="catoni"
#     )
#
#
# @njit
# def grad_coordinate_tmean(
#     loss_derivative, j, X, y, inner_products, fit_intercept,
# ):
#     return trimmed_mean(
#         grad_coordinate_per_sample(
#             loss_derivative, j, X, y, inner_products, fit_intercept
#         )
#     )
#
#
# def tmean_strategy_factory(loss, fit_intercept, **kwargs):
#     @njit
#     def grad_coordinate(X, y, j, inner_products):
#         return grad_coordinate_tmean(
#             loss.derivative, j, X, y, inner_products, fit_intercept
#         )
#
#     return Strategy(
#         grad_coordinate=grad_coordinate, n_samples_in_block=None, name="tmean"
#     )
#
#
# strategies_factory = {
#     "erm": erm_strategy_factory,
#     "mom": mom_strategy_factory,
#     "catoni": catoni_strategy_factory,
#     "tmean": tmean_strategy_factory,
# }
