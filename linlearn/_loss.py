# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from math import exp, log
import numpy as np
from numba import jit, njit, vectorize, void, prange

from ._estimator import holland_catoni_estimator
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, nb_float, np_float


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}

# __losses = [
#     "hinge",
#     "smoothed hinge",
#     "logistic",
#     "quadratic hinge",
#     "modified huber",
# ]

#
# Generic functions
#


def decision_function_factory(X, fit_intercept):

    if fit_intercept:

        @jit(void(nb_float[::1], nb_float[::1]), **jit_kwargs)
        # @njit
        def decision_function(w, out):
            out[:] = X.dot(w[1:])
            out += w[0]

    else:

        @jit(void(nb_float[::1], nb_float[::1]), **jit_kwargs)
        # @njit
        def decision_function(w, out):
            out[:] = X.dot(w)

    return decision_function


# @njit(parallel=True)
# @njit
# def steps_coordinate_descent(lip_const, X, fit_intercept):
#     # def col_squared_norm_dense(X, fit_intercept):
#     n_samples, n_features = X.shape
#     # lip_const = loss_lip()
#     if fit_intercept:
#         steps = np.zeros(n_features + 1, dtype=X.dtype)
#         # First squared norm is n_samples
#         steps[0] = 1 / lip_const
#         for j in prange(1, n_features + 1):
#             col_j_squared_norm = 0.0
#             for i in range(n_samples):
#                 col_j_squared_norm += X[i, j - 1] ** 2
#             steps[j] = n_samples / (lip_const * col_j_squared_norm)
#     else:
#         steps = np.zeros(n_features, dtype=X.dtype)
#         for j in prange(n_features):
#             col_j_squared_norm = 0.0
#             for i in range(n_samples):
#                 col_j_squared_norm += X[i, j - 1] ** 2
#             steps[j] = n_samples / (lip_const * col_j_squared_norm)
#     # print(steps)
#     # steps /= n_samples
#     return steps

@njit
def median_of_means(x, block_size):
    n = x.shape[0]
    n_blocks = int(n // block_size)
    last_block_size = n % block_size
    if last_block_size == 0:
        block_means = np.empty(n_blocks, dtype=x.dtype)
    else:
        block_means = np.empty(n_blocks + 1, dtype=x.dtype)

    # TODO:instanciates in the closure
    # This shuffle or the indexes to get different blocks each time
    permuted_indices = np.random.permutation(n)
    sum_block = 0.0
    n_block = 0
    for i in range(n):
        idx = permuted_indices[i]
        # Update current sum in the block
        sum_block += x[idx]
        if (i != 0) and ((i + 1) % block_size == 0):
            # It's the end of the block, save its mean
            block_means[n_block] = sum_block / block_size
            n_block += 1
            sum_block = 0.0

    if last_block_size != 0:
        block_means[n_blocks] = sum_block / last_block_size

    mom = np.median(block_means)
    return mom#, blocks_means

def steps_factory(fit_intercept, estimator="mom", percentage=0.0, n_samples_in_block=0, eps=0.0):


    if estimator == "erm":
        if fit_intercept:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features + 1, dtype=X.dtype)
                # First squared norm is n_samples
                steps[0] = 1 / lip_const
                for j in prange(1, n_features + 1):
                    col_j_squared_norm = 0.0
                    for i in range(n_samples):
                        col_j_squared_norm += X[i, j - 1] ** 2
                    steps[j] = n_samples / (lip_const * col_j_squared_norm)
                return steps
            return steps_func
        else:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features, dtype=X.dtype)
                for j in prange(n_features):
                    col_j_squared_norm = 0.0
                    for i in range(n_samples):
                        col_j_squared_norm += X[i, j - 1] ** 2
                    steps[j] = n_samples / (lip_const * col_j_squared_norm)
                return steps
            return steps_func

    elif estimator == "mom" or estimator == "gmom" or estimator == "implicit":
        if n_samples_in_block ==0:
            raise ValueError("You should provide n_samples_in_block for mom/gmom estimator")
        if fit_intercept:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features + 1, dtype=X.dtype)
                # First squared norm is n_samples
                steps[0] = 1 / lip_const
                for j in prange(1, n_features + 1):
                    steps[j] = 1 / (max(median_of_means(X[:, j - 1] * X[:, j - 1], n_samples_in_block), 1e-8) * lip_const)
                return steps
            return steps_func
        else:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features, dtype=X.dtype)
                for j in prange(n_features):
                    steps[j] = 1 / (max(median_of_means(X[:, j] * X[:, j], n_samples_in_block), 1e-8) * lip_const)
                return steps
            return steps_func

    elif estimator == "holland_catoni":
        if eps == 0.0:
            raise ValueError("you should provide eps for catoni/holland estimator")
        if fit_intercept:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features + 1, dtype=X.dtype)
                squared_coordinates = np.zeros(n_samples, dtype=X.dtype)

                steps[0] = 1 / lip_const
                for j in range(n_features):
                    squared_coordinates[:] = X[:, j] * X[:, j]
                    steps[j+1] = 1 / (holland_catoni_estimator(squared_coordinates, eps) * lip_const)

                return steps
            return steps_func
        else:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                steps = np.zeros(n_features, dtype=X.dtype)
                squared_coordinates = np.zeros(n_samples, dtype=X.dtype)

                for j in range(n_features):
                    squared_coordinates[:] = X[:, j] * X[:, j]
                    steps[j] = 1 / (holland_catoni_estimator(squared_coordinates, eps) * lip_const)

                return steps
            return steps_func

    elif estimator == "tmean":
        if percentage == 0.0:
            raise ValueError("you should provide percentage for tmean estimator")
        if fit_intercept:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                n_excluded_tails = int(n_samples * percentage / 2)
                steps = np.zeros(n_features + 1, dtype=X.dtype)
                squared_coordinates = np.zeros(n_samples, dtype=X.dtype)

                steps[0] = 1 / lip_const
                for j in range(n_features):
                    squared_coordinates[:] = X[:, j] * X[:, j]
                    squared_coordinates.sort()

                    steps[j+1] = 1 / (np.mean(squared_coordinates[n_excluded_tails:-n_excluded_tails]) * lip_const)

                return steps
            return steps_func
        else:
            @jit(**jit_kwargs)
            def steps_func(lip_const, X):
                n_samples, n_features = X.shape
                n_excluded_tails = int(n_samples * percentage / 2)
                steps = np.zeros(n_features, dtype=X.dtype)
                squared_coordinates = np.zeros(n_samples, dtype=X.dtype)

                for j in range(n_features):
                    squared_coordinates[:] = X[:, j] * X[:, j]
                    squared_coordinates.sort()

                    steps[j] = 1 / (np.mean(squared_coordinates[n_excluded_tails:-n_excluded_tails]) * lip_const)

                return steps

            return steps_func

    else:
        raise ValueError("Unknown estimator")


@njit
def steps_coordinate_descent(lip_const, X, block_size, fit_intercept, estimator="mom"):
    # def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    if fit_intercept:
        steps = np.zeros(n_features + 1, dtype=X.dtype)
        # First squared norm is n_samples
        steps[0] = 1 / lip_const
        for j in prange(1, n_features + 1):
            steps[j] = 1 / (max(median_of_means(X[:, j - 1] * X[:, j - 1], block_size), 1e-8) * lip_const)
    else:
        steps = np.zeros(n_features, dtype=X.dtype)
        for j in prange(n_features):
            steps[j] = 1 / (max(median_of_means(X[:, j] * X[:, j], block_size), 1e-8) * lip_const)

    return steps


################################################################
# Abstract loss
################################################################


class Loss(ABC):
    @abstractmethod
    def value_factory(self):
        pass

    @abstractmethod
    def deriv_factory(self):
        pass

    def value_batch_factory(self):
        value = self.value_factory()

        @jit(**jit_kwargs)
        def value_batch(y, z):
            val = 0.0
            n_samples = y.shape[0]
            for i in range(n_samples):
                val += value(y[i], z[i])
            return val / n_samples

        return value_batch


# TODO: take the losses from https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/linear_model/_sgd_fast.pyx


################################################################
# Logistic regression loss
################################################################


@jit(**jit_kwargs)
def sigmoid(z):
    # TODO: faster sigmoid
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        exp_z = exp(z)
        return exp_z / (1 + exp_z)


@vectorize(fastmath=True)
def sigmoid(z):
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        exp_z = exp(z)
        return exp_z / (1 + exp_z)


# TODO: faster logistic


class Logistic(Loss):
    def __init__(self):
        self.lip = 0.25

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            agreement = y * z
            if agreement > 0:
                return log(1 + exp(-agreement))
            else:
                return -agreement + log(1 + exp(agreement))
            # if agreement > 18.0:
            #     return exp(-agreement)
            # elif agreement < -18.0:
            #     return -agreement
            # else:
            #     return log(1.0 + exp(-agreement))

        return value

    def deriv_factory(self):
        @jit(**jit_kwargs)
        def deriv(y, z):
            return -y * sigmoid(-y * z)
            # agreement = y * z
            # if agreement > 18.0:
            #     return exp(-agreement) * -y
            # elif agreement < -18.0:
            #     return -y
            # else:
            #     return -y / (exp(agreement) + 1.0)

        return deriv


################################################################
# Least-squares loss
################################################################


class LeastSquares(Loss):
    def __init__(self):
        self.lip = 1

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            return 0.5 * (y - z) * (y - z)

        return value

    def deriv_factory(self):
        @jit(**jit_kwargs)
        def deriv(y, z):
            return z - y

        return deriv


# @njit
# def hinge_value(x):
#     if x > 1.0:
#         return 1.0 - x
#     else:
#         return 0.0
#
#
# @njit
# def hinge_derivative(x):
#     if x > 1.0

# @njit
# def smoothed_hinge_value(x, smoothness=1.0):
#     y = x.copy()
#     idx = x >= 1
#     y[idx] = 0
#     if x <= 1 - smoothness:
#         return 1 - x - smoothness / 2
#     # y[idx] = 1 - x[idx] - smoothness / 2
#     elif (x >= 1 - smoothness) and (x < 1):
#         return
#     # idx = (x >= 1 - smoothness) & (x < 1)
#     y[idx] = (1 - y)[idx] ** 2 / (2 * smoothness)
#     return y


# @njit
# def quadratic_hinge_loss(x):
#     if x < 1:
#         return
#     y = (1 - x) ** 2 / 2
#     idx = x >= 1
#     y[idx] = 0
#     return y
#
#
# def modified_huber_loss(x):
#     y = np.zeros(x.shape)
#     idx = x <= -1
#     y[idx] = -4 * x[idx]
#     idx = (x > -1) & (x < 1)
#     y[idx] = (1 - x[idx]) ** 2
#     return y
#
