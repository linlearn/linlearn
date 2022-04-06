# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from math import exp, log, fabs
import numpy as np
from numba import jit, njit, vectorize, void, prange

from .estimator.ch import holland_catoni_estimator
from .estimator.hg import alg2
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, nb_float, fast_median, fast_trimmed_mean, sum_sq, argmedian
from scipy.special import expit

# Options passed to the @jit decorator within this module
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

        @jit(**jit_kwargs)  # void(nb_float[::1], nb_float[::1]),
        def decision_function(w, out):
            out[:] = X.dot(w[1:])
            out += w[0]

    else:

        @jit(**jit_kwargs)  # void(nb_float[::1], nb_float[::1]),
        def decision_function(w, out):
            out[:] = X.dot(w)

    return decision_function

def batch_decision_function_factory(X, fit_intercept, n_classes, n_features):

    if fit_intercept:

        @jit(**jit_kwargs)  # void(nb_float[::1], nb_float[::1]),
        def decision_function(w, out, indices_batch):
            for ind in indices_batch:
                for k in range(n_classes):
                    out[ind, k] = w[0, k]
                    for j in range(n_features):
                        out[ind, k] += X[ind, j] * w[j+1, k]


    else:

        @jit(**jit_kwargs)  # void(nb_float[::1], nb_float[::1]),
        def decision_function(w, out, indices_batch):
            for ind in indices_batch:
                for k in range(n_classes):
                    out[ind, k] = 0.0
                    for j in range(n_features):
                        out[ind, k] += X[ind, j] * w[j, k]

    return decision_function


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

    # mom = np.median(block_means)
    return fast_median(block_means, len(block_means))#mom  # , blocks_means


#@jit(**jit_kwargs)
def compute_steps_cgd(
    X, estimator, fit_intercept, lip_const, percentage=0.0, n_samples_in_block=0, eps=0.0
):
    n_samples, n_features = X.shape
    int_fit_intercept = int(fit_intercept)
    steps = np.zeros(n_features + int_fit_intercept, dtype=X.dtype)
    if fit_intercept:
        steps[0] = 1 / lip_const
    if estimator == "erm":
        # First squared norm is n_samples
        sum_sq(X, 0, out=steps[int_fit_intercept:])
        # for j in prange(n_features):
        #     for i in range(n_samples):
        #         steps[j + int_fit_intercept] += X[i, j] * X[i, j]
        for j in prange(int_fit_intercept, n_features + int_fit_intercept):
            steps[j] = n_samples / (lip_const * max(steps[j], 1e-8))
        return steps

    elif estimator in ["mom", "gmom", "llm", "tmean", "ch"]:
        if n_samples_in_block == 0:
            raise ValueError(
                "You should provide n_samples_in_block for mom/gmom estimator"
            )
        for j in range(n_features):
            steps[j + int_fit_intercept] = 1 / (
                max(
                    median_of_means(
                        X[:, j] * X[:, j], n_samples_in_block
                    ),
                    1e-8,
                )
                * lip_const
            )
        return steps

    # elif estimator == "ch":
    #     if eps == 0.0:
    #         raise ValueError("you should provide eps for catoni/holland estimator")
    #     squared_coordinates = np.empty(n_samples, dtype=X.dtype)
    #
    #     for j in range(n_features):
    #         squared_coordinates[:] = X[:, j] * X[:, j]
    #         steps[j + int_fit_intercept] = 1 / (
    #             max(holland_catoni_estimator(squared_coordinates, eps), 1e-8) * lip_const
    #         )
    #
    #     return steps
    #
    # elif estimator == "tmean":
    #     # if percentage == 0.0:
    #     #     raise ValueError("you should provide percentage for tmean estimator")
    #
    #     # n_excluded_tails = int(n_samples * percentage / 2)
    #     squared_coordinates = np.empty(n_samples, dtype=X.dtype)
    #     for j in range(n_features):
    #         squared_coordinates[:] = X[:, j] * X[:, j]
    #         steps[j + int_fit_intercept] = 1 / (max(fast_trimmed_mean(squared_coordinates, n_samples, percentage), 1e-8) * lip_const)
    #
    #     return steps
    else:
        raise ValueError("Unknown estimator")

#@jit(**jit_kwargs)
def compute_steps(X, solver, estimator, fit_intercept, lip_const, percentage=0.0, n_blocks=0, eps=0.0):
    n_samples, n_features = X.shape
    int_fit_intercept = int(fit_intercept)

    if solver in ["sgd", "svrg", "saga"]:
        mean_sq_norms = np.mean(sum_sq(X, 1))
        # for i in range(n_samples):
        #     for j in range(n_features):
        #         sum_sq_norms += X[i, j] * X[i, j]
        step = 1 / (lip_const * max(int_fit_intercept, mean_sq_norms))
        return step
    elif solver in ["gd", "batch_gd", "md"]:
        if estimator == "erm":
            cov = X.T @ X
            step = n_samples / (lip_const * max(int_fit_intercept * n_samples, np.linalg.norm(cov, 2)))
            return step
        elif estimator == "mom":
            if n_blocks == 0:
                raise ValueError(
                    "You should provide n_blocks for mom/gmom/llm estimator"
                )

            square_coords = np.empty(n_samples, dtype=X.dtype)

            sum_norms = 0.0
            for j in range(n_features):
                # for i in range(n_samples):
                #     square_coords[i] = X[i, j] * X[i, j]
                square_coords[:] = X[:, j] * X[:, j]
                sum_norms += median_of_means(square_coords, int(n_samples/n_blocks))
            step = 1 / (lip_const * max(int_fit_intercept, sum_norms))
            return step

        elif estimator in ["llm", "gmom", "ch", "tmean", "hg"]:
            if n_blocks == 0:
                raise ValueError(
                    "You should provide n_blocks for mom/gmom/llm estimator"
                )
            # TODO : this is just an upper bound
            n_blocks += n_blocks % 2 + 1
            if n_blocks >= n_samples:
                n_blocks = n_samples - (n_samples % 2 + 1)
            n_samples_in_block = n_samples // n_blocks
            block_means = np.empty(n_blocks, dtype=X.dtype)
            # sum_sq = np.zeros(n_samples)
            # for i in range(n_samples):
            #     for j in range(n_features):
            #         sum_sq[i] += X[i, j] * X[i, j]
            square_norms = sum_sq(X, 1)
            sample_indices = np.arange(n_samples)#np.arange(n_blocks * n_samples_in_block)#
            np.random.shuffle(sample_indices)
            # Cumulative sum in the block
            sum_block = 0.0
            # Block counter
            counter = 0
            for i, idx in enumerate(sample_indices[:n_blocks*n_samples_in_block]):
                sum_block += square_norms[idx]
                if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                    block_means[counter] = sum_block / n_samples_in_block
                    counter += 1
                    sum_block = 0.0
            argmed = argmedian(block_means)

            cov = np.zeros((n_features, n_features))
            for i in sample_indices[
                     argmed * n_samples_in_block: (argmed + 1) * n_samples_in_block
                     ]:
                cov += np.outer(X[i], X[i])

            step = n_samples_in_block / (lip_const * max(int_fit_intercept * n_samples_in_block, np.linalg.norm(cov, 2)))
            return step

        # elif estimator == "ch":
        #     if eps == 0.0:
        #         raise ValueError("you should provide eps for ch estimator")
        #     # square_coords = np.empty(n_samples, dtype=X.dtype)
        #     # sum_norms = 0.0
        #     # for j in range(n_features):
        #     #     square_coords[:] = X[:, j] * X[:, j]
        #     #     sum_norms += holland_catoni_estimator(square_coords, eps)
        #     square_norms = sum_sq(X, 1)
        #     square_norm_average_estimate = holland_catoni_estimator(square_norms, eps)
        #     step = 1 / (lip_const * max(int_fit_intercept, square_norm_average_estimate))
        #     return step

        # elif estimator == "tmean":
        #     if percentage == 0.0:
        #         raise ValueError("you should provide percentage for tmean estimator")
        #     # square_coords = np.empty(n_samples, dtype=X.dtype)
        #     # sum_norms = 0.0
        #     # for j in range(n_features):
        #     #     square_coords[:] = X[:, j] * X[:, j]
        #     #     sum_norms += fast_trimmed_mean(square_coords, n_samples, percentage)
        #     square_norms = sum_sq(X, 1)
        #     square_norm_average_estimate = fast_trimmed_mean(square_norms, n_samples, percentage)
        #     step = 1 / (lip_const * max(int_fit_intercept, square_norm_average_estimate))
        #     return step
        # elif estimator == "hg":
        #     if percentage == 0.0:
        #         raise ValueError("you should provide percentage for hg estimator")
        #     # square_coords = np.empty(n_samples, dtype=X.dtype)
        #     # sum_norms = 0.0
        #     # for j in range(n_features):
        #     #     square_coords[:] = X[:, j] * X[:, j]
        #     #     sum_norms += fast_trimmed_mean(square_coords, n_samples, percentage)
        #     square_norms = np.expand_dims(sum_sq(X, 1), axis=1)
        #     square_norm_average_estimate = alg2(square_norms, 2*percentage)[0, 0]
        #     step = 1 / (lip_const * max(int_fit_intercept, square_norm_average_estimate))
        #     return step
        else:
            raise ValueError("Unknown estimator")

    else:
        raise ValueError("Unknown solver")



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


@jit(inline="always", **jit_kwargs)
def sigmoid(z):
    # TODO: faster sigmoid
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        exp_z = exp(z)
        return exp_z / (1 + exp_z)


# @vectorize(fastmath=True)
# def sigmoid(z):
#     if z > 0:
#         return 1 / (1 + exp(-z))
#     else:
#         exp_z = exp(z)
#         return exp_z / (1 + exp_z)


# TODO: faster logistic


class Logistic(Loss):
    def __init__(self):
        self.lip = 0.25

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            agreement = y * z[0]
            if agreement > 0:
                return log(1 + exp(-agreement))
            else:
                return -agreement + log(1 + exp(agreement))

        return value

    def deriv_factory(self):
        @jit(inline="always", **jit_kwargs)
        def deriv(y, z, out):
            out[0] = -y * expit(-y * z[0])#sigmoid(-y * z[0])#

        return deriv


################################################################
# Multiclass Logistic regression loss
################################################################


class MultiLogistic(Loss):
    def __init__(self, n_classes):
        self.lip = 0.25
        self.n_classes = n_classes

    def value_factory(self):
        n_classes = self.n_classes
        @jit(**jit_kwargs)
        def value(y, z):
            max_z = z[0]
            for k in range(1, n_classes):
                if z[k] > max_z:
                    max_z = z[k]
            exponentiated = np.empty_like(z)
            norm = 0.0
            for k in range(n_classes):
                exponentiated[k] = exp(z[k] - max_z)
                norm += exponentiated[k]
            return -log(exponentiated[y]/norm)

        return value

    def deriv_factory(self):
        n_classes = self.n_classes
        @jit(**jit_kwargs)
        def deriv(y, z, out):
            # TODO : unrolling loops makes the code much faster but less precise ...
            max_z = z[0]
            for k in range(1, n_classes):
                if z[k] > max_z:
                    max_z = z[k]

            # np.exp(z - np.max(z), out)
            norm = 0.0
            for k in range(n_classes):
                out[k] = np.exp(z[k] - max_z) # not much speed difference between exp and np.exp
                norm += out[k]
            for k in range(n_classes):
                out[k] /= norm
            # out /= out.sum()

            out[y] -= 1

        return deriv


################################################################
# Least-squares loss
################################################################


class LeastSquares(Loss):
    def __init__(self):
        self.lip = 1.0

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            return 0.5 * (y - z[0]) ** 2

        return value

    def deriv_factory(self):
        @jit(**jit_kwargs)
        def deriv(y, z, out):
            out[0] = z[0] - y

        return deriv


################################################################
# Huber loss
################################################################


class Huber(Loss):
    def __init__(self, delta=1.35):
        self.lip = 1.0
        self.delta = delta

    def value_factory(self):
        delta = self.delta
        @jit(**jit_kwargs)
        def value(y, z):
            diff = fabs(y - z[0])
            if diff < delta:
                return 0.5 * (diff) ** 2
            else:
                return delta * (diff - 0.5 * delta)

        return value

    def deriv_factory(self):
        delta = self.delta
        @jit(inline="always", **jit_kwargs)
        def deriv(y, z, out):

            diff = z[0] - y
            if diff < -delta:
                out[0] = -delta
            elif diff < delta:
                out[0] = z[0] - y
            else:
                out[0] = delta

        return deriv


################################################################
# Modified Huber loss
################################################################


class ModifiedHuber(Loss):
    def __init__(self):
        self.lip = 2.0

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            agreement = y * z[0]
            if agreement > -1:
                return max(0.0, 1 - agreement) ** 2
            else:
                return -4 * agreement

        return value

    def deriv_factory(self):
        @jit(inline="always", **jit_kwargs)
        def deriv(y, z, out):
            agreement = y * z[0]
            if agreement < -1:
                out[0] = -4 * y
            if agreement < 1:
                out[0] = -2 * y * (1 - agreement)
            else:
                out[0] = 0

        return deriv


################################################################
# Multiclass Modified Huber loss
################################################################


class MultiModifiedHuber(Loss):
    def __init__(self, n_classes):
        self.lip = 2.0
        self.n_classes = n_classes

    def value_factory(self):
        n_classes = self.n_classes
        @jit(**jit_kwargs)
        def value(y, z):
            val = 0.0
            for i in range(n_classes):
                if i == y:
                    agreement = z[i]
                else:
                    agreement = -z[i]
                if agreement > -1:
                    val += max(0.0, 1 - agreement) ** 2
                else:
                    val -= 4 * agreement
            return val

        return value

    def deriv_factory(self):
        n_classes = self.n_classes
        @jit(inline="always", **jit_kwargs)
        def deriv(y, z, out):
            for i in range(n_classes):
                if i == y:
                    agreement = z[i]
                    sign = +1
                else:
                    agreement = -z[i]
                    sign = -1

                if agreement < -1:
                    out[i] = -4 * sign
                if agreement < 1:
                    out[i] = -2 * sign * (1 - agreement)
                else:
                    out[i] = 0

        return deriv


################################################################
# Multiclass Squared-Hinge loss
################################################################


class MultiSquaredHinge(Loss):
    def __init__(self, n_classes):
        self.lip = 1
        self.n_classes = n_classes

    def value_factory(self):
        n_classes = self.n_classes
        @jit(**jit_kwargs)
        def value(y, z):
            val = 0.0
            for i in range(n_classes):
                if z[i] > -1:
                    val += 0.5 * (1 + z[i]) ** 2
            if z[y] > 1:
                val -= 0.5 * (1 + z[y]) ** 2
            else:
                val -= 2 * z[y]
            return val

        return value

    def deriv_factory(self):
        n_classes = self.n_classes
        @jit(**jit_kwargs)
        def deriv(y, z, out):
            for i in range(n_classes):
                if z[i] < 1:
                    out[i] = 0.0
                else:
                    out[i] = z[i] - 1
            if z[y] < 1:
                out[y] = z[y] - 1
            else:
                out[y] = 0.0

        return deriv


################################################################
# Squared-Hinge loss
################################################################


class SquaredHinge(Loss):
    def __init__(self):
        self.lip = 1

    def value_factory(self):
        @jit(**jit_kwargs)
        def value(y, z):
            agreement = y * z[0]
            if agreement > 1:
                return 0.0
            else:
                return 0.5 * (1 - agreement) ** 2

        return value

    def deriv_factory(self):
        @jit(inline="always", **jit_kwargs)
        def deriv(y, z, out):
            agreement = y * z[0]
            if agreement > 1:
                out[0] = 0.0
            else:
                out[0] = -y * (1 - agreement)

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
