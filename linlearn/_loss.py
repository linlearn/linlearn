# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from math import exp, log
import numpy as np
from numba import jit, njit, vectorize, void, prange

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
@njit
def steps_coordinate_descent(lip_const, X, fit_intercept):
    # def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    # lip_const = loss_lip()
    if fit_intercept:
        steps = np.zeros(n_features + 1, dtype=X.dtype)
        # First squared norm is n_samples
        steps[0] = 1 / lip_const
        for j in prange(1, n_features + 1):
            col_j_squared_norm = 0.0
            for i in range(n_samples):
                col_j_squared_norm += X[i, j - 1] ** 2
            steps[j] = n_samples / (lip_const * col_j_squared_norm)
    else:
        steps = np.zeros(n_features, dtype=X.dtype)
        for j in prange(n_features):
            col_j_squared_norm = 0.0
            for i in range(n_samples):
                col_j_squared_norm += X[i, j - 1] ** 2
            steps[j] = n_samples / (lip_const * col_j_squared_norm)
    # print(steps)
    # steps /= n_samples
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
