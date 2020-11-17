from math import exp, log
from numba import njit, jitclass, vectorize, prange
from numba.types import int64, float64, boolean

from collections import namedtuple

import numpy as np

Loss = namedtuple("Loss", ["value_single", "value_batch", "derivative", "lip"])


#
# Generic functions
#
@njit
def loss_value_batch(loss_value_single, y, z):
    val = 0.0
    n_samples = y.shape[0]
    for i in range(n_samples):
        val += loss_value_single(y[i], z[i])

    return val / n_samples


# @njit(parallel=True)
@njit
def steps_coordinate_descent(loss_lip, X, fit_intercept):
    # def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    lip_const = loss_lip()
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


# TODO: take the losses from https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/linear_model/_sgd_fast.pyx

#
# Functions for the logistic loss
#


# cdef class Log(Classification):
#     """Logistic regression loss for binary classification with y in {-1, 1}"""
#
#     cdef double loss(self, double p, double y) nogil:
#         cdef double z = p * y
#         # approximately equal and saves the computation of the log
#         if z > 18:
#             return exp(-z)
#         if z < -18:
#             return -z
#         return log(1.0 + exp(-z))
#
#     cdef double _dloss(self, double p, double y) nogil:
#         cdef double z = p * y
#         # approximately equal and saves the computation of the log
#         if z > 18.0:
#             return exp(-z) * -y
#         if z < -18.0:
#             return -y
#         return -y / (exp(z) + 1.0)
#
#     def __reduce__(self):
#         return Log, ()


@njit(fastmath=True)
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


@njit(fastmath=True, inline="always")
def logistic_value_single(y, z):
    s = y * z
    if s > 0:
        return log(1 + exp(-s))
    else:
        return -s + log(1 + exp(s))


@njit
def logistic_value_batch(y, z):
    return loss_value_batch(logistic_value_single, y, z)


@njit
def logistic_derivative(y, z):
    return -y * sigmoid(-y * z)


@njit
def logistic_lip():
    return 0.25


def logisic_factory():
    return Loss(
        value_single=logistic_value_single,
        value_batch=logistic_value_batch,
        derivative=logistic_derivative,
        lip=logistic_lip,
    )


losses_factory = {"logistic": logisic_factory}

# y = np.random.randn(100)
# z = np.random.randn(100)
#
# print(logistic_value_batch(y, z))
# print(logistic_derivative(y[0], z[0]))

# @njit

# def loss_batch(model, w):
#     out = 0.0
#     # TODO: allocate this in fit
#     n_samples = model.y.shape[0]
#     Xw = np.empty(n_samples)
#     # TODO: inner_prods or for loop ? No need for Xw
#     Xw = inner_prods(model.X, model.fit_intercept, w, Xw)
#     if model.sample_weight.size == 0:
#         for i in range(n_samples):
#             out += model.loss(model.y[i], Xw[i]) / n_samples
#     else:
#         for i in range(n_samples):
#             out += model.sample_weight[i] * model.loss(model.y[i], Xw[i]) / n_samples
#     return out


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
#
# losses = [
#     (hinge_loss, "Hinge"),
#     (smoothed_hinge_loss, "Smoothed hinge"),
#     (logistic_loss, "Logistic"),
#     (quadratic_hinge_loss, "Quadratic hinge"),
#     (modified_huber_loss, "Modified Huber"),
# ]


# def loss_batch(self, w):
#     return loss_batch(self, w)
#
#
# def grad_batch(self, w, out):
#     grad_batch(self, w, out)
#
#
# def grad_sample_coef(self, i, w):
#     return grad_sample_coef(self, i, w)


# This is the old Model class
# class Model(object):
#     def __init__(self, no_python_class, fit_intercept=True):
#         self.no_python = no_python_class(fit_intercept)
#         self.fit_intercept = fit_intercept
#         self._lips = None
#         self._lip_mean = None
#         self._lip_max = None
#
#     def set(self, X, y, sample_weight=None):
#         estimator = self.__class__.__name__
#
#         X, y = check_X_y(
#             X,
#             y,
#             accept_sparse=False,
#             accept_large_sparse=True,
#             dtype=["float64"],
#             order="C",
#             copy=False,
#             force_all_finite=True,
#             ensure_2d=True,
#             allow_nd=False,
#             multi_output=False,
#             ensure_min_samples=1,
#             ensure_min_features=1,
#             y_numeric=True,
#             estimator=estimator,
#         )
#
#         # For now, we must ensure that dtype of labels if float64
#         if y.dtype != "float64":
#             y = y.astype(np.float64)
#
#         if sample_weight is None:
#             # Use an empty np.array if no sample_weight is used
#             sample_weight = np.empty(0, dtype=np.float64)
#         else:
#             sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
#         self.no_python.set(X, y, sample_weight)
#
#         # lipschitz constants must be recomputed with a new call to set
#         self._lips = None
#         self._lip_mean = None
#         self._lip_max = None
#         return self
#
#     def loss(self, w):
#         return self.no_python.loss_batch(w)
#
#     @property
#     def is_set(self):
#         return self.no_python.is_set
#
#     @property
#     def fit_intercept(self):
#         return self.no_python.fit_intercept
#
#     @fit_intercept.setter
#     def fit_intercept(self, val):
#         if type(val) is bool:
#             self.no_python.fit_intercept = val
#         else:
#             raise ValueError("'fit_intercept' must be of boolean type")
#
#     @property
#     def n_samples(self):
#         if self.is_set:
#             return self.no_python.n_samples
#         else:
#             return None
#
#     @property
#     def n_features(self):
#         if self.is_set:
#             return self.no_python.n_features
#         else:
#             return None
#
#     @property
#     def X(self):
#         if self.is_set:
#             return self.no_python.X
#         else:
#             return None
#
#     @property
#     def y(self):
#         if self.is_set:
#             return self.no_python.y
#         else:
#             return None
#
#     @property
#     def sample_weight(self):
#         if self.is_set:
#             if self.no_python.sample_weight.size == 0:
#                 return None
#             else:
#                 return self.no_python.sample_weight
#         else:
#             return None
#
#     def __repr__(self):
#         fit_intercept = self.fit_intercept
#         n_samples = self.n_samples
#         n_features = self.n_features
#         r = self.__class__.__name__
#         r += "(fit_intercept={fit_intercept}".format(fit_intercept=fit_intercept)
#         r += ", n_samples={n_samples}".format(n_samples=n_samples)
#         r += ", n_features={n_features})".format(n_features=n_features)
#         return r
#
#     @property
#     def lips(self):
#         if self.is_set:
#             if self._lips is None:
#                 self._compute_lips()
#             return self._lips
#         else:
#             raise RuntimeError("You must use 'set' before using 'lips'")
#
#     @property
#     def lip_max(self):
#         if self.is_set:
#             if self._lip_max is None:
#                 self._lip_max = self.lips.max()
#             return self._lip_max
#         else:
#             raise RuntimeError("You must use 'set' before using 'lip_max'")
#
#     @property
#     def lip_mean(self):
#         if self.is_set:
#             if self._lip_mean is None:
#                 self._lip_mean = self.lips.mean()
#             return self._lip_mean
#         else:
#             raise RuntimeError("You must use 'set' before using 'lip_mean'")
#
#     def _compute_lips(self):
#         raise NotImplementedError()


# And some utils

# @njit
# def inner_prod(X, fit_intercept, i, w):
#     if fit_intercept:
#         return X[i].dot(w[1:]) + w[0]
#     else:
#         return X[i].dot(w)
#
#
# @njit
# def inner_prods(X, fit_intercept, w, out):
#     if fit_intercept:
#         # TODO: use out= in dot and + z[0] at the same time with parallelize ?
#         out[:] = X.dot(w[1:]) + w[0]
#     else:
#         out[:] = X.dot(w)
#     return out
#
#
# @njit
# def loss_sample(model, i, w):
#     z = inner_prod(model.X, model.fit_intercept, i, w)
#     if model.sample_weight.size == 0:
#         return model.value(model.y[i], z)
#     else:
#         return model.sample_weight[i] * model.value(model.y[i], z)
#
#
# @njit
# def loss_batch(model, w):
#     out = 0.0
#     # TODO: allocate this in fit
#     n_samples = model.y.shape[0]
#     Xw = np.empty(n_samples)
#     # TODO: inner_prods or for loop ? No need for Xw
#     Xw = inner_prods(model.X, model.fit_intercept, w, Xw)
#     if model.sample_weight.size == 0:
#         for i in range(n_samples):
#             out += model.loss(model.y[i], Xw[i]) / n_samples
#     else:
#         for i in range(n_samples):
#             out += model.sample_weight[i] * model.loss(model.y[i], Xw[i]) / n_samples
#     return out
#
#
# @njit
# def grad_sample_coef(model, i, w):
#     z = inner_prod(model.X, model.fit_intercept, i, w)
#     if model.sample_weight.size == 0:
#         return model.derivative(model.y[i], z)
#     else:
#         return model.sample_weight[i] * model.derivative(model.y[i], z)
#
#
# @njit
# def grad_sample(model, i, w, out):
#     c = grad_sample_coef(model, i, w)
#     if model.fit_intercept:
#         out[0] = c
#         out[1:] = c * model.X[i]
#     else:
#         out[:] = c * model.X[i]
#     return out
#
#
# @njit
# def grad_coordinate(model, j, inner_products):
#     grad = 0.0
#     # TODO: parallel ?
#     if model.fit_intercept:
#         if j == 0:
#             # In this case it's the derivative w.r.t the intercept
#             for i in range(model.n_samples):
#                 grad += model.derivative(model.y[i], inner_products[i])
#         else:
#             for i in range(model.n_samples):
#                 grad += model.X[i, j - 1] * model.derivative(
#                     model.y[i], inner_products[i]
#                 )
#     else:
#         # There is no intercept
#         for i in range(model.n_samples):
#             grad += model.X[i, j] * model.derivative(model.y[i], inner_products[i])
#     return grad / model.n_samples
#
#
# @njit
# def grad_batch(model, w, out):
#     out.fill(0)
#     if model.fit_intercept:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[1:] += c * model.X[i]
#             out[0] += c
#     else:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[:] += c * model.X[i]
#     return out
#
#
# @njit(parallel=True)
# def row_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.ones(n_samples, dtype=model.X.dtype)
#     else:
#         norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
#     for i in prange(n_samples):
#         for j in range(n_features):
#             norms_squared[i] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def row_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return row_squared_norm_dense(model.no_python)
#
#
# @njit(parallel=True)
# def col_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.zeros(n_features + 1, dtype=model.X.dtype)
#         # First squared norm is n_samples
#         norms_squared[0] = n_samples
#         for j in prange(1, n_features + 1):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j - 1] * model.X[i, j - 1]
#     else:
#         norms_squared = np.zeros(n_features, dtype=model.X.dtype)
#         for j in prange(n_features):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def col_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return col_squared_norm_dense(model.no_python)
