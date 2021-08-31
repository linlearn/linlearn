# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from numba import njit, jit, vectorize, prange, uintp

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, nb_float, np_float


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


vectorize_kwargs = {
    "nopython": NOPYTHON,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}

# @njit
# def inner_prod(X, fit_intercept, i, w):
#     if fit_intercept:
#         return X[i].dot(w[1:]) + w[0]
#     else:
#         return X[i].dot(w)

# TODO: definir ici une strategy

# Strategy = namedtuple("Strategy", ["grad_coordinate", "n_samples_in_block"])

# TODO


@njit
def decision_function(X, fit_intercept, w, out):
    if fit_intercept:
        # TODO: use out= in dot and + z[0] at the same time with parallelize ?
        out[:] = X.dot(w[1:]) + w[0]
    else:
        out[:] = X.dot(w)
    return out


@njit
def decision_function_coef_intercept(X, fit_intercept, coef, intercept, out):
    if fit_intercept:
        # TODO: use out= in dot and + z[0] at the same time with parallelize ?
        # intercept is in a (1,) ndarray, following scikit-learn
        out[:] = X.dot(coef) + intercept[0]
    else:
        out[:] = X.dot(coef)
    return out


################################################################
# Generic estimator
################################################################


class Estimator(ABC):
    def __init__(self, X, y, loss, fit_intercept):
        self.X = X
        self.y = y
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_samples = y.shape[0]

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def partial_deriv_factory(self):
        pass


################################################################
# Empirical risk minimizer (ERM)
################################################################

StateERM = namedtuple("StateERM", [])


class ERM(object):
    def __init__(self, X, y, loss, fit_intercept):
        Estimator.__init__(self, X, y, loss, fit_intercept)

    def get_state(self):
        return StateERM()
        # return StateERM()

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_sum = 0.0
                if j == 0:
                    for i in range(n_samples):
                        deriv_sum += deriv_loss(y[i], inner_products[i])
                    return deriv_sum / n_samples
                else:
                    for i in range(n_samples):
                        deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                    return deriv_sum / n_samples

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_sum = 0.0
                for i in range(y.shape[0]):
                    deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j]
                return deriv_sum / n_samples

            return partial_deriv


################################################################
# Median of means estimator (MOM)
################################################################


StateMOM = namedtuple("StateMOM", ["block_means", "sample_indices"])


class MOM(Estimator):
    """MOM (Median-of-Means) estimator.

    """

    def __init__(self, X, y, loss, fit_intercept, n_samples_in_block):
        super().__init__(X, y, loss, fit_intercept)
        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block
        if self.last_block_size > 0:
            self.n_blocks += 1

    def get_state(self):
        return StateMOM(
            block_means=np.empty(self.n_blocks, dtype=np_float),
            sample_indices=np.empty(self.n_samples, dtype=np.uintp),
        )

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        deriv_loss = self.loss.deriv_factory()
        n_samples = self.n_samples
        n_samples_in_block = self.n_samples_in_block
        n_blocks = self.n_blocks
        last_block_size = self.last_block_size

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                for i in range(n_samples):
                    sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                if j == 0:
                    for i in sample_indices:
                        derivatives_sum_block += deriv_loss(y[i], inner_products[i])
                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                            block_means[n_block] = (
                                derivatives_sum_block / n_samples_in_block
                            )
                            n_block += 1
                            derivatives_sum_block = 0.0
                else:
                    for i in sample_indices:
                        derivatives_sum_block += (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )
                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                            block_means[n_block] = (
                                derivatives_sum_block / n_samples_in_block
                            )
                            n_block += 1
                            derivatives_sum_block = 0.0

                if last_block_size != 0:
                    block_means[n_blocks] = derivatives_sum_block / last_block_size

                return np.median(block_means)

            return partial_deriv

        else:
            # Same function without an intercept
            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                for i in range(n_samples):
                    sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                for i in sample_indices:
                    derivatives_sum_block += (
                        deriv_loss(y[i], inner_products[i]) * X[i, j]
                    )
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[n_block] = (
                            derivatives_sum_block / n_samples_in_block
                        )
                        n_block += 1
                        derivatives_sum_block = 0.0

                if last_block_size != 0:
                    block_means[n_blocks] = derivatives_sum_block / last_block_size
                return np.median(block_means)

            return partial_deriv


################################################################
# Trimmed means estimator (TMEAN)
################################################################


StateTMean = namedtuple("StateTMean", ["deriv_samples"])


class TMean(Estimator):
    """Trimmed-mean estimator"""

    def __init__(self, X, y, loss, fit_intercept, percentage):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.percentage = percentage
        # Number of samples excluded from both tails (left and right)
        self.n_excluded_tails = int(self.n_samples * percentage / 2)

    def get_state(self):
        return StateTMean(deriv_samples=np.empty(self.n_samples, dtype=np_float))

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                if j == 0:
                    for i in range(n_samples):
                        deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                else:
                    for i in range(n_samples):
                        deriv_samples[i] = (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )

                # TODO: Hand-made mean ?
                # TODO: Try out different sorting mechanisms, since at some point the
                #  sorting order won't change much...
                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i]) * X[i, j]

                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv


################################################################
# "Catoni-Holland" robust estimator
################################################################


@vectorize(**vectorize_kwargs)
def catoni(x):
    return np.sign(x) * np.log(1 + np.sign(x) * x + x * x / 2)
    # if x > 0 else -np.log(1 - x + x*x/2)


@vectorize(**vectorize_kwargs)
def khi(x):
    return 0.62 - 1 / (1 + x * x)  # np.log(0.5 + x*x)#


@vectorize(**vectorize_kwargs)
def gud(x):
    return 2 * np.arctan(np.exp(x)) - np.pi / 2 if x < 12 else np.pi / 2


@jit(**jit_kwargs)
def estimate_sigma(x, eps=0.001):
    sigma = 1.0
    x_mean = x.mean()
    delta = 1
    khi0 = khi(0.0)
    while delta > eps:
        tmp = sigma * np.sqrt(1 - (khi((x - x_mean) / sigma)).mean() / khi0)
        delta = np.abs(tmp - sigma)
        sigma = tmp
    return sigma


@jit(**jit_kwargs)
def holland_catoni_estimator(x, eps=0.001):
    # if the array is constant, do not try to estimate scale
    # the following condition is supposed to reproduce np.allclose() behavior
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]

    s = estimate_sigma(x) * np.sqrt(len(x) / np.log(1 / eps))
    m = 0.0
    diff = 1.0
    while diff > eps:
        tmp = m + s * gud((x - m) / s).mean()
        diff = np.abs(tmp - m)
        m = tmp
    return m


from scipy.optimize import brentq


def standard_catoni_estimator(x, eps=0.001):
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]
    s = estimate_sigma(x)
    res = brentq(lambda u: s * catoni((x - u) / s).mean(), np.min(x), np.max(x))
    return res


StateCatoni = namedtuple("StateCatoni", ["deriv_samples"])


class Catoni(Estimator):
    def __init__(self, X, y, loss, fit_intercept, eps=0.001):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.eps = eps

    def get_state(self):
        return StateCatoni(deriv_samples=np.empty(self.n_samples, dtype=np_float))

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        eps = self.eps

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                if j == 0:
                    for i in range(n_samples):
                        deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                else:
                    for i in range(n_samples):
                        deriv_samples[i] = (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )

                return holland_catoni_estimator(deriv_samples, eps)

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i]) * X[i, j]

                return holland_catoni_estimator(deriv_samples, eps)

            return partial_deriv


# C5 = 0.01
# C = lambda p: C5
# print(
#     "WARNING : importing implementation of outlier robust gradient by (Prasad et al.) with arbitrary constant C(p)=%.2f"
#     % C5
# )

#
#
# def SSI(samples, subset_cardinality):
#     """original name of this function is smallest_subset_interval"""
#     if subset_cardinality < 2:
#         raise ValueError("subset_cardinality must be at least 2")
#     sorted_array = np.sort(samples)
#     differences = (
#         sorted_array[subset_cardinality - 1 :] - sorted_array[: -subset_cardinality + 1]
#     )
#     argmin = np.argmin(differences)
#     return sorted_array[argmin : argmin + subset_cardinality]
#
#
# def alg2(X, eps, delta=0.001):
#     # from Prasad et al. 2018
#     X_tilde = alg4(X, eps, delta)
#
#     n, p = X_tilde.shape
#
#     if p == 1:
#         return np.mean(X_tilde)
#
#     S = np.cov(X.T)
#     _, V = np.linalg.eigh(S)
#     PW = V[:, : p // 2] @ V[:, : p // 2].T
#
#     est1 = np.mean(X_tilde @ PW, axis=0, keepdims=True)
#
#     QV = V[:, p // 2 :]
#     est2 = alg2(X_tilde @ QV, eps, delta)
#     est2 = QV.dot(est2.T)
#     est2 = est2.reshape((1, p))
#     est = est1 + est2
#
#     return est
#
#
# def alg4(X, eps, delta=0.001):
#     # from Prasad et al. 2018
#     n, p = X.shape
#     if p == 1:
#         X_tilde = SSI(
#             X.flatten(),
#             max(
#                 2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))
#             ),
#         )
#         return X_tilde[:, np.newaxis]
#
#     a = np.array([alg2(X[:, i : i + 1], eps, delta / p) for i in range(p)])
#     dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
#     asort = np.argsort(dists)
#     X_tilde = X[
#         asort[
#             : ceil(
#                 n
#                 * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n))
#                 * (1 - eps)
#             )
#         ],
#         :,
#     ]
#     return X_tilde
#
#
# def gmom(xs, tol=1e-7):
#     # from Vardi and Zhang 2000
#     y = np.average(xs, axis=0)
#     eps = 1e-10
#     delta = 1
#     niter = 0
#     while delta > tol:
#         xsy = xs - y
#         dists = np.linalg.norm(xsy, axis=1)
#         inv_dists = 1 / dists
#         mask = dists < eps
#         inv_dists[mask] = 0
#         nb_too_close = (mask).sum()
#         ry = np.linalg.norm(np.dot(inv_dists, xsy))
#         cst = nb_too_close / ry
#         y_new = (
#             max(0, 1 - cst) * np.average(xs, axis=0, weights=inv_dists)
#             + min(1, cst) * y
#         )
#         delta = np.linalg.norm(y - y_new)
#         y = y_new
#         niter += 1
#     # print(niter)
#     return y


@jit(**jit_kwargs, parallel=True)
def row_squared_norm_dense(model):
    n_samples, n_features = model.X.shape
    if model.fit_intercept:
        norms_squared = np.ones(n_samples, dtype=model.X.dtype)
    else:
        norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
    for i in prange(n_samples):
        for j in range(n_features):
            norms_squared[i] += model.X[i, j] * model.X[i, j]
    return norms_squared


def row_squared_norm(model):
    # TODO: for C and F order with aliasing
    return row_squared_norm_dense(model.no_python)


@jit(**jit_kwargs, parallel=True)
def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    if fit_intercept:
        norms_squared = np.zeros(n_features + 1, dtype=X.dtype)
        # First squared norm is n_samples
        norms_squared[0] = n_samples
        for j in prange(1, n_features + 1):
            for i in range(n_samples):
                norms_squared[j] += X[i, j - 1] ** 2
    else:
        norms_squared = np.zeros(n_features, dtype=X.dtype)
        for j in prange(n_features):
            for i in range(n_samples):
                norms_squared[j] += X[i, j] * X[i, j]
    return norms_squared


def col_squared_norm(model):
    # TODO: for C and F order with aliasing
    return col_squared_norm_dense(model.no_python)


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
