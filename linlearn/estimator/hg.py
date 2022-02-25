# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause


from collections import namedtuple
import numpy as np
from numba import jit, prange
from ._base import Estimator, jit_kwargs
from .._utils import np_float
from math import ceil


StateHG = namedtuple(
    "StateHG",
    [
        "sample_gradients",
        "gradient",
        "loss_derivative",
        "partial_derivative",
    ],
)

C5 = 0.01
@jit(**jit_kwargs)
def C(p):
    return C5
print("WARNING : importing implementation of outlier robust gradient by (Prasad et al.) with arbitrary constant C(p)=%.2f"%C5)

@jit(**jit_kwargs)
def SSI(samples, subset_cardinality):
    """original name of this function is smallest_subset_interval"""
    if subset_cardinality < 2:
        raise ValueError("subset_cardinality must be at least 2")
    elif subset_cardinality >= len(samples):
        return samples
    samples.sort()
    differences = samples[subset_cardinality - 1:] - samples[:-subset_cardinality + 1]
    argmin = np.argmin(differences)
    return samples[argmin:argmin + subset_cardinality]


# @jit("float64[:,:](float64[:,:], float64, float64)", **jit_kwargs)
# def alg4(X, eps, delta=0.01):
#     # from Prasad et al. 2018
#     n, p = X.shape
#     if p == 1:
#         X_tilde = SSI(X.flatten(), max(2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))))
#         return X_tilde[:, np.newaxis]
#
#     a = np.array([alg2(X[:, i:i + 1], eps, delta / p) for i in range(p)])
#     dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
#     asort = np.argsort(dists)
#     X_tilde = X[asort[:ceil(n * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n)) * (1 - eps))], :]
#     return X_tilde


#@jit("float64[:,:](float64[:,:], float64, float64)", **jit_kwargs)
@jit(**jit_kwargs)
def alg2(X, eps, delta=0.01):
    # from Prasad et al. 2018
    sc_prods = 0

    # X_tilde = alg4(X, eps, delta)
    n, p = X.shape
    if p == 1:
        X_tilde = SSI(X.flatten(), max(2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))))
        X_tilde = np.expand_dims(X_tilde, 1)# X_tilde[:, np.newaxis]
    else:

        a = np.array([alg2(X[:, i:i + 1], eps, delta / p).sum() for i in range(p)])
        dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
        asort = np.argsort(dists)
        X_tilde = X[asort[:ceil(n * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n)) * (1 - eps))], :]


    n, p = X_tilde.shape

    if p == 1:
        return np.array([[np.mean(X_tilde)]])

    S = np.cov(X.T)

    _, V = np.linalg.eigh(S)
    PV = V[:, :p // 2]
    PW = PV @ PV.T

    est1 = np.expand_dims((X_tilde @ PW).sum(axis=0)/X_tilde.shape[0], 0)
    #est1 = np.mean(X_tilde @ PW, axis=0, keepdims=True)

    QV = V[:, p // 2:]
    est2 = alg2(X_tilde @ QV, eps, delta)
    est2 = QV.dot(est2.T)
    est2 = est2.reshape((1, p))
    est = est1 + est2

    return est


class HG(Estimator):
    def __init__(self, X, y, loss, n_classes, fit_intercept, delta=0.01, eps=0.01):
        super().__init__(X, y, loss, n_classes, fit_intercept)
        self.delta = delta
        self.eps = eps


    def get_state(self):
        return StateHG(
            sample_gradients=np.empty(
                (self.n_samples, self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
            partial_derivative=np.empty(self.n_classes, dtype=np_float),
        )

    def partial_deriv_factory(self):
        raise ValueError(
            "Hubergrad estimator does not support CGD, use mom estimator instead"
        )

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_classes = self.n_classes
        n_features = self.n_features
        eps = self.eps
        delta = self.delta

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                gradient = state.gradient
                sample_gradients = state.sample_gradients
                deriv = state.loss_derivative
                for i in prange(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv)
                    for k in range(n_classes):
                        sample_gradients[i, 0, k] = deriv[k]
                    for j in range(n_features):
                        for k in range(n_classes):
                            sample_gradients[i, j + 1, k] = X[i, j] * deriv[k]

                gradient[:] = alg2(sample_gradients.reshape((n_samples, -1)), 2*eps, delta).reshape((gradient.shape))
                return 0

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):

                gradient = state.gradient
                sample_gradients = state.sample_gradients
                deriv = state.loss_derivative
                for i in prange(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv)
                    for j in range(n_features):
                        for k in range(n_classes):
                            sample_gradients[i, j, k] = X[i, j] * deriv[k]

                gradient[:] = alg2(sample_gradients.reshape((n_samples, -1)), 2*eps, delta).reshape((gradient.shape))
                return 0

            return grad
