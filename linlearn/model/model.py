import numpy as np
from numba import njit


@njit
def inner_prod(X, fit_intercept, i, w):
    if fit_intercept:
        return X[i].dot(w[1:]) + w[0]
    else:
        return X[i].dot(w)


@njit
def inner_prods(X, fit_intercept, w, out):
    if fit_intercept:
        # TODO: use out= in dot and + z[0] at the same time with parallelize ?
        out[:] = X.dot(w[1:]) + w[0]
    else:
        out[:] = X.dot(w)
    return out


@njit
def loss_sample(model, i, w):
    z = inner_prod(model.X, model.fit_intercept, i, w)
    return model.value(model.y[i], z)


@njit
def loss_batch(model, w):
    out = 0.
    # TODO: allocate this in fit
    n_samples = model.y.shape[0]
    Xw = np.empty(n_samples)
    # TODO: inner_prods or for loop ? No need for Xw
    Xw = inner_prods(model.X, model.fit_intercept, w, Xw)
    for i in range(n_samples):
        out += model.loss(model.y[i], Xw[i]) / n_samples
    return out


@njit
def grad_sample_coef(model, i, w):
    z = inner_prod(model.X, model.fit_intercept, i, w)
    return model.derivative(model.y[i], z)


@njit
def grad_sample(model, i, w, out):
    c = grad_sample_coef(model, i, w)
    if model.fit_intercept:
        out[0] = c
        out[1:] = c * model.X[i]
    else:
        out[:] = c * model.X[i]
    return out


@njit
def grad_batch(model, w, out):
    out.fill(0)
    if model.fit_intercept:
        for i in range(model.n_samples):
            c = grad_sample_coef(model, i, w) / model.n_samples
            out[1:] += c * model.X[i]
            out[0] += c
    else:
        for i in range(model.n_samples):
            c = grad_sample_coef(model, i, w) / model.n_samples
            out[:] += c * model.X[i]
    return out


class Model(object):

    def __init__(self, no_python_class, fit_intercept=True):
        self.no_python = no_python_class(fit_intercept)
        self.fit_intercept = fit_intercept

    def set(self, X, y):
        # TODO: here all the checks about X and y : C-order and contiguous, etc.
        self.no_python.set(X, y)
        return self

    @property
    def is_set(self):
        return self.no_python.is_set

    @property
    def fit_intercept(self):
        return self.no_python.fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, val):
        if type(val) is bool:
            self.no_python.fit_intercept = val
        else:
            raise ValueError("'fit_intercept' must be of boolean type")

    @property
    def n_samples(self):
        if self.is_set:
            return self.no_python.n_samples
        else:
            return None

    @property
    def n_features(self):
        if self.is_set:
            return self.no_python.n_features
        else:
            return None

    @property
    def X(self):
        return self.no_python.X

    @property
    def y(self):
        return self.no_python.y

    def __repr__(self):
        fit_intercept = self.fit_intercept
        n_samples = self.n_samples
        n_features = self.n_features
        r = self.__class__.__name__
        r += "(fit_intercept={fit_intercept}"\
            .format(fit_intercept=fit_intercept)
        r += ", n_samples={n_samples}".format(n_samples=n_samples)
        r += ", n_features={n_features})".format(n_features=n_features)
        return r
