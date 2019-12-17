import numpy as np
from linlearn.utils import check_X_y, _check_sample_weight
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
    if model.sample_weight.size == 0:
        return model.value(model.y[i], z)
    else:
        return model.sample_weight[i] * model.value(model.y[i], z)


@njit
def loss_batch(model, w):
    out = 0.
    # TODO: allocate this in fit
    n_samples = model.y.shape[0]
    Xw = np.empty(n_samples)
    # TODO: inner_prods or for loop ? No need for Xw
    Xw = inner_prods(model.X, model.fit_intercept, w, Xw)
    if model.sample_weight.size == 0:
        for i in range(n_samples):
            out += model.loss(model.y[i], Xw[i]) / n_samples
    else:
        for i in range(n_samples):
            out += model.sample_weight[i] * model.loss(model.y[i], Xw[i]) \
                   / n_samples
    return out


@njit
def grad_sample_coef(model, i, w):
    z = inner_prod(model.X, model.fit_intercept, i, w)
    if model.sample_weight.size == 0:
        return model.derivative(model.y[i], z)
    else:
        return model.sample_weight[i] * model.derivative(model.y[i], z)


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

    def set(self, X, y, sample_weight=None):
        estimator = self.__class__.__name__

        X, y = check_X_y(
            X, y, accept_sparse=False, accept_large_sparse=True,
            dtype=['float64'], order='C', copy=False, force_all_finite=True,
            ensure_2d=True, allow_nd=False, multi_output=False,
            ensure_min_samples=1, ensure_min_features=1, y_numeric=True,
            estimator=estimator)

        # For now, we must ensure that dtype of labels if float64
        if y.dtype != 'float64':
            y = y.astype(np.float64)

        if sample_weight is None:
            # Use an empty np.array if no sample_weight is used
            sample_weight = np.empty(0, dtype=np.float64)
        else:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=np.float64)
            pass
        self.no_python.set(X, y, sample_weight)
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
