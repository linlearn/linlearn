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
