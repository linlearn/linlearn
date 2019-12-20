from math import exp, log
from numba import njit, jitclass, vectorize
from numba.types import int64, float64, boolean
from linlearn.model.model import Model
from linlearn.model.utils import loss_batch, grad_batch, grad_sample_coef, \
    row_norm


@njit(fastmath=True)
def sigmoid(z):
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        exp_z = exp(z)
        return exp_z / (1 + exp_z)


# TODO: faster sigmoid

@vectorize(fastmath=True)
def sigmoid(z):
    if z > 0:
        return 1 / (1 + exp(-z))
    else:
        exp_z = exp(z)
        return exp_z / (1 + exp_z)


# TODO: faster logistic

@njit(fastmath=True)
def logistic(z):
    if z > 0:
        return log(1 + exp(-z))
    else:
        return -z + log(1 + exp(z))


specs = [
    ('fit_intercept', boolean),
    ('X', float64[:, ::1]),
    ('y', float64[::1]),
    ('sample_weight', float64[::1]),
    ('n_samples', int64),
    ('n_features', int64),
    ('is_set', boolean)
]
@jitclass(specs)
class LogisticNoPython(object):

    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept
        self.is_set = False

    def set(self, X, y, sample_weight):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.is_set = True
        return self

    def loss(self, y, z):
        return logistic(y * z)

    def derivative(self, y, z):
        return - y * sigmoid(-y * z)

    def loss_batch(self, w):
        return loss_batch(self, w)

    def grad_batch(self, w, out):
        grad_batch(self, w, out)

    def grad_sample_coef(self, i, w):
        return grad_sample_coef(self, i, w)


class Logistic(Model):

    def __init__(self, fit_intercept=True):
        Model.__init__(self, LogisticNoPython, fit_intercept)

    def _compute_lips(self):
        if self.sample_weight is None:
            self._lips = row_norm(self)
        else:
            self._lips = self.sample_weight * row_norm(self)
        self._lips /= 4
