from numba import jitclass
from numba.types import int64, float64, boolean
from linlearn.model.model import Model
from linlearn.model.utils import loss_batch, grad_batch, grad_sample_coef


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
class LeastSquaresNoPython(object):

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
        return 0.5 * (y - z) ** 2

    def derivative(self, y, z):
        return z - y

    def loss_batch(self, w):
        return loss_batch(self, w)

    def grad_batch(self, w, out):
        grad_batch(self, w, out)

    def grad_sample_coef(self, i, w):
        return grad_sample_coef(self, i, w)


class LeastSquares(Model):

    def __init__(self, fit_intercept=True):
        Model.__init__(self, LeastSquaresNoPython, fit_intercept)
