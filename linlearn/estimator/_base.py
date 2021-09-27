# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the base ``Estimator`` class for all estimators.
"""

from abc import ABC, abstractmethod
from .._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH


# Options passed to the @jit decorator within this module
jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


# Options passed to the @vectorize decorator within this module
vectorize_kwargs = {
    "nopython": NOPYTHON,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


################################################################
# Generic estimator
################################################################


class Estimator(ABC):
    """Base abstract estimator class for internal use only.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X.

    loss : Loss
        A loss class for which gradients will be estimated by the estimator.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Attributes
    ----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_weights : int
        This is `n_features` if `fit_intercept=False` and `n_features` otherwise.
    """

    def __init__(self, X, y, loss, fit_intercept):
        self.X = X
        self.y = y
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_samples, self.n_features = X.shape
        self.n_weights = self.n_features + int(self.fit_intercept)

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def partial_deriv_factory(self):
        pass

    @abstractmethod
    def grad_factory(self):
        pass
