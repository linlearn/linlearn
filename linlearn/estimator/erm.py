# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the ``ERM`` class the empirical risk minimizer estimator.

``StateERM`` is a place-holder for the ERM estimator containing:

    gradient: numpy.ndarray
        A numpy array of shape (n_weights,) containing gradients computed by the
        `grad` function returned by the `grad_factory` factory function.
"""

from collections import namedtuple
import numpy as np
from numba import jit
from ._base import Estimator, jit_kwargs
from .._utils import np_float

StateERM = namedtuple("StateERM", ["gradient"])


class ERM(Estimator):
    """Empirical risk minimization estimator. This is the standard statistical
    learning approach, that corresponds to gradients equal to the average of each
    individual sample loss. Using this estimator should match the results of standard
    linear methods from other libraries.

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
        Estimator.__init__(self, X, y, loss, fit_intercept)

    def get_state(self):
        """Returns the state of the ERM estimator, which is a place-holder used for
        computations.

        Returns
        -------
        output : StateERM
            State of the ERM estimator
        """
        return StateERM(gradient=np.empty(self.n_weights, dtype=np_float))

    def partial_deriv_factory(self):
        """Partial derivatives factory. This returns a jit-compiled function allowing to
        compute partial derivatives of the considered goodness-of-fit.

        Returns
        -------
        output : function
            A jit-compiled function allowing to compute partial derivatives.
        """
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                """Computes the partial derivative of the goodness-of-fit with
                respect to coordinate `j`, given the value of the `inner_products` and
                `state`.

                Parameters
                ----------
                j : int
                    Partial derivative is with respect to this coordinate

                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateERM
                    The state of the ERM estimator (not used here, but this allows
                    all estimators to have the same prototypes for `partial_deriv`).

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
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
                """Computes the partial derivative of the goodness-of-fit with
                respect to coordinate `j`, given the value of the `inner_products` and
                `state`.

                Parameters
                ----------
                j : int
                    Partial derivative is with respect to this coordinate

                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateERM
                    The state of the ERM estimator (not used here, but this allows
                    all estimators to have the same prototypes for `partial_deriv`).

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
                deriv_sum = 0.0
                for i in range(y.shape[0]):
                    deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j]
                return deriv_sum / n_samples

            return partial_deriv

    def grad_factory(self):
        """Gradient factory. This returns a jit-compiled function allowing to
        compute the gradient of the considered goodness-of-fit.

        Returns
        -------
        output : function
            A jit-compiled function allowing to compute gradients.
        """
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                """Computes the gradient of the goodness-of-fit, given the value of the
                 `inner_products` and `state`.

                Parameters
                ----------
                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateERM
                    The state of the ERM estimator, which contains a place-holder for
                    the returned gradient.

                Returns
                -------
                output : numpy.array
                    A numpy array of shape (n_weights,) containing the gradient.
                """
                gradient = state.gradient
                gradient.fill(0.0)
                for i in range(n_samples):
                    deriv = deriv_loss(y[i], inner_products[i])
                    gradient[0] += deriv
                    gradient[1:] += deriv * X[i]
                gradient /= n_samples

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                """Computes the gradient of the goodness-of-fit, given the value of the
                 `inner_products` and `state`.

                Parameters
                ----------
                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateERM
                    The state of the ERM estimator, which contains a place-holder for
                    the returned gradient.

                Returns
                -------
                output : numpy.array
                    A numpy array of shape (n_weights,) containing the gradient.
                """
                gradient = state.gradient
                gradient.fill(0.0)
                for i in range(n_samples):
                    gradient += deriv_loss(y[i], inner_products[i]) * X[i]
                gradient /= n_samples

            return grad
