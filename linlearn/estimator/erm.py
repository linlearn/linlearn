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

StateERM = namedtuple("StateERM", ["gradient", "loss_derivative", "partial_derivative"])


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

    def __init__(self, X, y, loss, n_classes, fit_intercept):
        Estimator.__init__(self, X, y, loss, n_classes, fit_intercept)

    def get_state(self):
        """Returns the state of the ERM estimator, which is a place-holder used for
        computations.

        Returns
        -------
        output : StateERM
            State of the ERM estimator
        """
        return StateERM(
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
            partial_derivative=np.empty(self.n_classes, dtype=np_float),
        )

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
        n_classes = self.n_classes

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
                deriv = state.loss_derivative
                partial_derivative = state.partial_derivative
                for k in range(n_classes):
                    partial_derivative[k] = 0.0
                if j == 0:
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], deriv)
                        for k in range(n_classes):
                            partial_derivative[k] += deriv[k]
                    for k in range(n_classes):
                        partial_derivative[k] /= n_samples
                else:
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], deriv)
                        for k in range(n_classes):
                            partial_derivative[k] += deriv[k] * X[i, j - 1]
                    for k in range(n_classes):
                        partial_derivative[k] /= n_samples

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
                deriv = state.loss_derivative
                partial_derivative = state.partial_derivative
                for k in range(n_classes):
                    partial_derivative[k] = 0.0
                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv)
                    for k in range(n_classes):
                        partial_derivative[k] += deriv[k] * X[i, j]
                for k in range(n_classes):
                    partial_derivative[k] /= n_samples

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
        n_features = self.n_features
        n_classes = self.n_classes

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
                deriv = state.loss_derivative
                for j in range(n_features + 1):
                    for k in range(n_classes):
                        gradient[j, k] = 0.0
                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv)
                    for k in range(n_classes):
                        gradient[0, k] += deriv[k]
                    for j in range(n_features):
                        for k in range(n_classes):
                            gradient[j + 1, k] += X[i, j] * deriv[k]
                for j in range(n_features + 1):
                    for k in range(n_classes):
                        gradient[j, k] /= n_samples

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
                deriv = state.loss_derivative
                for j in range(n_features):
                    for k in range(n_classes):
                        gradient[j, k] = 0.0

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv)
                    for j in range(n_features):
                        for k in range(n_classes):
                            gradient[j, k] += X[i, j] * deriv[k]
                for j in range(n_features):
                    for k in range(n_classes):
                        gradient[j, k] /= n_samples

            return grad
