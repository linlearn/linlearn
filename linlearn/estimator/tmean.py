# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the ``TMean`` class for the trimmed-means robust estimator.

`StateTMean` is a place-holder for the TMean estimator containing:

"""

from collections import namedtuple
import numpy as np
from numba import jit
from ._base import Estimator, jit_kwargs
from .._utils import np_float, trimmed_mean, fast_trimmed_mean

StateTMean = namedtuple(
    "StateTMean",
    [
        "deriv_samples",
        "deriv_samples_outer_prods",
        "gradient",
        "loss_derivative",
        "partial_derivative",
    ],
)


class TMean(Estimator):
    """Trimmed-mean estimator"""

    def __init__(self, X, y, loss, n_classes, fit_intercept, percentage):
        Estimator.__init__(self, X, y, loss, n_classes, fit_intercept)
        self.percentage = percentage
        # Number of samples excluded from both tails (left and right)
        self.n_excluded_tails = int(self.n_samples * percentage / 2)

    def get_state(self):
        return StateTMean(
            deriv_samples=np.empty(
                (self.n_samples, self.n_classes), dtype=np_float, order="F"
            ),
            deriv_samples_outer_prods=np.empty(
                (self.n_samples, self.n_classes), dtype=np_float, order="F"
            ),
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
                order="F",
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
            partial_derivative=np.empty(self.n_classes, dtype=np_float),
        )

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_classes = self.n_classes
        n_excluded_tails = self.n_excluded_tails
        percentage = self.percentage

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                partial_derivative = state.partial_derivative
                if j == 0:
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], deriv_samples[i])
                else:
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], deriv_samples[i])
                        for k in range(n_classes):
                            deriv_samples[i, k] *= X[i, j - 1]

                # TODO: Hand-made mean ?
                # TODO: Try out different sorting mechanisms, since at some point the
                #  sorting order won't change much...
                for k in range(n_classes):
                    partial_derivative[k] = trimmed_mean(deriv_samples[:, k], n_samples, percentage)
                    # partial_derivative[k] = fast_trimmed_mean(deriv_samples[:, k], n_samples, percentage)

            return partial_deriv

        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                partial_derivative = state.partial_derivative
                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv_samples[i])
                    for k in range(n_classes):
                        deriv_samples[i, k] *= X[i, j]

                for k in range(n_classes):
                    partial_derivative[k] = trimmed_mean(deriv_samples[:, k], n_samples, percentage)
                    # partial_derivative[k] = fast_trimmed_mean(deriv_samples[:, k], n_samples, percentage)

            return partial_deriv

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        percentage = self.percentage
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv_samples[i])

                    for k in range(n_classes):
                        deriv_samples_outer_prods[i, k] = deriv_samples[i, k]

                for k in range(n_classes):

                    # gradient[0, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)
                    gradient[0, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)

                for k in range(n_classes):
                    for j in range(n_features):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                deriv_samples[i, k] * X[i, j]
                            )
                        # gradient[j + 1, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)
                        gradient[j + 1, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv_samples[i])

                for j in range(n_features):
                    for k in range(n_classes):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                    deriv_samples[i, k] * X[i, j]
                            )
                        # gradient[j, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)

                        gradient[j, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, percentage)
                return 0
            return grad
