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
        self.n_excluded_tails = max(1, int(len(X) * percentage))
        self.one_hot_cols = np.sum(X == 0.0, axis=0) > X.shape[0]/20
        if fit_intercept:
            self.one_hot_cols = np.insert(self.one_hot_cols, 0, False)


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
        one_hot_cols = self.one_hot_cols
        n_excluded_tails = self.n_excluded_tails

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

                if one_hot_cols[j]:
                    for k in range(n_classes):
                        partial_derivative[k] = trimmed_mean(deriv_samples[:, k], n_samples, n_excluded_tails)
                else:
                    for k in range(n_classes):
                        partial_derivative[k] = fast_trimmed_mean(deriv_samples[:, k], n_samples, n_excluded_tails)

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

                if one_hot_cols[j]:
                    for k in range(n_classes):
                        partial_derivative[k] = trimmed_mean(deriv_samples[:, k], n_samples, n_excluded_tails)
                else:
                    for k in range(n_classes):
                        partial_derivative[k] = fast_trimmed_mean(deriv_samples[:, k], n_samples, n_excluded_tails)

            return partial_deriv

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        one_hot_cols = self.one_hot_cols
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

                    gradient[0, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)
                    # gradient[0, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)

                for k in range(n_classes):
                    for j in range(n_features):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                deriv_samples[i, k] * X[i, j]
                            )
                        if one_hot_cols[j]:
                            gradient[j + 1, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)
                        else:
                            gradient[j + 1, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)

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
                    # if one_hot_cols[j]:
                    #     tmean_fct = trimmed_mean
                    # else:
                    #     tmean_fct = fast_trimmed_mean
                    for k in range(n_classes):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                    deriv_samples[i, k] * X[i, j]
                            )

                        if one_hot_cols[j]:
                            gradient[j, k] = trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)
                        else:
                            gradient[j, k] = fast_trimmed_mean(deriv_samples_outer_prods[:, k], n_samples, n_excluded_tails)

                return 0
            return grad
