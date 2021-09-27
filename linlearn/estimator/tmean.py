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
from .._utils import np_float

StateTMean = namedtuple(
    "StateTMean", ["deriv_samples", "deriv_samples_outer_prods", "gradient"]
)


class TMean(Estimator):
    """Trimmed-mean estimator"""

    def __init__(self, X, y, loss, fit_intercept, percentage):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.percentage = percentage
        # Number of samples excluded from both tails (left and right)
        self.n_excluded_tails = int(self.n_samples * percentage / 2)

    def get_state(self):
        return StateTMean(
            deriv_samples=np.empty(self.n_samples, dtype=np_float),
            deriv_samples_outer_prods=np.empty(self.n_samples, dtype=np_float),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
        )

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                if j == 0:
                    for i in range(n_samples):
                        deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                else:
                    for i in range(n_samples):
                        deriv_samples[i] = (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )

                # TODO: Hand-made mean ?
                # TODO: Try out different sorting mechanisms, since at some point the
                #  sorting order won't change much...
                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i]) * X[i, j]

                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)

                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])

                deriv_samples_outer_prods[:] = deriv_samples
                deriv_samples_outer_prods.sort()

                gradient[0] = np.mean(
                    deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                )
                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    deriv_samples_outer_prods.sort()
                    gradient[j + 1] = np.mean(
                        deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                    )

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)

                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])

                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    deriv_samples_outer_prods.sort()
                    gradient[j] = np.mean(
                        deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                    )

            return grad
