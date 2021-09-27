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
                    # TODO : this isn't efficient, full sorting is not necessary, use np.partition ?
                    partitioned = np.partition(deriv_samples[:, k], [n_excluded_tails, n_samples-n_excluded_tails-1])

                    #deriv_samples[:, k].sort()
                    # partial_derivative[k] = np.mean(
                    #     deriv_samples[n_excluded_tails:-n_excluded_tails, k]
                    # )
                    partial_derivative[k] = np.mean(
                        partitioned[n_excluded_tails:-n_excluded_tails]
                    )
                    partial_derivative[k] = (1 - percentage) * partial_derivative[
                        k
                    ] + percentage * (
                        partitioned[n_excluded_tails]
                        + partitioned[-n_excluded_tails - 1]
                    )

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
                    # TODO : this isn't efficient, full sorting is not necessary, use np.partition ?
                    # deriv_samples[:, k].sort()
                    # partial_derivative[k] = np.mean(
                    #     deriv_samples[n_excluded_tails:-n_excluded_tails, k]
                    # )
                    # partial_derivative[k] = (1 - percentage) * partial_derivative[
                    #     k
                    # ] + percentage * (
                    #     deriv_samples[n_excluded_tails, k]
                    #     + deriv_samples[-n_excluded_tails - 1, k]
                    # )
                    partitioned = np.partition(deriv_samples[:, k], [n_excluded_tails, n_samples-n_excluded_tails-1])

                    #deriv_samples[:, k].sort()
                    # partial_derivative[k] = np.mean(
                    #     deriv_samples[n_excluded_tails:-n_excluded_tails, k]
                    # )
                    partial_derivative[k] = np.mean(
                        partitioned[n_excluded_tails:-n_excluded_tails]
                    )
                    partial_derivative[k] = (1 - percentage) * partial_derivative[
                        k
                    ] + percentage * (
                        partitioned[n_excluded_tails]
                        + partitioned[-n_excluded_tails - 1]
                    )

                # deriv_samples.sort()
                # return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

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
                    partitioned = np.partition(deriv_samples_outer_prods[:, k], [n_excluded_tails, n_samples-n_excluded_tails-1])

                    gradient[0, k] = np.mean(
                        partitioned[n_excluded_tails:-n_excluded_tails]
                    )
                    gradient[0, k] = (1 - percentage) * gradient[0, k] + percentage * (
                        partitioned[n_excluded_tails]
                        + partitioned[-n_excluded_tails - 1]
                    )

                    # deriv_samples_outer_prods[:, k].sort()
                    # gradient[0, k] = np.mean(
                    #     deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails, k]
                    # )
                for k in range(n_classes):
                    for j in range(n_features):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                deriv_samples[i, k] * X[i, j]
                            )
                        # deriv_samples_outer_prods[:, k].sort()
                        # gradient[j + 1, k] = np.mean(
                        #     deriv_samples_outer_prods[
                        #         n_excluded_tails:-n_excluded_tails, k
                        #     ]
                        # )
                        partitioned = np.partition(deriv_samples_outer_prods[:, k], [n_excluded_tails, n_samples-n_excluded_tails-1])

                        gradient[j + 1, k] = np.mean(
                            partitioned[n_excluded_tails:-n_excluded_tails]
                        )
                        gradient[j+1, k] = (1 - percentage) * gradient[j+1, k] + percentage * (
                            partitioned[n_excluded_tails]
                            + partitioned[-n_excluded_tails - 1]
                        )

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
                        # deriv_samples_outer_prods[:, k].sort()
                        # gradient[j, k] = np.mean(
                        #     deriv_samples_outer_prods[
                        #         n_excluded_tails:-n_excluded_tails, k
                        #     ]
                        # )
                        partitioned = np.partition(deriv_samples_outer_prods[:, k],
                                                   [n_excluded_tails, n_samples - n_excluded_tails - 1])

                        gradient[j, k] = np.mean(
                            partitioned[n_excluded_tails:-n_excluded_tails]
                        )
                        gradient[j, k] = (1 - percentage) * gradient[j, k] + percentage * (
                                partitioned[n_excluded_tails]
                                + partitioned[-n_excluded_tails - 1]
                        )

            return grad
