# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``CGD`` class, for the coordinate gradient descent solver.
"""

import numpy as np
from math import fabs
from numpy.random import permutation
from numba import jit

from ._base import Solver, jit_kwargs
from .._utils import rand_choice_nb


class CGD(Solver):
    def __init__(
        self,
        X,
        y,
        loss,
        n_classes,
        fit_intercept,
        estimator,
        penalty,
        max_iter,
        tol,
        random_state,
        steps,
        history,
        importance_sampling=False,
    ):
        super(CGD, self).__init__(
            X=X,
            y=y,
            loss=loss,
            n_classes=n_classes,
            fit_intercept=fit_intercept,
            estimator=estimator,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            history=history,
        )

        # Automatic steps
        self.steps = steps
        self.importance_sampling = importance_sampling

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_classes = self.n_classes
        weights_dim1 = self.weights_shape[0]
        partial_deriv_estimator = self.estimator.partial_deriv_factory()
        penalize = self.penalty.apply_one_unscaled_factory()
        steps = self.steps

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_steps = self.steps.copy()
        scaled_steps *= self.penalty.strength

        if self.importance_sampling:
            coord_csum_probas = np.cumsum(1 / self.steps)
            coord_csum_probas /= coord_csum_probas[-1]

            @jit(**jit_kwargs)
            def prepare_coordinates(coords):
                rand_choice_nb(weights_dim1, coord_csum_probas, coords)

        else:

            @jit(**jit_kwargs)
            def prepare_coordinates(coords):
                np.random.shuffle(coords)

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                # weights = state_cgd.weights
                # inner_products = state_cgd.inner_products
                # for idx in range(n_weights):
                #     coordinates[idx] = idx
                prepare_coordinates(coordinates)
                # np.random.shuffle(coordinates)
                w_j_new = state_estimator.loss_derivative
                delta_j = state_estimator.partial_derivative
                for j in coordinates:
                    partial_deriv_estimator(j, inner_products, state_estimator)

                    for k in range(n_classes):
                        w_j_new[k] = weights[j, k] - steps[j] * delta_j[k]
                    if j != 0:
                        # It's not the intercept so we penalize
                        # TODO: compute the
                        for k in range(n_classes):
                            w_j_new[k] = penalize(w_j_new[k], scaled_steps[j])

                    # Update the inner products
                    for k in range(n_classes):
                        delta_j[k] = w_j_new[k] - weights[j, k]
                        # Update the maximum update change
                        abs_delta_j = fabs(delta_j[k])

                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_j_new[k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        if j == 0:
                            for i in range(n_samples):
                                inner_products[i, k] += delta_j[k]
                        else:
                            for i in range(n_samples):
                                inner_products[i, k] += delta_j[k] * X[i, j - 1]

                    for k in range(n_classes):
                        weights[j, k] = w_j_new[k]

                return max_abs_delta, max_abs_weight

            return cycle

        else:
            # There is no intercept, so the code changes slightly
            @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                # for idx in range(n_weights):
                #     coordinates[idx] = idx
                prepare_coordinates(coordinates)
                # np.random.shuffle(coordinates)
                # use available place holders in estimator state to avoid allocation
                w_j_new = state_estimator.loss_derivative
                delta_j = state_estimator.partial_derivative
                for j in coordinates:

                    partial_deriv_estimator(j, inner_products, state_estimator)
                    for k in range(n_classes):
                        w_j_new[k] = weights[j, k] - steps[j] * delta_j[k]
                        w_j_new[k] = penalize(w_j_new[k], scaled_steps[j])

                    # Update the inner products
                    for k in range(n_classes):
                        delta_j[k] = w_j_new[k] - weights[j, k]
                        # Update the maximum update change
                        abs_delta_j = fabs(delta_j[k])

                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_j_new[k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        for i in range(n_samples):
                            inner_products[i, k] += delta_j[k] * X[i, j]

                    for k in range(n_classes):
                        weights[j, k] = w_j_new[k]
                return max_abs_delta, max_abs_weight

            return cycle
