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
        n_weights = self.n_weights
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
                rand_choice_nb(n_weights, coord_csum_probas, coords)

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

                for j in coordinates:
                    partial_deriv_j = partial_deriv_estimator(
                        j, inner_products, state_estimator
                    )

                    if j == 0:
                        # It's the intercept, so we don't penalize
                        w_j_new = weights[j] - steps[j] * partial_deriv_j
                    else:
                        # It's not the intercept
                        w_j_new = weights[j] - steps[j] * partial_deriv_j
                        # TODO: compute the
                        w_j_new = penalize(w_j_new, scaled_steps[j])

                    # Update the inner products
                    delta_j = w_j_new - weights[j]
                    # Update the maximum update change
                    abs_delta_j = fabs(delta_j)

                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_j_new)
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                    if j == 0:
                        for i in range(n_samples):
                            inner_products[i] += delta_j
                    else:
                        for i in range(n_samples):
                            inner_products[i] += delta_j * X[i, j - 1]
                    weights[j] = w_j_new

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

                for j in coordinates:

                    partial_deriv_j = partial_deriv_estimator(
                        j, inner_products, state_estimator
                    )
                    w_j_new = weights[j] - steps[j] * partial_deriv_j
                    w_j_new = penalize(w_j_new, scaled_steps[j])
                    # Update the inner products
                    delta_j = w_j_new - weights[j]
                    # Update the maximum update change
                    abs_delta_j = fabs(delta_j)
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_j_new)
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                    for i in range(n_samples):
                        inner_products[i] += delta_j * X[i, j]

                    weights[j] = w_j_new
                return max_abs_delta, max_abs_weight

            return cycle
