"""
This module implement the `GD` class for (batch) gradient descent.
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from math import fabs
from numba import jit

from .._loss import decision_function_factory
from . import Solver
from ._base import _jit_kwargs


class GD(Solver):
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
        step,
        history,
    ):
        super(GD, self).__init__(
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
        self.step = step

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_weights = self.n_weights
        grad_estimator = self.estimator.grad_factory()
        decision_function = decision_function_factory(X, fit_intercept)
        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * self.step

        if fit_intercept:

            @jit(**_jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):

                decision_function(weights, inner_products)

                grad_estimator(inner_products, state_estimator)
                grad = state_estimator.gradient
                w_new = weights - step * grad

                max_abs_delta = fabs(w_new[0] - weights[0])
                max_abs_weight = fabs(w_new[0])

                for j in range(1, n_weights):
                    w_new[j] = penalize(w_new[j], scaled_step)
                    # Update the maximum update change
                    abs_delta_j = fabs(w_new[j] - weights[j])
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_new[j])
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                weights[:] = w_new

                return max_abs_delta, max_abs_weight

            return cycle

        else:
            # There is no intercept, so the code changes slightly
            @jit(**_jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                decision_function(weights, inner_products)

                grad_estimator(inner_products, state_estimator)
                grad = state_estimator.gradient
                w_new = weights - step * grad
                for j in coordinates:
                    w_new[j] = penalize(w_new[j], scaled_step)
                    # Update the maximum update change
                    abs_delta_j = fabs(w_new[j] - weights[j])
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_new[j])
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                weights[:] = w_new
                return max_abs_delta, max_abs_weight

            return cycle
