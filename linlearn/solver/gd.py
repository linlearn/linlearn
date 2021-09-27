# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``GD`` class, for gradient descent solver.
"""

import numpy as np
from math import fabs
from numpy.random import permutation
from numba import jit

from ._base import Solver, OptimizationResult, jit_kwargs
from .._loss import decision_function_factory
from .._utils import np_float


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

            @jit(**jit_kwargs)
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
            @jit(**jit_kwargs)
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


class batch_GD(Solver):
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
        batch_size=1,
    ):
        super(batch_GD, self).__init__(
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
        self.batch_size = batch_size

    def cycle_factory(self):

        X = self.X
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_weights = self.n_weights
        deriv_loss = self.loss.deriv_factory()
        decision_function = decision_function_factory(X, fit_intercept)

        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step
        n_samples_batch = int(self.batch_size * n_samples)

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * self.step

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(sample_indices, weights, inner_products, state_estimator):
                np.random.shuffle(sample_indices)

                decision_function(weights, inner_products)

                grad = state_estimator.gradient
                grad.fill(0.0)
                for i in range(n_samples_batch):
                    deriv = deriv_loss(y[i], inner_products[i])
                    grad[0] += deriv
                    grad[1:] += deriv * X[i]
                grad /= n_samples

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
            @jit(**jit_kwargs)
            def cycle(sample_indices, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                np.random.shuffle(sample_indices)

                decision_function(weights, inner_products)

                grad = state_estimator.gradient
                grad.fill(0.0)
                for i in range(n_samples_batch):
                    grad += deriv_loss(y[i], inner_products[i]) * X[i]
                grad /= n_samples

                w_new = weights - step * grad
                for j in range(n_weights):
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

    def solve(self, w0=None, dummy_first_step=False):
        X = self.X
        fit_intercept = self.fit_intercept
        inner_products = np.empty(self.n_samples, dtype=np_float)
        coordinates = np.arange(self.n_weights, dtype=np.uintp)
        weights = np.empty(self.n_weights, dtype=np_float)
        sample_indices = np.arange(self.n_samples, dtype=np.uintp)
        tol = self.tol
        max_iter = self.max_iter
        history = self.history
        if w0 is not None:
            weights[:] = w0
        else:
            weights.fill(0.0)

        # Computation of the initial inner products
        decision_function = decision_function_factory(X, fit_intercept)
        decision_function(weights, inner_products)

        random_state = self.random_state
        if random_state is not None:

            @jit(**jit_kwargs)
            def numba_seed_numpy(rnd_state):
                np.random.seed(rnd_state)

            numba_seed_numpy(random_state)

        # Get the cycle function
        cycle = self.cycle_factory()
        # Get the objective function
        # objective = self.objective_factory()
        # # Compute the first value of the objective
        # obj = objective(weights, inner_products)
        # Get the estimator state (a place-holder for the estimator's internal
        # computations)
        state_estimator = self.estimator.get_state()

        # TODO: First value for tolerance is 1.0 or NaN
        # history.update(epoch=0, obj=obj, tol=1.0, update_bar=True)
        if dummy_first_step:
            cycle(coordinates, weights, inner_products, state_estimator)
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)
            decision_function(weights, inner_products)

        history.update(weights)

        for n_iter in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(
                sample_indices, weights, inner_products, state_estimator
            )
            # Compute the new value of objective
            # obj = objective(weights, inner_products)
            if max_abs_weight == 0.0:
                current_tol = 0.0
            else:
                current_tol = max_abs_delta / max_abs_weight

            # TODO: tester tous les cas "max_abs_weight == 0.0" etc..
            # history.update(epoch=n_iter, obj=obj, tol=current_tol, update_bar=True)
            history.update(weights)

            if current_tol < tol:
                history.close_bar()
                return OptimizationResult(
                    w=weights, n_iter=n_iter, success=True, tol=tol, message=None
                )

        history.close_bar()
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
