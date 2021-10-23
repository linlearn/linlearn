# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``GD`` class, for gradient descent solver.
"""

import numpy as np
from math import fabs
from warnings import warn
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
        n_classes,
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
        self.step = step

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept

        n_features = self.n_features
        n_samples = self.n_samples
        n_classes = self.n_classes
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
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                decision_function(weights, inner_products)
                # for k in range(n_classes):
                #     for i in range(n_samples):
                #         inner_products[i, k] = weights[0,k]
                #         for j in range(n_features):
                #             inner_products[i, k] += X[i, j] * weights[j+1, k]

                grad_estimator(inner_products, state_estimator)
                grad = state_estimator.gradient
                # TODO : allocate w_new somewhere ?
                w_new = weights - step * grad

                for k in range(n_classes):
                    abs_delta_j = fabs(w_new[0, k] - weights[0, k])
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_new[0, k])
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                    weights[0, k] = w_new[0, k]
                    for j in range(n_features):

                        w_new[j + 1, k] = penalize(w_new[j + 1, k], scaled_step)
                        # Update the maximum update change
                        abs_delta_j = fabs(w_new[j + 1, k] - weights[j + 1, k])
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_new[j + 1, k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        weights[j + 1, k] = w_new[j + 1, k]

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
                # TODO : allocate w_new somewhere ?
                w_new = weights - step * grad

                for k in range(n_classes):
                    for j in range(n_features):

                        w_new[j, k] = penalize(w_new[j, k], scaled_step)
                        # Update the maximum update change
                        abs_delta_j = fabs(w_new[j, k] - weights[j, k])
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_new[j, k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        weights[j, k] = w_new[j, k]

                return max_abs_delta, max_abs_weight

            return cycle


class batch_GD(Solver):
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
        step,
        history,
        batch_size=1.0,
    ):
        super(batch_GD, self).__init__(
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
        self.step = step
        self.batch_size = batch_size

    def cycle_factory(self):

        X = self.X
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_classes = self.n_classes
        n_features = self.n_features
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
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                np.random.shuffle(sample_indices)

                decision_function(weights, inner_products)

                grad = state_estimator.gradient
                deriv = state_estimator.loss_derivative

                for k in range(n_classes):
                    for j in range(n_features + 1):
                        grad[j, k] = 0.0
                for i in range(n_samples_batch):
                    ind = sample_indices[i]
                    deriv_loss(y[ind], inner_products[ind], deriv)
                    for k in range(n_classes):
                        grad[0, k] += deriv[k]
                        for j in range(n_features):
                            grad[j + 1, k] += (
                                deriv[k] * X[ind, j]
                            )  # np.outer(X[i], deriv)
                for k in range(n_classes):
                    for j in range(n_features + 1):
                        grad[j, k] /= n_samples_batch
                # TODO : allocate w_new somewhere ?
                w_new = weights - step * grad

                for k in range(n_classes):
                    abs_delta_j = fabs(w_new[0, k] - weights[0, k])
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_new[0, k])
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                    weights[0, k] = w_new[0, k]
                    for j in range(n_features):

                        w_new[j + 1, k] = penalize(w_new[j + 1, k], scaled_step)
                        # Update the maximum update change
                        abs_delta_j = fabs(w_new[j + 1, k] - weights[j + 1, k])
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_new[j + 1, k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        weights[j + 1, k] = w_new[j + 1, k]

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
                deriv = state_estimator.loss_derivative
                for k in range(n_classes):
                    for j in range(n_features):
                        grad[j, k] = 0.0

                for i in range(n_samples_batch):
                    ind = sample_indices[i]
                    deriv_loss(y[ind], inner_products[ind], deriv)
                    for k in range(n_classes):
                        for j in range(n_features):
                            grad[j, k] += deriv[k] * X[ind, j]  # np.outer(X[i], deriv)
                for k in range(n_classes):
                    for j in range(n_features):
                        grad[j, k] /= n_samples_batch

                w_new = weights - step * grad

                for k in range(n_classes):
                    for j in range(n_features):
                        w_new[j, k] = penalize(w_new[j, k], scaled_step)
                        # Update the maximum update change
                        abs_delta_j = fabs(w_new[j, k] - weights[j, k])
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_new[j, k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        weights[j, k] = w_new[j, k]

                return max_abs_delta, max_abs_weight

            return cycle

    def solve(self, w0=None, dummy_first_step=False):
        X = self.X
        fit_intercept = self.fit_intercept
        inner_products = np.empty((self.n_samples, self.n_classes), dtype=np_float)
        weights = np.empty(self.weights_shape, dtype=np_float)
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
            cycle(sample_indices, weights, inner_products, state_estimator)
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
        if tol > 0:
            warn("Maximum iteration number reached, solver may have not converged")
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
