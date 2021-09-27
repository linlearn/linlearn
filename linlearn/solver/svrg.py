# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``SVRG`` class, a variant of variance-reduced stochastic
gradient descent.
"""

import numpy as np
from math import fabs
from numba import jit

from ._base import Solver, OptimizationResult, jit_kwargs
from .._loss import decision_function_factory
from .._utils import np_float


class SVRG(Solver):
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
        super(SVRG, self).__init__(
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
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_features = self.n_features
        loss = self.loss
        n_classes = self.n_classes
        deriv_loss = loss.deriv_factory()
        decision_function = decision_function_factory(X, fit_intercept)
        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * self.step / n_samples

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(
                weights, inner_products, state_estimator, inner_prod1, inner_prod2
            ):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                derivative = state_estimator.loss_derivative
                deriv_tilde = state_estimator.partial_derivative

                mu = state_estimator.gradient
                for k in range(n_classes):
                    for j in range(n_features + 1):
                        mu[j, k] = 0.0
                w_new = weights.copy()
                decision_function(weights, inner_products)

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], derivative)
                    for k in range(n_classes):
                        mu[0, k] += derivative[k]
                        for j in range(n_features):
                            mu[j + 1, k] += X[i, j] * derivative[k]
                for k in range(n_classes):
                    for j in range(n_features + 1):
                        mu[j, k] /= n_samples

                deriv_new = derivative  # renaming
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    for k in range(n_classes):
                        inner_prod1[k] = w_new[0, k]
                        inner_prod2[k] = weights[0, k]
                        for j in range(n_features):
                            inner_prod1[k] += X[ind, j] * w_new[j, k]
                            inner_prod2[k] += X[ind, j] * weights[j, k]

                    deriv_loss(y[ind], inner_prod1, deriv_new)
                    deriv_loss(y[ind], inner_prod2, deriv_tilde)

                    for k in range(n_classes):
                        deriv_new[k] -= deriv_tilde[k]
                        w_new[0, k] -= step * (deriv_new[k] + mu[0, k])
                        for j in range(n_features):
                            w_new[j + 1, k] -= step * (
                                X[ind, j] * deriv_new[k] + mu[j + 1, k]
                            )
                            w_new[j + 1, k] = penalize(w_new[j + 1, k], scaled_step)

                for k in range(n_classes):
                    for j in range(n_features + 1):
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

        else:
            # There is no intercept, so the code changes slightly
            @jit(**jit_kwargs)
            def cycle(
                weights, inner_products, state_estimator, inner_prod1, inner_prod2
            ):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                mu = state_estimator.gradient
                loss_derivative = state_estimator.loss_derivative
                deriv_tilde = state_estimator.partial_derivative
                w_new = weights.copy()

                decision_function(weights, inner_products)
                for k in range(n_classes):
                    for j in range(n_features):
                        mu[j, k] = 0.0
                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], loss_derivative)
                    for k in range(n_classes):
                        for j in range(n_features):
                            mu[j, k] += (
                                X[i, j] * loss_derivative[k]
                            )  # np.outer(X[i], loss_derivative)
                for k in range(n_classes):
                    for j in range(n_features):
                        mu[j, k] /= n_samples

                deriv_new = loss_derivative  # renaming
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    for k in range(n_classes):
                        inner_prod1[k] = 0.0
                        inner_prod2[k] = 0.0
                        for j in range(n_features):
                            inner_prod1[k] += X[ind, j] * w_new[j, k]
                            inner_prod2[k] += X[ind, j] * weights[j, k]

                    deriv_loss(y[ind], inner_prod1, deriv_new)
                    deriv_loss(y[ind], inner_prod2, deriv_tilde)

                    for k in range(n_classes):
                        deriv_new[k] -= deriv_tilde[k]
                        for j in range(n_features):
                            w_new[j, k] -= step * (X[ind, j] * deriv_new[k] + mu[j, k])
                            w_new[j, k] = penalize(w_new[j, k], scaled_step)

                for k in range(n_classes):
                    for j in range(n_features):
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
        inner_prod1 = np.empty(self.n_classes, dtype=np_float)
        inner_prod2 = np.empty(self.n_classes, dtype=np_float)
        weights = np.empty(self.weights_shape, dtype=np_float)
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
            cycle(weights, inner_products, state_estimator, inner_prod1, inner_prod2)
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)
            decision_function(weights, inner_products)

        history.update(weights)

        for n_iter in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(
                weights, inner_products, state_estimator, inner_prod1, inner_prod2
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
