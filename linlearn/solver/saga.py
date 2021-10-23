# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``SAGA`` class, a variant of variance-reduced stochastic
gradient descent.
"""

import numpy as np
from math import fabs
from warnings import warn
from numpy.random import permutation
from numba import jit

from ._base import Solver, OptimizationResult, jit_kwargs
from .._loss import decision_function_factory
from .._utils import np_float


class SAGA(Solver):
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
        super(SAGA, self).__init__(
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
        n_samples = X.shape[0]
        n_weights = self.n_weights
        n_features = self.n_features
        n_classes = self.n_classes

        deriv_loss = self.loss.deriv_factory()
        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * step

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(
                weights,
                inner_products,
                mean_grad,
                grad_update,
                loss_derivative,
                inner_prod,
                init,
            ):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                if init:
                    for k in range(n_classes):
                        for j in range(n_features + 1):
                            mean_grad[j, k] = 0.0
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], loss_derivative)
                        for k in range(n_classes):
                            mean_grad[0, k] += loss_derivative[k]
                            for j in range(n_features):
                                mean_grad[j + 1, k] += (
                                    X[i, j] * loss_derivative[k]
                                )  # np.outer(X[i], loss_derivative)
                    for k in range(n_classes):
                        for j in range(n_features + 1):
                            mean_grad[j, k] /= n_samples

                w_new = weights.copy()
                for i in range(n_samples):

                    ind = np.random.randint(n_samples)
                    for k in range(n_classes):
                        inner_prod[k] = w_new[0, k]
                        for j in range(n_features):
                            inner_prod[k] += X[ind, j] * w_new[j + 1, k]

                    deriv_loss(y[ind], inner_prod, grad_update[0])
                    deriv_loss(y[ind], inner_products[ind], loss_derivative)

                    for k in range(n_classes):
                        grad_update[0, k] -= loss_derivative[k]
                        w_new[0, k] -= step * (grad_update[0, k] + mean_grad[0, k])
                        mean_grad[0, k] += grad_update[0, k] / n_samples
                        for j in range(n_features):
                            grad_update[j + 1, k] = grad_update[0, k] * X[ind, j]
                            w_new[j + 1, k] -= step * (
                                grad_update[j + 1, k] + mean_grad[j + 1, k]
                            )
                            w_new[j + 1, k] = penalize(w_new[j + 1, k], scaled_step)
                            mean_grad[j + 1, k] += grad_update[j + 1, k] / n_samples

                        inner_products[ind, k] = inner_prod[k]

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
                weights,
                inner_products,
                mean_grad,
                grad_update,
                loss_derivative,
                inner_prod,
                init,
            ):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                if init:
                    for k in range(n_classes):
                        for j in range(n_features):
                            mean_grad[j, k] = 0.0
                    for i in range(n_samples):
                        deriv_loss(y[i], inner_products[i], loss_derivative)
                        for k in range(n_classes):
                            for j in range(n_features):
                                mean_grad[j, k] += (
                                    X[i, j] * loss_derivative[k]
                                )  # np.outer(X[i], loss_derivative)
                    for k in range(n_classes):
                        for j in range(n_features):
                            mean_grad[j, k] /= n_samples

                w_new = weights.copy()
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    for k in range(n_classes):
                        inner_prod[k] = 0.0
                        for j in range(n_features):
                            inner_prod[k] += X[ind, j] * w_new[j, k]

                    deriv_loss(y[ind], inner_prod, grad_update[0])
                    deriv_loss(y[ind], inner_products[ind], loss_derivative)
                    for k in range(n_classes):
                        loss_derivative[k] = grad_update[0, k] - loss_derivative[k]
                        for j in range(n_features):
                            grad_update[j, k] = loss_derivative[k] * X[ind, j]
                            w_new[j, k] -= step * (grad_update[j, k] + mean_grad[j, k])
                            w_new[j, k] = penalize(w_new[j, k], scaled_step)
                            mean_grad[j, k] += grad_update[j, k] / n_samples

                        inner_products[ind, k] = inner_prod[k]

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
        # We use intp and not uintp since j-1 is np.float64 when j has type np.uintp
        # (namely np.uint64 on most machines), and this fails in nopython mode for
        # coverage analysis

        weights = np.empty(self.weights_shape, dtype=np_float)
        mean_grad = np.empty(self.weights_shape, dtype=np_float)
        grad_update = np.empty(self.weights_shape, dtype=np_float)
        loss_derivative = np.empty(self.n_classes, dtype=np_float)
        inner_prod = np.empty(self.n_classes, dtype=np_float)
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

        if dummy_first_step:
            cycle(
                weights,
                inner_products,
                mean_grad,
                grad_update,
                loss_derivative,
                inner_prod,
                True,
            )
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)
            decision_function(weights, inner_products)

        # TODO: First value for tolerance is 1.0 or NaN
        # history.update(epoch=0, obj=obj, tol=1.0, update_bar=True)
        history.update(weights)
        init = True

        for n_iter in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(
                weights,
                inner_products,
                mean_grad,
                grad_update,
                loss_derivative,
                inner_prod,
                init,
            )
            init = False
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
