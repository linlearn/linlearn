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
from .._loss import decision_function_factory, batch_decision_function_factory
from .._utils import np_float, hardthresh


class LLC19(Solver):
    def __init__(
        self,
        X,
        y,
        loss,
        n_classes,
        fit_intercept,
        estimator,
        max_iter,
        tol,
        step,
        history,
        sparsity_ub=None,
    ):
        super(LLC19, self).__init__(
            X=X,
            y=y,
            loss=loss,
            n_classes=n_classes,
            fit_intercept=fit_intercept,
            estimator=estimator,
            penalty="none",
            max_iter=max_iter,
            tol=tol,
            history=history,
        )

        # Automatic steps
        self.step = step
        self.sparsity_ub = sparsity_ub or int(X.shape[1] / 100)

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept

        n_features = self.n_features
        n_samples = self.n_samples
        n_classes = self.n_classes
        grad_estimator = self.estimator.grad_factory()
        decision_function = decision_function_factory(X, fit_intercept)

        step = self.step
        sparsity_ub = self.sparsity_ub

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                decision_function(weights, inner_products)

                grad_estimator(inner_products, state_estimator)

                grad = state_estimator.gradient
                # TODO : allocate w_new somewhere ?

                w_new = weights - step * grad
                hardthresh(w_new, sparsity_ub)

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
                        # Update the maximum update change
                        abs_delta_j = fabs(w_new[j + 1, k] - weights[j + 1, k])
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        # Update the maximum weight
                        abs_w_j_new = fabs(w_new[j + 1, k])
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new

                        weights[j + 1, k] = w_new[j + 1, k]

                return max_abs_delta, max_abs_weight, n_samples

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
                hardthresh(w_new, sparsity_ub)

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

                return max_abs_delta, max_abs_weight, n_samples

            return cycle

    def solve(self, w0=None, dummy_first_step=False):
        X = self.X
        fit_intercept = self.fit_intercept
        inner_products = np.empty((self.n_samples, self.n_classes), dtype=np_float, order="F")
        coordinates = np.arange(self.weights_shape[0], dtype=np.intp)
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

        # Get the cycle function
        cycle = self.cycle_factory()
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

        history.update(weights, 0)

        for n_iter in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight, sc_prods = cycle(
                coordinates, weights, inner_products, state_estimator
            )
            # Compute the new value of objective
            # obj = objective(weights, inner_products)
            if max_abs_weight == 0.0:
                current_tol = 0.0
            else:
                current_tol = max_abs_delta / max_abs_weight

            # TODO: tester tous les cas "max_abs_weight == 0.0" etc..
            # history.update(epoch=n_iter, obj=obj, tol=current_tol, update_bar=True)
            history.update(weights, sc_prods)

            if current_tol < tol:
                history.close_bar()
                return OptimizationResult(
                    w=weights, n_iter=n_iter, success=True, tol=tol, message=None
                )

        history.close_bar()
        if tol > 0:
            warn("Maximum iteration number reached, solver may not have converged")
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
