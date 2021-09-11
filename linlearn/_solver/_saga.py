"""
This module implement the `SAGA` class for stochastic average gradient

References
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from math import fabs
import numpy as np
from numba import jit

from .._loss import decision_function_factory
from . import Solver
from ._base import _jit_kwargs, OptimizationResult
from .._utils import np_float


class SAGA(Solver):
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
        super(SAGA, self).__init__(
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
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = X.shape[0]
        n_weights = self.n_weights

        deriv_loss = self.loss.deriv_factory()
        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * step

        if fit_intercept:

            @jit(**_jit_kwargs)
            def cycle(weights, inner_products, mean_grad, grad_update, init):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                if init:
                    mean_grad.fill(0.0)
                    for i in range(n_samples):
                        deriv = deriv_loss(y[i], inner_products[i])
                        mean_grad[0] += deriv
                        mean_grad[1:] += deriv * X[i]
                    mean_grad /= n_samples

                w_new = weights.copy()
                for i in range(n_samples):

                    j = np.random.randint(n_samples)
                    new_j_inner_prod = np.dot(w_new[1:], X[j]) + w_new[0]
                    grad_update[0] = 1
                    grad_update[1:] = X[j]
                    grad_update *= deriv_loss(y[j], new_j_inner_prod) - deriv_loss(
                        y[j], inner_products[j]
                    )

                    w_new -= step * (grad_update + mean_grad)

                    mean_grad += grad_update / n_samples
                    inner_products[j] = new_j_inner_prod

                    for j in range(1, n_weights):
                        w_new[j] = penalize(w_new[j], scaled_step)
                for j in range(n_weights):
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
            def cycle(weights, inner_products, mean_grad, grad_update, init):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                if init:
                    mean_grad.fill(0.0)
                    for i in range(n_samples):
                        mean_grad += deriv_loss(y[i], inner_products[i]) * X[i]
                    mean_grad /= n_samples

                w_new = weights.copy()
                for i in range(n_samples):

                    j = np.random.randint(n_samples)
                    new_j_inner_prod = np.dot(w_new, X[j])

                    grad_update[:] = (
                        deriv_loss(y[j], new_j_inner_prod)
                        - deriv_loss(y[j], inner_products[j])
                    ) * X[j]
                    inner_products[j] = new_j_inner_prod

                    w_new -= step * (grad_update + mean_grad)
                    mean_grad += grad_update / n_samples

                    for j in range(n_weights):
                        w_new[j] = penalize(w_new[j], scaled_step)
                for j in range(n_weights):
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
        # We use intp and not uintp since j-1 is np.float64 when j has type np.uintp
        # (namely np.uint64 on most machines), and this fails in nopython mode for
        # coverage analysis

        weights = np.empty(self.n_weights, dtype=np_float)
        mean_grad = np.empty(self.n_weights, dtype=np_float)
        grad_update = np.empty(self.n_weights, dtype=np_float)
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

            @jit(**_jit_kwargs)
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
            cycle(weights, inner_products, mean_grad, grad_update, True)
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
                weights, inner_products, mean_grad, grad_update, init
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
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
