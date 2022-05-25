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
from .._utils import np_float, softthresh, hardthresh, omega, grad_omega, prox


@jit(**jit_kwargs)
def h(uu, lamda, p):
    stu = softthresh(uu, lamda)
    a = np.power(stu, 1 / (p - 1)).sum()
    b = np.power(stu, p / (p - 1)).sum() ** (1 - 2 / p)
    return a * b


# @jit(**jit_kwargs)
# def omega(th, p, C):
#     return C * ((np.power(np.abs(th), p).sum()) ** (2/p))
#
# @jit(**jit_kwargs)
# def grad_omega(th, p, C):
#     return 2 * C * ((np.linalg.norm(th.flatten(), p))**(2-p)) * np.sign(th) * np.power(np.abs(th), p-1)
#
#
# @jit(**jit_kwargs)
# def prox(u, R, p, C):
#     # first figure out lambda
#
#     lamda1, lamda2 = 0, np.max(np.abs(u))
#     while np.abs(lamda2 - lamda1) > 1e-5:
#         mid = (lamda1 + lamda2) / 2
#         if h(u, mid, p) > R:
#             lamda1 = mid
#         else:
#             lamda2 = mid
#     lamda = lamda1
#     stu = softthresh(u, lamda)
#     return - np.sign(u) * np.power(stu, 1 / (p - 1)) / (
#             (2 * C) * (np.linalg.norm(stu.flatten(), p / (p - 1)) ** ((2 - p) / (p - 1))))


class MD(Solver):
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
        stage_length,
        R,
        sparsity_ub=None,
    ):
        #assert n_classes == 1
        super(MD, self).__init__(
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

        self.R = R
        self.stage_length = stage_length
        d = self.n_features
        self.sparsity_ub = sparsity_ub or int(d/100)
        self.p = 1 + 1/np.log(d)
        p = self.p
        self.dgf_factor = (np.exp(1)/2)*np.log(d)*(d ** ((p-1)*(2-p)/p))

        self.step = step

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept

        n_features = self.n_features
        n_samples = self.n_samples
        n_classes = self.n_classes
        grad_estimator = self.estimator.grad_factory()
        decision_function = decision_function_factory(X, fit_intercept)
        R = self.R
        p = self.p
        C = self.dgf_factor
        step = self.step

        if self.estimator == "llm":
            @jit(**jit_kwargs)
            def step_scaler(state):
                return 1 / np.sqrt(1 + state.n_grad_calls)
        else:
            @jit(**jit_kwargs)
            def step_scaler(state):
                return 1.0

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(w0, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0

                decision_function(weights, inner_products)

                grad_estimator(inner_products, state_estimator)

                grad = state_estimator.gradient
                # TODO : allocate w_new somewhere ?

                g_omega = grad_omega(weights - w0, p, C)

                w_new = w0 + prox(step * step_scaler(state_estimator) * grad - g_omega, R, p, C)

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

                #weights[:] = w_new[:]

                return max_abs_delta, max_abs_weight

            return cycle

        else:
            # There is no intercept, so the code changes slightly
            @jit(**jit_kwargs)
            def cycle(w0, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                decision_function(weights, inner_products)

                grad_estimator(inner_products, state_estimator)
                grad = state_estimator.gradient
                # TODO : allocate w_new somewhere ?
                w_new = w0 + prox(step * step_scaler(state_estimator) * grad - grad_omega(weights - w0, p, C), R, p, C)

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
        weights = np.empty(self.weights_shape, dtype=np_float)
        sample_indices = np.arange(self.n_samples, dtype=np.uintp)
        tol = self.tol
        max_iter = self.max_iter
        stage_length = self.stage_length
        history = self.history
        sb = self.sparsity_ub
        print("called MD solver")
        print("stage length is %d " % stage_length)
        if w0 is not None:
            weights[:] = w0
        else:
            weights.fill(0.0)

        # Computation of the initial inner products
        decision_function = decision_function_factory(X, fit_intercept)
        decision_function(weights, inner_products)

        # Get the cycle function
        cycle = self.cycle_factory()
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

        history.update(weights, 0)

        n_iter = 0
        while n_iter + stage_length <= max_iter:
            print("starting stage %d" % (n_iter // stage_length))

            for t in range(stage_length):
                max_abs_delta, max_abs_weight = cycle(
                    w0, weights, inner_products, state_estimator
                )

                # Compute the new value of objective
                # obj = objective(weights, inner_products)
                if max_abs_weight == 0.0:
                    current_tol = 0.0
                else:
                    current_tol = max_abs_delta / max_abs_weight

                # TODO: tester tous les cas "max_abs_weight == 0.0" etc..
                # history.update(epoch=n_iter, obj=obj, tol=current_tol, update_bar=True)
                history.update(weights, 0)

                if current_tol < tol:
                    history.close_bar()
                    return OptimizationResult(
                        w=weights, n_iter=n_iter, success=True, tol=tol, message=None
                    )
                n_iter += 1

            weights = hardthresh(weights, sb)
            w0[:] = weights[:]


        history.close_bar()
        if tol > 0:
            warn("Maximum iteration number reached, solver may not have converged")
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )


