# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``SAGA`` class, for stochastic gradient descent solver.
"""

import numpy as np
from math import fabs
from warnings import warn
from numpy.random import permutation
from numba import jit

from ._base import Solver, OptimizationResult, jit_kwargs
from .._utils import np_float


class SGD(Solver):
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
        exponent=0.5,
    ):
        super(SGD, self).__init__(
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
        self.exponent = exponent

    def cycle_factory(self):

        X = self.X
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        n_weights = self.n_weights
        exponent = self.exponent
        loss = self.loss
        deriv_loss = loss.deriv_factory()

        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step  # / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        penalty_strength = self.penalty.strength
        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(weights, epoch, state_estimator, inner_prod):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                # grad = state_estimator.gradient
                derivative = state_estimator.loss_derivative
                w_new = weights.copy()
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    iter_step = step / max(
                        n_samples, (1 + epoch * n_samples + i) ** exponent
                    )
                    scaled_iter_step = iter_step * penalty_strength

                    for k in range(n_classes):
                        inner_prod[k] = weights[0, k]
                        for j in range(n_features):
                            inner_prod[k] += X[ind, j] * weights[j + 1, k]

                    deriv_loss(y[ind], inner_prod, derivative)

                    # np.outer(X[ind], grad[0], grad[1:])

                    for k in range(n_classes):
                        w_new[0, k] -= iter_step * derivative[k]
                        for j in range(n_features):
                            w_new[j + 1, k] -= iter_step * X[ind, j] * derivative[k]
                            w_new[j + 1, k] = penalize(
                                w_new[j + 1, k], scaled_iter_step
                            )

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

                return max_abs_delta, max_abs_weight

            return cycle

        else:
            # There is no intercept, so the code changes slightly
            @jit(**jit_kwargs)
            def cycle(weights, epoch, state_estimator, inner_prod):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                # grad = state_estimator.gradient
                loss_derivative = state_estimator.loss_derivative
                w_new = weights.copy()
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    for k in range(n_classes):
                        inner_prod[k] = 0.0
                        for j in range(n_features):
                            inner_prod[k] += X[ind, j] * weights[j, k]

                    deriv_loss(y[ind], inner_prod, loss_derivative)
                    # np.outer(X[ind], loss_derivative, grad)
                    iter_step = step / max(
                        n_samples, (1 + epoch * n_samples + i) ** exponent
                    )
                    scaled_iter_step = iter_step * penalty_strength

                    for k in range(n_classes):
                        for j in range(n_features):
                            w_new[j, k] -= iter_step * loss_derivative[k] * X[ind, j]
                            w_new[j, k] = penalize(w_new[j, k], scaled_iter_step)

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

        weights = np.empty(self.weights_shape, dtype=np_float)
        tol = self.tol
        max_iter = self.max_iter
        inner_prod = np.empty(self.n_classes, dtype=np_float)
        history = self.history
        if w0 is not None:
            weights[:] = w0
        else:
            weights.fill(0.0)

        random_state = self.random_state
        if random_state is not None:

            @jit(**jit_kwargs)
            def numba_seed_numpy(rnd_state):
                np.random.seed(rnd_state)

            numba_seed_numpy(random_state)

        # Get the cycle function
        cycle = self.cycle_factory()

        # Get the estimator state (a place-holder for the estimator's internal
        # computations)
        state_estimator = self.estimator.get_state()

        # TODO: First value for tolerance is 1.0 or NaN
        # history.update(epoch=0, obj=obj, tol=1.0, update_bar=True)
        if dummy_first_step:
            cycle(weights, 0, state_estimator, inner_prod)
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)

        history.update(weights)

        for epoch in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(
                weights, epoch, state_estimator, inner_prod
            )
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
                    w=weights, n_iter=epoch, success=True, tol=tol, message=None
                )

        history.close_bar()
        if tol > 0:
            warn("Maximum iteration number reached, solver may have not converged")
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
