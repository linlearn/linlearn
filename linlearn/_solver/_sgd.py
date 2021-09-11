"""
This module implement the `SGD` class for stochastic gradient descent

References
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from math import fabs
import numpy as np
from numba import jit

from . import Solver
from ._base import _jit_kwargs, OptimizationResult
from .._utils import np_float


class SGD(Solver):
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
        exponent=0.5,
    ):
        super(SGD, self).__init__(
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
        self.exponent = exponent

    def cycle_factory(self):

        X = self.X
        y = self.y
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_weights = self.n_weights
        exponent = self.exponent
        loss = self.loss
        deriv_loss = loss.deriv_factory()

        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step  # / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * self.step  # / n_samples

        if fit_intercept:

            @jit(**_jit_kwargs)
            def cycle(weights, epoch, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                grad = state_estimator.gradient
                w_new = weights.copy()
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    grad[0] = 1
                    grad[1:] = X[ind]
                    grad *= deriv_loss(y[ind], np.dot(X[ind], weights[1:]) + weights[0])

                    w_new -= step * grad / ((1 + epoch * n_samples + i) ** exponent)
                    for j in range(1, n_weights):
                        w_new[j] = penalize(
                            w_new[j],
                            scaled_step / ((1 + epoch * n_samples + i) ** exponent),
                        )
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
            def cycle(weights, epoch, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                grad = state_estimator.gradient
                w_new = weights.copy()
                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    grad[:] = deriv_loss(y[ind], np.dot(X[ind], weights)) * X[ind]

                    w_new -= step * grad / ((1 + epoch * n_samples + i) ** exponent)
                    for j in range(n_weights):
                        w_new[j] = penalize(
                            w_new[j],
                            scaled_step / ((1 + epoch * n_samples + i) ** exponent),
                        )
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

        weights = np.empty(self.n_weights, dtype=np_float)
        tol = self.tol
        max_iter = self.max_iter
        history = self.history
        if w0 is not None:
            weights[:] = w0
        else:
            weights.fill(0.0)

        random_state = self.random_state
        if random_state is not None:

            @jit(**_jit_kwargs)
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
            cycle(weights, 0, state_estimator)
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)

        history.update(weights)

        for epoch in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(weights, epoch, state_estimator)
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
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
