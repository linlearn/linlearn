# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the ``SVRG`` class, a variant of variance-reduced stochastic
gradient descent.
"""

import numpy as np
from math import fabs
from numpy.random import permutation
from numba import jit

from ._base import Solver, jit_kwargs
from .._loss import decision_function_factory


class SVRG(Solver):
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
        super(SVRG, self).__init__(
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
        n_samples = self.estimator.n_samples
        n_weights = self.n_weights
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        decision_function = decision_function_factory(X, fit_intercept)
        penalize = self.penalty.apply_one_unscaled_factory()
        step = self.step / n_samples

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_step = self.penalty.strength * self.step / n_samples

        if fit_intercept:

            @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                expanded_sample = np.empty(X.shape[1] + 1)
                expanded_sample[0] = 1
                mu = np.zeros(X.shape[1] + 1)
                w_new = weights.copy()
                decision_function(weights, inner_products)
                for i in range(n_samples):
                    derivative = deriv_loss(y[i], inner_products[i])
                    mu[0] += derivative
                    mu[1:] += derivative * X[i]
                mu /= n_samples

                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    expanded_sample[1:] = X[ind]
                    deriv_new = deriv_loss(y[ind], np.dot(X[ind], w_new[1:]) + w_new[0])
                    deriv_tilde = deriv_loss(
                        y[ind], np.dot(X[ind], weights[1:]) + weights[0]
                    )
                    w_new -= step * ((deriv_new - deriv_tilde) * expanded_sample + mu)

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
            @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products, state_estimator):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                mu = np.zeros(X.shape[1])
                w_new = weights.copy()

                decision_function(weights, inner_products)
                for i in range(n_samples):
                    mu += deriv_loss(y[i], inner_products[i]) * X[i]
                mu /= n_samples

                for i in range(n_samples):
                    ind = np.random.randint(n_samples)
                    deriv_new = deriv_loss(y[ind], np.dot(X[ind], w_new))
                    deriv_tilde = deriv_loss(y[ind], np.dot(X[ind], weights))
                    w_new -= step * ((deriv_new - deriv_tilde) * X[ind] + mu)

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
