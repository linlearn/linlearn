"""
This module implement the `CGD` class for coordinate gradient descent.
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

from math import fabs
import numpy as np
from numba import jit

from . import Solver
from ._base import _jit_kwargs
from ._learning_rate import learning_rates_factory


class CGD(Solver):
    def __init__(
        self,
        X,
        y,
        *,
        loss,
        fit_intercept,
        estimator,
        penalty,
        lr_factor,
        max_iter,
        tol,
        random_state,
        history,
    ):
        Solver.__init__(
            self,
            X,
            y,
            loss=loss,
            fit_intercept=fit_intercept,
            estimator=estimator,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            history=history,
        )
        self.learning_rate_ = np.empty(self.n_weights)
        self.lr_factor = lr_factor

    def _compute_learning_rate(self):
        # Get the correct function computing the learning rates
        learning_rates = learning_rates_factory(
            self.X, self.fit_intercept, self.estimator
        )
        # Compute the learning rates. Result is saved in self.learning_rate_
        learning_rates(self.loss.lip, self.learning_rate_)
        # Apply the lr_factor (it defaults to 1.0) if necessary
        if self.lr_factor != 1.0:
            self.learning_rate_ *= self.lr_factor

    def cycle_factory(self):
        X = self.X
        matrix_type = self._matrix_type
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        partial_deriv_estimator = self.estimator.partial_deriv_factory()
        penalize = self.penalty.apply_one_unscaled_factory()
        learning_rates = self.learning_rate_

        # The learning rates scaled by the strength of the penalization (we use the
        # apply_one_unscaled penalization function)
        scaled_steps = learning_rates.copy()
        scaled_steps *= self.penalty.strength

        if matrix_type == "csc":
            X_data = self._X_data
            X_indices = self._X_indices
            X_indptr = self._X_indptr
            if fit_intercept:

                @jit(**_jit_kwargs)
                def cycle(coordinates, weights, inner_products, state_estimator):
                    """Cycle with sparse CSC matrix and intercept"""
                    max_abs_delta = 0.0
                    max_abs_weight = 0.0
                    np.random.shuffle(coordinates)
                    for j in coordinates:
                        partial_deriv_j = partial_deriv_estimator(
                            j, inner_products, state_estimator
                        )
                        w_j_new = weights[j] - learning_rates[j] * partial_deriv_j
                        if j != 0:
                            w_j_new = penalize(w_j_new, scaled_steps[j])
                        delta_j = w_j_new - weights[j]
                        abs_delta_j = fabs(delta_j)
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        abs_w_j_new = fabs(w_j_new)
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new
                        if j == 0:
                            for i in range(n_samples):
                                inner_products[i] += delta_j
                        else:
                            col_start = X_indptr[j - 1]
                            col_end = X_indptr[j]
                            for idx in range(col_start, col_end):
                                i = X_indices[idx]
                                inner_products[i] += delta_j * X_data[idx]
                        weights[j] = w_j_new
                    return max_abs_delta, max_abs_weight

                return cycle
            else:

                @jit(**_jit_kwargs)
                def cycle(coordinates, weights, inner_products, state_estimator):
                    """Cycle with sparse CSC matrix and no intercept"""
                    max_abs_delta = 0.0
                    max_abs_weight = 0.0
                    np.random.shuffle(coordinates)
                    for j in coordinates:
                        partial_deriv_j = partial_deriv_estimator(
                            j, inner_products, state_estimator
                        )
                        w_j_new = weights[j] - learning_rates[j] * partial_deriv_j
                        w_j_new = penalize(w_j_new, scaled_steps[j])
                        delta_j = w_j_new - weights[j]
                        abs_delta_j = fabs(delta_j)
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        abs_w_j_new = fabs(w_j_new)
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new
                        col_start = X_indptr[j]
                        col_end = X_indptr[j + 1]
                        for idx in range(col_start, col_end):
                            i = X_indices[idx]
                            inner_products[i] += delta_j * X_data[idx]
                        weights[j] = w_j_new
                    return max_abs_delta, max_abs_weight

                return cycle

        else:
            # TODO: if matrix_type=='f' this is optimal. If matrix_type=='c' this is
            #  slower. If matrix_type=='csr' this is slow (to be implemented). Raise
            #  appropriate warnings here or in the solver.
            if fit_intercept:

                @jit(**_jit_kwargs)
                def cycle(coordinates, weights, inner_products, state_estimator):
                    """Cycle with a dense matrix and intercept"""
                    max_abs_delta = 0.0
                    max_abs_weight = 0.0
                    np.random.shuffle(coordinates)
                    for j in coordinates:
                        partial_deriv_j = partial_deriv_estimator(
                            j, inner_products, state_estimator
                        )
                        w_j_new = weights[j] - learning_rates[j] * partial_deriv_j
                        if j != 0:
                            w_j_new = penalize(w_j_new, scaled_steps[j])
                        delta_j = w_j_new - weights[j]
                        abs_delta_j = fabs(delta_j)
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        abs_w_j_new = fabs(w_j_new)
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new
                        if j == 0:
                            for i in range(n_samples):
                                inner_products[i] += delta_j
                        else:
                            for i in range(n_samples):
                                inner_products[i] += delta_j * X[i, j - 1]
                        weights[j] = w_j_new
                    return max_abs_delta, max_abs_weight

                return cycle
            else:

                @jit(**_jit_kwargs)
                def cycle(coordinates, weights, inner_products, state_estimator):
                    max_abs_delta = 0.0
                    max_abs_weight = 0.0
                    np.random.shuffle(coordinates)
                    for j in coordinates:
                        partial_deriv_j = partial_deriv_estimator(
                            j, inner_products, state_estimator
                        )
                        w_j_new = weights[j] - learning_rates[j] * partial_deriv_j
                        w_j_new = penalize(w_j_new, scaled_steps[j])
                        delta_j = w_j_new - weights[j]
                        abs_delta_j = fabs(delta_j)
                        if abs_delta_j > max_abs_delta:
                            max_abs_delta = abs_delta_j
                        abs_w_j_new = fabs(w_j_new)
                        if abs_w_j_new > max_abs_weight:
                            max_abs_weight = abs_w_j_new
                        for i in range(n_samples):
                            inner_products[i] += delta_j * X[i, j]
                        weights[j] = w_j_new
                    return max_abs_delta, max_abs_weight

                return cycle
