# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implements the class `Solver`, which is a parent abstract class for all
the solvers defined in the `linlearn._solver` module.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from numba import jit

from .._loss import decision_function_factory
from .._utils import (
    NOPYTHON,
    NOGIL,
    BOUNDSCHECK,
    FASTMATH,
    np_float,
    matrix_type,
)

_jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


OptimizationResult = namedtuple(
    "OptimizationResult", ["n_iter", "tol", "success", "w", "message"]
)


class Solver(ABC):
    def __init__(
        self,
        X,
        y,
        *,
        loss,
        fit_intercept,
        estimator,
        penalty,
        max_iter,
        tol,
        random_state,
        history,
    ):
        self.X = X
        self.y = y
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.estimator = estimator
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_samples, self.n_features = self.X.shape
        if self.fit_intercept:
            self.n_weights = self.n_features + 1
        else:
            self.n_weights = self.n_features

        self._matrix_type = matrix_type(X)
        if self._matrix_type in {"csc", "csr"}:
            self._X_data = X.data
            self._X_indices = X.indices
            self._X_indptr = X.indptr
        else:
            self._X_data = None
            self._X_indices = None
            self._X_indptr = None

        self.learning_rate_ = None
        self.history = history

    def objective_factory(self):

        value_loss = self.loss.value_batch_factory()
        value_penalty = self.penalty.value_factory()
        y = self.y
        if self.fit_intercept:

            @jit(**_jit_kwargs)
            def objective(weights, inner_products):
                obj = value_loss(y, inner_products)
                obj += value_penalty(weights[1:])
                return obj

            return objective
        else:

            @jit(**_jit_kwargs)
            def objective(weights, inner_products):
                obj = value_loss(y, inner_products)
                obj += value_penalty(weights)
                return obj

            return objective

    @abstractmethod
    def cycle_factory(self):
        pass

    # @abstractmethod
    # def _get_state(self):
    #     pass

    @abstractmethod
    def _compute_learning_rate(self):
        pass

    def solve(self, w0=None, dummy_first_step=False):
        X = self.X
        fit_intercept = self.fit_intercept
        inner_products = np.empty(self.n_samples, dtype=np_float)
        coordinates = np.arange(self.n_weights, dtype=np.intp)
        weights = np.empty(self.n_weights, dtype=np_float)
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

        # Compute the learning rates
        self._compute_learning_rate()

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
            cycle(coordinates, weights, inner_products, state_estimator)
            if w0 is not None:
                weights[:] = w0
            else:
                weights.fill(0.0)
            decision_function(weights, inner_products)

        history.update(weights)

        for n_iter in range(1, max_iter + 1):
            max_abs_delta, max_abs_weight = cycle(
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
