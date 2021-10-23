# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains the base Solver class
"""

from abc import ABC, abstractmethod
import numpy as np
from warnings import warn
from math import fabs
from numpy.random import permutation
from numba import jit
import matplotlib.pyplot as plt
import math
from collections import namedtuple

# from .history import History
# from linlearn.model.utils import inner_prods

# from .strategy import grad_coordinate_erm, decision_function, strategy_classes
# from ._estimator import decision_function_
from .._loss import decision_function_factory
from .._utils import (
    NOPYTHON,
    NOGIL,
    BOUNDSCHECK,
    FASTMATH,
    nb_float,
    np_float,
    rand_choice_nb,
)


# TODO: good default for tol when using duality gap
# TODO: step=float or {'best', 'auto'}
# TODO: random_state same thing as in scikit

OptimizationResult = namedtuple(
    "OptimizationResult", ["n_iter", "tol", "success", "w", "message"]
)


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


class Solver(ABC):
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
        history,
    ):
        self.X = X
        self.y = y
        self.loss = loss
        self.n_classes = n_classes
        self.fit_intercept = fit_intercept
        self.estimator = estimator
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_samples, self.n_features = self.X.shape
        if self.fit_intercept:
            self.n_weights = (self.n_features + 1) * self.n_classes
            self.weights_shape = (self.n_features + 1, self.n_classes)
        else:
            self.n_weights = self.n_features * self.n_classes
            self.weights_shape = (self.n_features, self.n_classes)

        self.history = history
        self.history.allocate_record(self.weights_shape)
        self.history.allocate_record(1)

    @abstractmethod
    def cycle_factory(self):
        pass

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
        if tol > 0:
            warn("Maximum iteration number reached, solver may have not converged")
        return OptimizationResult(
            w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        )
