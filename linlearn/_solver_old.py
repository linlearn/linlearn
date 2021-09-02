# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from math import fabs
from collections import namedtuple
import numpy as np
from numpy.random import permutation, shuffle
from numba import njit, jit, void, boolean, uintp
from numba.experimental import jitclass
import matplotlib.pyplot as plt

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, nb_float, np_float
from ._loss_old import decision_function_factory
# from ._penalty import value

# from .strategy import grad_coordinate_erm, decision_function


# from .history import History
# from linlearn.model.utils import inner_prods


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


################################################################
# Coordinate gradient descent CGD
################################################################

#
# spec_state_cgd = [
#     ("n_samples", uintp),
#     ("n_features", uintp),
#     ("n_weights", uintp),
#     ("inner_products", FLOAT[::1]),
#     ("fit_intercept", boolean),
#     ("coordinates", uintp[::1]),
#     ("weights", FLOAT[::1]),
# ]
#
#
# @jitclass(spec_state_cgd)
# class StateCGD(object):
#     def __init__(self, n_samples, n_features, fit_intercept=True):
#         self.n_samples = n_samples
#         self.n_features = n_features
#         self.fit_intercept = fit_intercept
#         if self.fit_intercept:
#             self.n_weights = self.n_features + 1
#         else:
#             self.n_weights = self.n_features
#         self.inner_products = np.zeros((self.n_samples,), dtype=NP_FLOAT)
#         self.coordinates = np.empty(self.n_weights, dtype=np.uintp)
#         self.weights = np.empty(self.n_weights, dtype=NP_FLOAT)
#

# TODO: good default for tol when using duality gap
# TODO: step=float or {'best', 'auto'}
# TODO: random_state same thing as in scikit


OptimizationResult = namedtuple(
    "OptimizationResult", ["n_iter", "tol", "success", "w", "message"]
)


class CGD(object):
    def __init__(
        self, X, y, loss, fit_intercept, estimator, penalty, max_iter, tol,
            random_state, steps
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

        self.steps = steps

    def objective_factory(self):

        value_batch = self.loss.value_batch_factory()

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def objective(weights, inner_products):
                return value_batch(weights, inner_products)

            return objective
        else:

            @jit(**jit_kwargs)
            def objective(weights, inner_products):
                return value_batch(weights, inner_products)

            return objective
            # obj = value_batch(value_loss, state_loss, y, inner_products)
            # if fit_intercept:
            #     obj += value(value_one_penalty, state_penalty, w[1:])
            # else:
            #     obj += value(value_one_penalty, state_penalty, w)
            # return obj

    def cycle_factory(self):

        X = self.X
        fit_intercept = self.fit_intercept
        n_samples = self.estimator.n_samples
        n_weights = self.n_weights
        partial_deriv_estimator = self.estimator.partial_deriv_factory()
        steps = self.steps

        if fit_intercept:

            @njit # @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                # weights = state_cgd.weights
                # inner_products = state_cgd.inner_products
                for idx in range(n_weights):
                    coordinates[idx] = idx

                shuffle(coordinates)

                for j in coordinates:
                    partial_deriv_j = partial_deriv_estimator(j, inner_products)
                    if j == 0:
                        # It's the intercept, so we don't penalize
                        w_j_new = weights[j] - steps[j] * partial_deriv_j
                    else:
                        # It's not the intercept
                        w_j_new = weights[j] - steps[j] * partial_deriv_j
                        # w_j_new = apply_one_penalty(state_penalty, w_j_new, steps[j])

                    # Update the inner products
                    delta_j = w_j_new - weights[j]

                    # Update the maximum update change
                    abs_delta_j = fabs(delta_j)
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j

                    # Update the maximum weight
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
            # There is no intercept, so the code changes slightly
            @njit # @jit(**jit_kwargs)
            def cycle(coordinates, weights, inner_products):
                max_abs_delta = 0.0
                max_abs_weight = 0.0
                for idx in range(n_weights):
                    coordinates[idx] = idx
                shuffle(coordinates)

                for j in coordinates:
                    partial_deriv_j = partial_deriv_estimator(j, inner_products)
                    w_j_new = weights[j] - steps[j] * partial_deriv_j
                    # w_j_new = apply_one_penalty(state_penalty, w_j_new, steps[j])

                    # Update the inner products
                    delta_j = w_j_new - weights[j]
                    # Update the maximum update change
                    abs_delta_j = fabs(delta_j)
                    if abs_delta_j > max_abs_delta:
                        max_abs_delta = abs_delta_j
                    # Update the maximum weight
                    abs_w_j_new = fabs(w_j_new)
                    if abs_w_j_new > max_abs_weight:
                        max_abs_weight = abs_w_j_new

                    for i in range(n_samples):
                        inner_products[i] += delta_j * X[i, j]

                    weights[j] = w_j_new
                return max_abs_delta, max_abs_weight

            return cycle

    def solve(self, w0=None):

        # n_samples, n_features = X.shape
        # Initialize the state of the solver and the initial model weights
        # state_cgd = StateCGD(n_samples, n_features, fit_intercept)

        X = self.X
        fit_intercept = self.fit_intercept
        inner_products = np.empty(self.n_samples, dtype=np_float)
        weights = np.empty(self.n_weights, dtype=np_float)
        coordinates = np.empty(self.n_weights, dtype=np.uintp)

        max_iter = self.max_iter

        history = History("CGD", max_iter, True)

        # ,weights = state_cgd.weights

        if w0 is not None:
            weights[:] = w0
        else:
            weights.fill(0.0)

        # Computation of the initial inner products
        # inner_products = np.empty(n_samples, dtype=X.dtype)
        # Compute the inner products X . w + b
        # TODO: decision function should be given by the strategy

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
        objective = self.objective_factory()

        obj = objective(weights, inner_products)
        print("obj:", obj)

        # TODO: First value for tolerance is 1.0 or NaN
        history.update(epoch=0, obj=obj, tol=1.0, update_bar=True)

        for n_iter in range(1, max_iter + 1):
            # Sample a permutation of the coordinates
            # coordinates = permutation(w_size)
            # Launch the coordinates cycle

            # print("--------------------------------")
            # print("cycle:", cycle)

            # Launch a cycle of coordinate descent. Note that this modifies the weights
            # vectors (through state_cgd.weights, passed to this function)

            max_abs_delta, max_abs_weight = cycle(coordinates, weights, inner_products)
            #
            # max_abs_delta, max_abs_weight = cgd_cycle(
            #     deriv_loss,
            #     state_loss,
            #     partial_deriv_estimator,
            #     state_estimator,
            #     apply_one_penalty,
            #     state_penalty,
            #     X,
            #     y,
            #     steps,
            #     state_cgd,
            # )

            # print("max_abs_delta:", max_abs_delta, "max_abs_weight:", max_abs_weight)
            # def cgd_cycle(
            #     deriv_loss,
            #     state_loss,
            #     deriv_estimator,
            #     state_estimator,
            #     apply_penalty,
            #     state_penalty,
            #     X,
            #     y,
            #     steps,
            #     state_cgd,
            # ):

            # Compute the new value of objective
            obj = objective(weights, inner_products)
            print("obj:", obj)

            # Did we reached the required tolerance within the max_iter number of cycles ?
            # if (
            #     max_abs_weight == 0.0 or max_abs_delta / max_abs_weight < tol
            # ) and cycle <= max_iter:
            #     success = True
            # else:
            #     success = False

            if max_abs_weight == 0.0:
                current_tol = 0.0
            else:
                current_tol = max_abs_delta / max_abs_weight

            # print(
            #     "max_abs_delta: ",
            #     max_abs_delta,
            #     ", max_abs_weight: ",
            #     max_abs_weight,
            #     ", current_tol: ",
            #     current_tol,
            #     ", tol: ",
            #     tol,
            # )

            # TODO: tester tous les cas "max_abs_weight == 0.0" etc..
            history.update(epoch=n_iter, obj=obj, tol=current_tol, update_bar=True)

            # Decide if we stop or not
        # #     if current_tol < tol:
        # #         history.close_bar()
        # #         return OptimizationResult(
        # #             w=weights, n_iter=cycle, success=True, tol=tol, message=None
        # #         )
        # #
        # # history.close_bar()
        # #
        # # return OptimizationResult(
        # #     w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
        # )


# @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
# def cgd_cycle(
#     deriv_loss,
#     state_loss,
#     deriv_estimator,
#     state_estimator,
#     apply_one_penalty,
#     state_penalty,
#     X,
#     y,
#     steps,
#     state_cgd,
# ):
#     """Performs one cycle over all coordinates of coordinate gradient descent
#     """
#     max_abs_delta = 0.0
#     max_abs_weight = 0.0
#
#     n_samples = state_cgd.n_samples
#     n_weights = state_cgd.n_weights
#     fit_intercept = state_cgd.fit_intercept
#     coordinates = state_cgd.coordinates
#     weights = state_cgd.weights
#     inner_products = state_cgd.inner_products
#
#     for idx in range(n_weights):
#         coordinates[idx] = idx
#
#     shuffle(coordinates)
#
#     for j in coordinates:
#         # partial_deriv_j = 1e-2
#         partial_deriv_j = deriv_estimator(
#             deriv_loss, state_loss, j, X, y, state_cgd, state_estimator
#         )
#         if fit_intercept and j == 0:
#             # It's the intercept, so we don't penalize
#             w_j_new = weights[j] - steps[j] * partial_deriv_j
#         else:
#             # It's not the intercept
#             w_j_new = weights[j] - steps[j] * partial_deriv_j
#             # w_j_new = apply_one_penalty(state_penalty, w_j_new, steps[j])
#
#         # Update the inner products
#         delta_j = w_j_new - weights[j]
#
#         # Update the maximum update change
#         abs_delta_j = fabs(delta_j)
#         if abs_delta_j > max_abs_delta:
#             max_abs_delta = abs_delta_j
#
#         # Update the maximum weight
#         abs_w_j_new = fabs(w_j_new)
#         if abs_w_j_new > max_abs_weight:
#             max_abs_weight = abs_w_j_new
#
#         if fit_intercept:
#             if j == 0:
#                 for i in range(n_samples):
#                     inner_products[i] += delta_j
#             else:
#                 for i in range(n_samples):
#                     inner_products[i] += delta_j * X[i, j - 1]
#         else:
#             for i in range(n_samples):
#                 inner_products[i] += delta_j * X[i, j]
#         weights[j] = w_j_new
#
#     return max_abs_delta, max_abs_weight
#

# Attributes
# xndarray
# The solution of the optimization.
# successbool
# Whether or not the optimizer exited successfully.
# statusint
# Termination status of the optimizer. Its value depends on the underlying solver. Refer to message for details.
# messagestr
# Description of the cause of the termination.
# fun, jac, hess: ndarray
# Values of objective function, its Jacobian and its Hessian (if available). The Hessians may be approximations, see the documentation of the function in question.
# hess_invobject
# Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers. The type of this attribute may be either np.ndarray or scipy.sparse.linalg.LinearOperator.
# nfev, njev, nhevint
# Number of evaluations of the objective functions and of its Jacobian and Hessian.
# nitint
# Number of iterations performed by the optimizer.
# maxcvfloat
# The maximum constraint violation.


def coordinate_gradient_descent(
    loss,
    estimator,
    penalty,
    w0,
    X,
    y,
    fit_intercept,
    steps,
    max_iter,
    tol,
    history,
    random_state=None,
):
    n_samples, n_features = X.shape
    # Initialize the state of the solver and the initial model weights
    state_cgd = StateCGD(n_samples, n_features, fit_intercept)
    inner_products = state_cgd.inner_products
    weights = state_cgd.weights
    weights[:] = w0

    # Computation of the initial inner products
    # inner_products = np.empty(n_samples, dtype=X.dtype)
    # Compute the inner products X . w + b
    # TODO: decision function should be given by the strategy
    decision_function(X, fit_intercept, weights, out=inner_products)

    @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
    def numba_seed_numpy(random_state):
        np.random.seed(random_state)

    if random_state is not None:
        numba_seed_numpy(random_state)

    value_loss = loss.value
    deriv_loss = loss.deriv
    state_loss = loss.state

    state_estimator = estimator.state
    partial_deriv_estimator = estimator.partial_deriv

    state_penalty = penalty.state
    apply_one_penalty = penalty.apply_one
    value_one_penalty = penalty.value_one

    n_weights = state_cgd.n_weights

    # Objective function
    # TODO: can be njitted right ?

    @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
    def objective(state_loss, state_penalty, w):
        obj = value_batch(value_loss, state_loss, y, inner_products)
        if fit_intercept:
            obj += value(value_one_penalty, state_penalty, w[1:])
        else:
            obj += value(value_one_penalty, state_penalty, w)
        return obj

    # grad_coordinate = strategy.grad_coordinate
    # partial_deriv_mom(loss_deriv, j, X, y, state_solver, state_mom)

    # Value of the objective at initialization
    obj = objective(state_loss, state_penalty, weights)

    # TODO: First value for tolerance is 1.0 or NaN
    history.update(epoch=0, obj=obj, tol=1.0, update_bar=True)

    for cycle in range(1, max_iter + 1):
        # Sample a permutation of the coordinates
        # coordinates = permutation(w_size)
        # Launch the coordinates cycle

        # print("--------------------------------")
        # print("cycle:", cycle)

        # Launch a cycle of coordinate descent. Note that this modifies the weights
        # vectors (through state_cgd.weights, passed to this function)
        max_abs_delta, max_abs_weight = cgd_cycle(
            deriv_loss,
            state_loss,
            partial_deriv_estimator,
            state_estimator,
            apply_one_penalty,
            state_penalty,
            X,
            y,
            steps,
            state_cgd,
        )

        # print("max_abs_delta:", max_abs_delta, "max_abs_weight:", max_abs_weight)
        # def cgd_cycle(
        #     deriv_loss,
        #     state_loss,
        #     deriv_estimator,
        #     state_estimator,
        #     apply_penalty,
        #     state_penalty,
        #     X,
        #     y,
        #     steps,
        #     state_cgd,
        # ):

        # Compute the new value of objective
        obj = objective(state_loss, state_penalty, weights)

        # Did we reached the required tolerance within the max_iter number of cycles ?
        # if (
        #     max_abs_weight == 0.0 or max_abs_delta / max_abs_weight < tol
        # ) and cycle <= max_iter:
        #     success = True
        # else:
        #     success = False

        if max_abs_weight == 0.0:
            current_tol = 0.0
        else:
            current_tol = max_abs_delta / max_abs_weight

        # print(
        #     "max_abs_delta: ",
        #     max_abs_delta,
        #     ", max_abs_weight: ",
        #     max_abs_weight,
        #     ", current_tol: ",
        #     current_tol,
        #     ", tol: ",
        #     tol,
        # )

        # TODO: tester tous les cas "max_abs_weight == 0.0" etc..
        history.update(epoch=cycle, obj=obj, tol=current_tol, update_bar=True)

        # Decide if we stop or not
        if current_tol < tol:
            history.close_bar()
            return OptimizationResult(
                w=weights, n_iter=cycle, success=True, tol=tol, message=None
            )

    history.close_bar()

    return OptimizationResult(
        w=weights, n_iter=max_iter + 1, success=False, tol=tol, message=None
    )

    # TODO: stopping criterion max(weigth difference) / max(weight) + duality gap
    # TODO: and then use
    # if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
    #     # the biggest coordinate update of this iteration was smaller than
    #     # the tolerance: check the duality gap as ultimate stopping
    #     # criterion
    #
    #     # XtA = np.dot(X.T, R) - l2_reg * W.T
    #     for ii in range(n_features):
    #         for jj in range(n_tasks):
    #             XtA[ii, jj] = _dot(
    #                 n_samples, X_ptr + ii * n_samples, 1, & R[0, jj], 1
    #             ) - l2_reg * W[jj, ii]
    #
    #     # dual_norm_XtA = np.max(np.sqrt(np.sum(XtA ** 2, axis=1)))
    #     dual_norm_XtA = 0.0
    #     for ii in range(n_features):
    #         # np.sqrt(np.sum(XtA ** 2, axis=1))
    #         XtA_axis1norm = _nrm2(n_tasks, & XtA[ii, 0], 1)
    #         if XtA_axis1norm > dual_norm_XtA:
    #             dual_norm_XtA = XtA_axis1norm
    #
    #     # TODO: use squared L2 norm directly
    #     # R_norm = linalg.norm(R, ord='fro')
    #     # w_norm = linalg.norm(W, ord='fro')
    #     R_norm = _nrm2(n_samples * n_tasks, & R[0, 0], 1)
    #     w_norm = _nrm2(n_features * n_tasks, & W[0, 0], 1)
    #     if (dual_norm_XtA > l1_reg):
    #         const = l1_reg / dual_norm_XtA
    #         A_norm = R_norm * const
    #         gap = 0.5 * (R_norm ** 2 + A_norm ** 2)
    #     else:
    #         const = 1.0
    #         gap = R_norm ** 2
    #
    #     # ry_sum = np.sum(R * y)
    #     ry_sum = _dot(n_samples * n_tasks, & R[0, 0], 1, & Y[0, 0], 1)
    #
    #     # l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
    #     l21_norm = 0.0
    #     for ii in range(n_features):
    #         l21_norm += _nrm2(n_tasks, & W[0, ii], 1)
    #
    #         gap += l1_reg * l21_norm - const * ry_sum + \
    #                0.5 * l2_reg * (1 + const ** 2) * (w_norm ** 2)
    #
    #         if gap < tol:
    #             # return if we reached desired tolerance
    #             break
    #     else:
    #         # for/else, runs if for doesn't end with a `break`
    #         with gil:
    #             warnings.warn("Objective did not converge. You might want to "
    #                           "increase the number of iterations. Duality "
    #                           "gap: {}, tolerance: {}".format(gap, tol),
    #                           ConvergenceWarning)

    # TODO: return more than just this... return an object that include for things than
    #  this


#
# def coordinate_gradient_descent_factory():
#     pass
#
#
# solvers_factory = {"cgd": coordinate_gradient_descent_factory}


# Dans SAG critere d'arret :
# if status == -1:
#     break
# # check if the stopping criteria is reached
# max_change = 0.0
# max_weight = 0.0
# for idx in range(n_features * n_classes):
#     max_weight = fmax
#     {{name}}(max_weight, fabs(weights[idx]))
#     max_change = fmax
#     {{name}}(max_change,
#              fabs(weights[idx] -
#                   previous_weights[idx]))
#     previous_weights[idx] = weights[idx]
# if ((max_weight != 0 and max_change / max_weight <= tol)
#         or max_weight == 0 and max_change == 0):
#     if verbose:
#         end_time = time(NULL)
#         with gil:
#             print("convergence after %d epochs took %d seconds" %
#                   (n_iter + 1, end_time - start_time))
#     break
# elif verbose:
#     printf('Epoch %d, change: %.8f\n', n_iter + 1,
#            max_change / max_weight)


# @njit
# def gd(model, w, max_epochs, step):
#     callback = History(True)
#     obj = model.loss_batch(w)
#     callback.update(obj)
#     g = np.empty(w.shape)
#     for epoch in range(max_epochs):
#         model.grad_batch(w, out=g)
#         w[:] = w[:] - step * g
#         obj = model.loss_batch(w)
#         callback.update(obj)
#     return w

#
# # TODO: good default for tol when using duality gap
# # TODO: step=float or {'best', 'auto'}
# # TODO: random_state same thing as in scikit
# # TODO:
#
#
# @njit
# def svrg_epoch(model, prox, w, w_old, gradient_memory, step, indices):
#     # This implementation assumes dense data and a separable prox_old
#     X = model.X
#     n_samples, n_features = X.shape
#     # TODO: indices.shape[0] == model.X.shape[0] == model.y.shape[0] ???
#     for idx in range(n_samples):
#         i = indices[idx]
#         c_new = model.grad_sample_coef(i, w)
#         c_old = model.grad_sample_coef(i, w_old)
#         if model.fit_intercept:
#             # Intercept is never penalized
#             w[0] = w[0] - step * ((c_new - c_old) + gradient_memory[0])
#             for j in range(1, n_features + 1):
#                 w_j = w[j] - step * ((c_new - c_old) * X[i, j - 1] + gradient_memory[j])
#                 w[j] = prox.call_single(w_j, step)
#         else:
#             for j in range(w.shape[0]):
#                 w_j = w[j] - step * ((c_new - c_old) * X[i, j] + gradient_memory[j])
#                 w[j] = prox.call_single(w_j, step)
#     return w
#
#
# class SVRG(object):
#     def __init__(
#         self,
#         step="best",
#         rand_type="unif",
#         tol=1e-10,
#         max_iter=10,
#         verbose=True,
#         print_every=1,
#         random_state=-1,
#     ):
#         self.step = step
#         self.rand_type = rand_type
#         self.tol = tol
#         self.max_iter = max_iter
#         self.print_every = print_every
#         self.random_state = random_state
#         self.verbose = verbose
#         self.history = History("SVRG", self.max_iter, self.verbose)
#
#     def set(self, model, prox):
#         self.model = model
#         self.prox = prox
#         return self
#
#     # def loss_batch(self, w):
#     #     return loss_batch(self.features, self.labels, self.loss, w)
#     #
#     # def grad_batch(self, w, out):
#     #     grad_batch(self.features, self.labels, self.loss, w, out=out)
#
#     def solve(self, w):
#         # TODO: check that set as been called
#         # TODO: save gradient_memory, so that we can continue training later
#
#         gradient_memory = np.empty(w.shape)
#         w_old = np.empty(w.shape)
#
#         model = self.model.no_python
#         prox = self.prox.no_python
#
#         n_samples = model.n_samples
#         obj = model.loss_batch(w) + prox.value(w)
#
#         history = self.history
#         # features = self.features
#         # labels = self.labels
#         # loss = self.loss
#         step = self.step
#
#         history.update(epoch=0, obj=obj, step=step, tol=0.0, update_bar=False)
#
#         for epoch in range(1, self.max_iter + 1):
#             # At the beginning of each epoch we compute the full gradient
#             # TODO: veriifer que c'est bien le cas... qu'il ne faut pas le
#             #  faire a la fin de l'epoque
#             w_old[:] = w
#
#             # Compute the full gradient
#             model.grad_batch(w, gradient_memory)
#             # grad_batch(self.features, self.labels, self.loss, w, out=gradient_memory)
#
#             # TODO: en fonction de rand_type...
#             indices = np.random.randint(n_samples, size=n_samples)
#
#             # Launch the epoch pass
#             svrg_epoch(model, prox, w, w_old, gradient_memory, step, indices)
#
#             obj = model.loss_batch(w) + prox.value(w)
#             history.update(epoch=epoch, obj=obj, step=step, tol=0.0, update_bar=True)
#
#         history.close_bar()
#         return w

