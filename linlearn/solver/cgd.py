import numpy as np
from numpy.random import permutation
from numba import jit, prange, njit, jitclass
from .history import History
from linlearn.model.utils import inner_prods

# TODO: good default for tol when using duality gap
# TODO: step=float or {'best', 'auto'}
# TODO: random_state same thing as in scikit

@njit
def cgd_cycle(model, prox, w, inner_products, steps, coordinates):
    # This implementation assumes dense data and a separable prox
    # TODO: F order C order
    X = model.X
    n_samples, n_features = X.shape
    w_size = w.shape[0]

    for idx in range(w_size):
        j = coordinates[idx]
        grad_j = model.grad_coordinate(j, inner_products)

        if model.fit_intercept and j == 0:
            # It's the intercept, so we don't penalize
            w_j_new = w[j] - steps[j] * grad_j
        else:
            # It's not the intercept
            w_j_new = w[j] - steps[j] * grad_j
            w_j_new = prox.call_single(w_j_new, steps[j])

        # Update the inner products
        delta_j = w_j_new - w[j]
        if model.fit_intercept:
            if j == 0:
                for i in range(n_samples):
                    inner_products[i] += delta_j
            else:
                for i in range(n_samples):
                    inner_products[i] += delta_j * X[i, j - 1]
        else:
            for i in range(n_samples):
                inner_products[i] += delta_j * X[i, j]
        w[j] = w_j_new


class CGD(object):

    def __init__(self, steps, rand_type='perm', tol=1e-10,
                 max_iter=10, verbose=True, print_every=1, random_state=-1):
        # TODO: steps is {'auto'} or np.array
        self.steps = steps
        self.rand_type = rand_type
        self.tol = tol
        self.max_iter = max_iter
        self.print_every = print_every
        # TODO: random_state is unused
        self.random_state = random_state
        self.verbose = verbose
        self.history = History('CGD', self.max_iter, self.verbose)
        self.model = None
        self.prox = None

    def set(self, model, prox):
        # TODO: test that model is an instance of Model and that it has all the
        #  required methods
        # TODO: same thing for prox
        self.model = model
        self.prox = prox
        return self

    def solve(self, w):
        if self.model is None:
            raise ValueError("You must call 'set' before 'solve'")
        model = self.model.no_python
        prox = self.prox.no_python
        n_samples = model.n_samples

        # Computation of the initial inner products
        inner_products = np.empty(model.n_samples, dtype=np.float64)
        inner_prods(model.X, model.fit_intercept, w, inner_products)

        obj = model.loss_batch(w) + prox.value(w)

        history = self.history
        steps = self.steps
        w_size = w.shape[0]

        history.update(epoch=0, obj=obj, tol=0., update_bar=False)
        for cycle in range(1, self.max_iter + 1):
            # Sample a permutation of the coordinates
            coordinates = permutation(w_size)
            # Launch the coordinates cycle
            cgd_cycle(model, prox, w, inner_products, steps, coordinates)
            # Compute new value of objective
            obj = model.loss_batch(w) + prox.value(w)
            history.update(epoch=cycle, obj=obj, tol=0., update_bar=True)

        history.close_bar()
        return w
