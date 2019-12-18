import numpy as np
from numba import jit, prange, njit, jitclass
from .history import History


# TODO: good default for tol when using duality gap
# TODO: step=float or {'best', 'auto'}
# TODO: random_state same thing as in scikit
# TODO:

@njit
def svrg_epoch(model, prox, w, w_old, gradient_memory, step, indices):
    # This implementation assumes dense data and a separable prox
    X = model.X
    n_samples, n_features = X.shape
    # TODO: indices.shape[0] == model.X.shape[0] == model.y.shape[0] ???
    for idx in range(n_samples):
        i = indices[idx]
        c_new = model.grad_sample_coef(i, w)
        c_old = model.grad_sample_coef(i, w_old)
        if model.fit_intercept:
            # Intercept is never penalized
            w[0] = w[0] - step * ((c_new - c_old) + gradient_memory[0])
            for j in range(1, n_features + 1):
                w_j = w[j] - step * ((c_new - c_old) * X[i, j-1]
                                     + gradient_memory[j])
                w[j] = prox.call_single(w_j, step)
        else:
            for j in range(w.shape[0]):
                w_j = w[j] - step * ((c_new - c_old) * X[i, j]
                                     + gradient_memory[j])
                w[j] = prox.call_single(w_j, step)
    return w


class SVRG(object):

    def __init__(self, step='best', rand_type='unif', tol=1e-10,
                 max_iter=10, verbose=True, print_every=1, random_state=-1):
        self.step = step
        self.rand_type = rand_type
        self.tol = tol
        self.max_iter = max_iter
        self.print_every = print_every
        self.random_state = random_state
        self.verbose = verbose
        self.history = History('SVRG', self.max_iter, self.verbose)

    def set(self, model, prox):
        self.model = model
        self.prox = prox
        return self

    # def loss_batch(self, w):
    #     return loss_batch(self.features, self.labels, self.loss, w)
    #
    # def grad_batch(self, w, out):
    #     grad_batch(self.features, self.labels, self.loss, w, out=out)

    def solve(self, w):
        # TODO: check that set as been called
        # TODO: save gradient_memory, so that we can continue training later

        gradient_memory = np.empty(w.shape)
        w_old = np.empty(w.shape)

        model = self.model.no_python
        prox = self.prox.no_python

        n_samples = model.n_samples
        obj = model.loss_batch(w) + prox.value(w)

        history = self.history
        # features = self.features
        # labels = self.labels
        # loss = self.loss
        step = self.step

        history.update(epoch=0, obj=obj, step=step, tol=0., update_bar=False)

        for epoch in range(1, self.max_iter + 1):
            # At the beginning of each epoch we compute the full gradient
            # TODO: veriifer que c'est bien le cas... qu'il ne faut pas le
            #  faire a la fin de l'epoque
            w_old[:] = w

            # Compute the full gradient
            model.grad_batch(w, gradient_memory)
            # grad_batch(self.features, self.labels, self.loss, w, out=gradient_memory)

            # TODO: en fonction de rand_type...
            indices = np.random.randint(n_samples, size=n_samples)

            # Launch the epoch pass
            svrg_epoch(model, prox, w, w_old, gradient_memory, step, indices)

            obj = model.loss_batch(w) + prox.value(w)
            history.update(epoch=epoch, obj=obj, step=step, tol=0.,
                           update_bar=True)

        history.close_bar()
        return w

    # @property
    # def step(self):
    #     return self._step
    #
    # @step.setter
    # def step(self, val):
    #     self._step = val
    #
    #     if val is None:
    #         val = 0.
    #     if self._solver is not None:
    #         self._solver.set_step(val)
    #
    # @property
    # def rand_type(self):
    #     if self._rand_type == unif:
    #         return "unif"
    #     if self._rand_type == perm:
    #         return "perm"
    #     else:
    #         raise ValueError("No known ``rand_type``")
    #
    # @rand_type.setter
    # def rand_type(self, val):
    #     if val not in ["unif", "perm"]:
    #         raise ValueError("``rand_type`` can be 'unif' or " "'perm'")
    #     else:
    #         if val == "unif":
    #             enum_val = unif
    #         if val == "perm":
    #             enum_val = perm
    #         self._set("_rand_type", enum_val)
    #
    # def _set_rand_max(self, model):
    #     model_rand_max = model._rand_max
    #     self._set("_rand_max", model_rand_max)
    #


# step = 1 / model.lip_max()
# callback_svrg = inspector(model, n_iter=n_iter)
# w_svrg = svrg(model, w0, idx_samples, n_iter=model.n_samples * n_iter,
#               step=step, callback=callback_svrg)

# # @jit(nopython=True, nogil=True)
# def sgd(loss_epoch, grad_sample_coef, w, max_epochs, n_samples, step, callback):
#     # w = w0.copy()
#     for epoch in range(max_epochs):
#         callback(w)
#         for iter in range(n_samples):
#             i = np.random.randint(n_samples)
#             xi = X[i]
#             yi = y[i]
#             c = grad_sample_coef(xi, yi, w, intercept)
#             w[0] -= c
#             w[1:] -= step * c * xi
#     return w
