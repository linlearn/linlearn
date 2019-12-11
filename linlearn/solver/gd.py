import numpy as np
from numba import njit
from .history import History


@njit
def gd(model, w, max_epochs, step):
    callback = History(True)
    obj = model.loss_batch(w)
    callback.update(obj)
    g = np.empty(w.shape)
    for epoch in range(max_epochs):
        model.grad_batch(w, out=g)
        w[:] = w[:] - step * g
        obj = model.loss_batch(w)
        callback.update(obj)
    return w
