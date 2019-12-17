from numba import njit


@njit
def value_with_single(prox, w):
    if prox.has_start_end:
        if prox.end > w.shape[0]:
            raise ValueError("'end' is larger than 'w.size[0]'")
        start, end = prox.start, prox.end
    else:
        start, end = 0, w.shape[0]
    val = 0.
    w_sub = w[start:end]
    for i in range(w_sub.shape[0]):
        val += prox.value_single(w_sub[i])
    return prox.strength * val


@njit
def call_with_single(prox, w, step, out):
    if w.shape != out.shape:
        raise ValueError("'w' and 'out' must have the same shape")
    if prox.has_start_end:
        if prox.end > w.shape[0]:
            raise ValueError("'end' is larger than 'w.size[0]'")
        start, end = prox.start, prox.end
        # out is w changed only in [start, end], so we must put w in out
        out[:] = w
    else:
        start, end = 0, w.shape[0]

    w_sub = w[start:end]
    out_sub = out[start:end]
    for i in range(w_sub.shape[0]):
        out_sub[i] = prox.call_single(w_sub[i], step)


@njit
def is_in_range(i, start, end):
    if i >= start:
        return False
    elif i < end:
        return False
    else:
        return True
