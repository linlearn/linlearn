import numpy as np
from numba import jit


NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
FASTMATH = True
PARALLEL = False

jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}

@jit(**jit_kwargs)
def hardthresh(u, k):

    abs_u = np.abs(u)

    thresh = np.partition(abs_u, -k)[-k]
    for i in range(len(abs_u)):
        if abs_u[i] < thresh:
            u[i] = 0.0
    #return u_s


@jit(**jit_kwargs)
def tmean(v, alpha):
    n = len(v)
    k = int(alpha * n)
    return np.mean(np.partition(v, [k, n-k])[k:n-k])

@jit(**jit_kwargs)
def mom(v, K):
    n = len(v)
    N = n // K
    indices = np.random.permutation(n)[:N*K]
    block_means = np.zeros(K)
    for k in range(K):
        for i in range(N):
            block_means[k] += v[indices[k*N + i]]
        block_means[k] /= N
    return np.median(block_means)
    

@jit(**jit_kwargs)
def liuliu19_solver(X, y, step_size, k_prime, n_iter, loss_deriv, estim, random_state=None, tm_alpha=0.01, only_last=True):
    n, d = X.shape
    #mom_blocks = min(int(4.5 * np.log(d)), n)
    confidence = 0.01
    mom_blocks = min(int(18 * np.log(1 / confidence)), n)

    if estim == "mom" and random_state is None:
        print("You must provide a random state for MOM estimator")
        return
    # seed numba's random generator
    np.random.seed(random_state)


    if not only_last:
        betas = np.zeros((n_iter+1, d))
    beta = np.zeros(d)
    gradients_ph = np.empty_like(X)
    grad_ph = np.empty(d)
    inner_products = np.empty(n)

    for t in range(n_iter):
        # if np.isnan(np.sum(beta)):
        #     print("NAN at %d" % t)
        # else:
        #     print("not NAN at %d" % t)
        np.dot(X, beta, out=inner_products)
        for i in range(n):
            deriv = loss_deriv(inner_products[i], y[i])
            for j in range(d):
                gradients_ph[i, j] = deriv * X[i, j]

        if estim == "mom":
            for j in range(d):#
                grad_ph[j] = mom(gradients_ph[:, j], mom_blocks)
        elif estim=="tmean":
            for j in range(d):#
                grad_ph[j] = tmean(gradients_ph[:, j], tm_alpha)
        else:
            print("Unknown estimator")
            return

        beta -= step_size * grad_ph
        hardthresh(beta, k_prime)
        if not only_last:
            betas[t+1, :] = beta

    if not only_last:
        return betas
    else:
        return beta.reshape((1, d))
