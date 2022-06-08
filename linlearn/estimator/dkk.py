# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause


from collections import namedtuple
import numpy as np
from numba import jit, objmode
from ._base import Estimator, jit_kwargs
from .._utils import np_float
from sklearn.utils import check_array

from scipy.linalg import eigh


@jit(**jit_kwargs)
def find_t(g, w, eps):
    p1, p2 = int((len(g)-1)/max(1, 2*(1 - 2*eps))), (len(g)-1)
    while p2 - p1 > 1:
        mid = (p1 + p2) // 2
        t = np.partition(g, mid)[mid]
        sm = 0.0
        for i in range(len(g)):
            if g[i] >= t:
                sm += w[i]
        if sm >= eps:
            p1 = mid
        else:
            p2 = mid
    return t

# #@jit(**jit_kwargs)
# def find_t2(g, w, eps):
#     p1, p2 = int((len(g)-1)/max(1, 2*(1 - 2*eps))), (len(g)-1)
#     mid = (p1 + p2) // 2
#     argpart = np.argpartition(g, mid)
#     t = g[argpart[mid]]
#     sm = 0.0
#     for i in range(mid, len(g)):
#         sm += w[argpart[i]]
#     if sm >= eps:
#         indices = argpart[mid:]
#     else:
#         indices = argpart[:mid]
#     vals = g[indices]
#     weight_vals = w[indices]
#     n_vals = len(vals)
#     while n_vals > 1:
#         mid = n_vals // 2
#         argpart = np.argpartition(vals, mid)
#         t = vals[argpart[mid]]
#         if sm >= eps:
#             for i in range(mid):
#                 sm -= weight_vals[argpart[i]]
#         else:
#             for i in range(mid+1, n_vals):
#                 sm += weight_vals[argpart[i]]
#
#         if sm >= eps:
#             indices = argpart[mid:]
#         else:
#             indices = argpart[:mid]
#
#         vals = g[indices]
#         weight_vals = w[indices]
#         n_vals = len(vals)
#
#     return t

@jit(**jit_kwargs)
def projected_partition(A, p, r, ind, B):
    A[r], A[ind] = A[ind], A[r]
    B[r], B[ind] = B[ind], B[r]

    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            if A[j] < x:
                A[j], A[i] = A[i], A[j]
                B[j], B[i] = B[i], B[j]

    A[i+1], A[r] = A[r], A[i+1]
    B[i+1], B[r] = B[r], B[i+1]
    return i + 1

@jit(**jit_kwargs)
def projected_findKth_QS(A, n, k, B):
    N = n
    AA = A
    BB = B
    K = k

    while N > 1:

        kk1 = projected_partition(AA, 0, N-1, N//2, BB)+1
        if kk1 == K:
            break
        elif K < kk1:
            # AA = AA[:kk1-1]
            N = kk1-1
        else:
            AA = AA[kk1:]
            BB = BB[kk1:]
            N = N - kk1
            K = K - kk1


@jit(**jit_kwargs)
def find_t3(g, w, eps):
    #p1, p2 = int((len(g)-1)/max(1, 2*(1 - 2*eps))), (len(g)-1)
    mid = len(g)//2#(p1 + p2) // 2
    projected_findKth_QS(g, len(g), mid+1, w)
    sm = 0.0
    for i in range(mid, len(g)):
        sm += w[i]
    if sm > eps:
        vals = g[mid:]
        weight_vals = w[mid:]
    else:
        vals = g[:mid]
        weight_vals = w[:mid]
    n_vals = len(vals)

    while n_vals > 1:
        mid = n_vals // 2
        projected_findKth_QS(vals, n_vals, mid+1, weight_vals)

        if sm > eps:
            for i in range(mid):
                sm -= weight_vals[i]
        else:
            for i in range(mid, n_vals):
                sm += weight_vals[i]

        if sm > eps:
            vals = vals[mid:]
            weight_vals = weight_vals[mid:]

        else:
            vals = vals[:mid]
            weight_vals = weight_vals[:mid]

        n_vals = len(vals)

    return vals[0]

@jit(**jit_kwargs)
def dkk(vecs, eps):
    n, d = vecs.shape
    w = np.empty(len(vecs))
    ph1 = np.empty(vecs.shape)
    Sigma = np.empty((d, d))
    w.fill(1.0 / len(vecs))
    sum_w = 1.0
    # print("called dkk")
    while sum_w > 1 - 2 * eps:
        mu = np.dot(w, vecs) / sum_w
        # print(mu)
        # print("sum_w = ", sum_w)
        for i in range(n):
            for j in range(d):
                ph1[i, j] = vecs[i, j] - mu[j]
        # ph1 = vecs - mu[np.newaxis, :]

        Sigma.fill(0.0)
        for j1 in range(d):
            for j2 in range(j1, d):
                Sigma[j1, j2] = 0.0
                for i in range(n):
                    Sigma[j1, j2] += w[i] * ph1[i, j1] * ph1[i, j2]
                Sigma[j1, j2] /= sum_w
                Sigma[j2, j1] = Sigma[j1, j2]
        if np.linalg.norm(Sigma) < 1e-3:
            return mu

        #Sigma = ph1.T @ (w[:, np.newaxis] * ph1)
        # TODO : Figure out how to compute only first eigenvector in Numba
        with objmode(eig='float64[:]'):  # annotate return type
            # this region is executed by object-mode.
            _, eig = eigh(Sigma, subset_by_index=[d - 1, d - 1])
            eig = eig.flatten()
            #np.ascontiguousarray(eigvec)
        # eigvals, eigvecs = np.linalg.eigh(Sigma)
        # eigvec = eigvecs[:, np.argmax(eigvals)]

        g = np.square(ph1 @ eig).reshape(n)

        # g = np.zeros(n)#
        # for i in range(n):
        #     for j in range(d):
        #         g[i] += ph1[i, j] * eig[j]
        #     g[i] = g[i] * g[i]
        f = g.copy()

        # ________________________________
        # asg = np.argsort(g)#[::-1]
        # sm = 0.0
        # ind = n-1
        # while sm < eps:
        #     sm += w[asg[ind]]
        #     ind -= 1
        # t = g[asg[ind + 1]]
        # ________________________________
        t = find_t3(g, w, eps)
        # print(np.min(g), np.max(g))
        # print("t = ", t)
        # print(np.linalg.norm(Sigma))

        # f[f < t] = 0
        m = 0.0
        #print("pass 2")
        for i in range(n):
            if f[i] < t:
                f[i] = 0.0
            elif f[i] > m and w[i] > 0:
                m = f[i]
        # print("m = ", m)
        #print("pass 3")
        w = np.multiply(w, 1 - f / m)
        sum_w = np.sum(w)
    mu = np.dot(w, vecs) / sum_w

    return mu





StateDKK = namedtuple(
    "StateDKK",
    [
        "deriv_samples",
        "deriv_samples_outer_prods",
        "gradient",
        "loss_derivative",
    ],
)


class DKK(Estimator):
    def __init__(self, X, y, loss, n_classes, fit_intercept, eps):
        super().__init__(X, y, loss, n_classes, fit_intercept)
        self.eps = eps

    def get_state(self):
        return StateDKK(
            deriv_samples=np.empty(
                (self.n_samples, self.n_classes), dtype=np_float, order="F"
            ),
            deriv_samples_outer_prods=np.empty(
                (self.n_samples, self.n_classes), dtype=np_float, order="F"
            ),
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
        )

    def partial_deriv_factory(self):
        raise ValueError(
            "dkk estimator does not support CGD, use mom/tmean/ch estimator instead"
        )

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_features = self.n_features
        n_classes = self.n_classes
        eps = self.eps

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv_samples[i])

                    for k in range(n_classes):
                        deriv_samples_outer_prods[i, k] = deriv_samples[i, k]

                gradient[0, :] = dkk(deriv_samples_outer_prods, eps)

                for j in range(n_features):
                    for k in range(n_classes):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                deriv_samples[i, k] * X[i, j]
                            )
                    gradient[j + 1, :] = dkk(deriv_samples_outer_prods, eps)
                    # with objmode():  # annotate return type
                    #     # this region is executed by object-mode.
                    #     check_array(gradient[j + 1:j+2, :])

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                for i in range(n_samples):
                    deriv_loss(y[i], inner_products[i], deriv_samples[i])

                for j in range(n_features):

                    for k in range(n_classes):
                        for i in range(n_samples):
                            deriv_samples_outer_prods[i, k] = (
                                    deriv_samples[i, k] * X[i, j]
                            )

                    gradient[j, :] = dkk(deriv_samples_outer_prods, eps)
                    # with objmode():  # annotate return type
                    #     # this region is executed by object-mode.
                    #     check_array(gradient[j:j+1, :])

                return 0
            return grad
