# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the ``GMOM`` class for the geometric median-of-means robust
estimator.

``StateGMOM`` is a place-holder for the GMOM estimator containing:


    gradient: numpy.ndarray
        A numpy array of shape (n_weights,) containing gradients computed by the
        `grad` function returned by the `grad_factory` factory function.

    TODO: fill the missing things in StateCH
"""

from collections import namedtuple
import numpy as np
from numba import jit
from ._base import Estimator, jit_kwargs
from .._utils import np_float


@jit(**jit_kwargs)
def gmom_njit(xs, tol=1e-10):
    # from Vardi and Zhang 2000
    n_elem, n_dim = xs.shape
    # TODO : avoid the memory allocations in this function
    y = np.zeros(n_dim)
    dists = np.zeros(n_elem)
    inv_dists = np.zeros(n_elem)

    xsy = np.zeros_like(xs)
    for i in range(n_elem):
        y += xs[i]
    y /= n_elem
    eps = 1e-10
    delta = 1
    niter = 0
    while delta > tol:
        xsy[:] = xs - y
        dists.fill(0.0)
        for i in range(n_dim):
            dists += xsy[:, i] ** 2  # np.linalg.norm(xsy, axis=1)
        dists[:] = np.sqrt(dists)
        inv_dists[:] = 1 / dists
        mask = dists < eps
        inv_dists[mask] = 0
        nb_too_close = (mask).sum()
        ry = np.sqrt(
            np.sum(np.dot(inv_dists, xsy) ** 2)
        )  # np.linalg.norm(np.dot(inv_dists, xsy))
        cst = nb_too_close / ry
        y_new = (
            max(0, 1 - cst) * np.dot(inv_dists, xs) / np.sum(inv_dists)
            + min(1, cst) * y
        )
        delta = np.sqrt(np.sum((y - y_new) ** 2))  # np.linalg.norm(y - y_new)
        y = y_new
        niter += 1
    # print(niter)
    return y


StateGMOM = namedtuple(
    "StateGMOM",
    [
        "block_means",
        "sample_indices",
        "grads_sum_block",
        "gradient",
        "loss_derivative",
        "partial_derivative",
    ],
)


class GMOM(Estimator):
    def __init__(self, X, y, loss, n_classes, fit_intercept, n_samples_in_block):
        super().__init__(X, y, loss, n_classes, fit_intercept)
        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block
        if self.last_block_size > 0:
            self.n_blocks += 1

    def get_state(self):
        return StateGMOM(
            block_means=np.empty(
                (
                    self.n_blocks,
                    self.n_features + int(self.fit_intercept),
                    self.n_classes,
                ),
                dtype=np_float,
            ),
            sample_indices=np.arange(self.n_samples, dtype=np.uintp),
            grads_sum_block=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
            partial_derivative=np.empty(self.n_classes, dtype=np_float),
        )

    def partial_deriv_factory(self):
        raise ValueError(
            "gmom estimator does not support CGD, use mom estimator instead"
        )

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples_in_block = self.n_samples_in_block
        n_classes = self.n_classes
        n_features = self.n_features
        n_blocks = self.n_blocks
        last_block_size = self.last_block_size

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                gradient = state.gradient
                # Cumulative sum in the block
                grads_sum_block = state.grads_sum_block
                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                for j in range(n_features + 1):
                    for k in range(n_classes):
                        grads_sum_block[j, k] = 0.0
                # Block counter
                counter = 0
                deriv = state.loss_derivative
                for i, idx in enumerate(sample_indices):
                    deriv_loss(y[idx], inner_products[idx], deriv)
                    for k in range(n_classes):
                        grads_sum_block[0, k] += deriv[k]
                        for j in range(n_features):
                            grads_sum_block[j + 1, k] += (
                                X[idx, j] * deriv[k]
                            )  # np.outer(X[idx], deriv)

                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        for j in range(n_features + 1):
                            for k in range(n_classes):
                                block_means[counter, j, k] = (
                                    grads_sum_block[j, k] / n_samples_in_block
                                )
                                grads_sum_block[j, k] = 0.0
                        counter += 1

                if last_block_size != 0:
                    for j in range(n_features + 1):
                        for k in range(n_classes):
                            block_means[counter, j, k] = (
                                grads_sum_block[j, k] / last_block_size
                            )

                # TODO : possible optimizations in the next line by rewriting gmom_njit with out parameter
                #  and preallocated place holders ...
                gradient[:] = gmom_njit(block_means.reshape((n_blocks, -1))).reshape(
                    block_means.shape[1:]
                )

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                gradient = state.gradient
                # Cumulative sum in the block
                grads_sum_block = state.grads_sum_block
                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                for j in range(n_features):
                    for k in range(n_classes):
                        grads_sum_block[j, k] = 0.0
                # Block counter
                counter = 0
                deriv = state.loss_derivative
                for i, idx in enumerate(sample_indices):
                    deriv_loss(y[idx], inner_products[idx], deriv)
                    for j in range(n_features):
                        for k in range(n_classes):
                            grads_sum_block[j, k] += X[idx, j] * deriv[k]

                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        for j in range(n_features + 1):
                            for k in range(n_classes):
                                block_means[counter, j, k] = (
                                    grads_sum_block[j, k] / n_samples_in_block
                                )
                                grads_sum_block[j, k] = 0.0
                        counter += 1

                if last_block_size != 0:
                    for j in range(n_features):
                        for k in range(n_classes):
                            block_means[counter, j, k] = (
                                grads_sum_block[j, k] / last_block_size
                            )

                # TODO : possible optimizations in the next line by rewriting gmom_njit with out parameter
                #  and preallocated place holders ...
                gradient[:] = gmom_njit(block_means.reshape((n_blocks, -1))).reshape(
                    block_means.shape[1:]
                )

            return grad
