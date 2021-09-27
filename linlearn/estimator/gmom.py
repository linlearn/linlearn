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


StateGMOM = namedtuple(
    "StateGMOM", ["block_means", "sample_indices", "grads_sum_block", "gradient"]
)


class GMOM(Estimator):
    def __init__(self, X, y, loss, fit_intercept, n_samples_in_block):
        super().__init__(X, y, loss, fit_intercept)
        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block
        if self.last_block_size > 0:
            self.n_blocks += 1

    def get_state(self):
        return StateGMOM(
            block_means=np.empty(
                (self.n_blocks, self.X.shape[1] + int(self.fit_intercept)),
                dtype=np_float,
            ),
            sample_indices=np.arange(self.n_samples, dtype=np.uintp),
            grads_sum_block=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
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
        n_blocks = self.n_blocks
        last_block_size = self.last_block_size

        # @njit
        @jit(**jit_kwargs)
        def gmom_njit(xs, tol=1e-7):
            # from Vardi and Zhang 2000
            n_elem, n_dim = xs.shape
            y = np.zeros(n_dim)
            dists = np.zeros(n_elem)
            inv_dists = np.zeros(n_elem)

            xsy = np.zeros_like(xs)
            for i in range(n_elem):
                y += xs[i]
            y /= xs.shape[0]
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
                grads_sum_block.fill(0.0)
                # Block counter
                counter = 0
                deriv = 0.0
                for i, idx in enumerate(sample_indices):
                    deriv = deriv_loss(y[idx], inner_products[idx])
                    grads_sum_block[0] += deriv
                    grads_sum_block[1:] += deriv * X[idx]

                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = grads_sum_block / n_samples_in_block
                        counter += 1
                        grads_sum_block.fill(0.0)
                if last_block_size != 0:
                    block_means[counter] = grads_sum_block / last_block_size

                gradient[:] = gmom_njit(block_means)

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
                grads_sum_block.fill(0.0)
                # Block counter
                counter = 0
                for i, idx in enumerate(sample_indices):
                    grads_sum_block += deriv_loss(y[idx], inner_products[idx]) * X[idx]

                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = grads_sum_block / n_samples_in_block
                        counter += 1
                        grads_sum_block.fill(0.0)
                if last_block_size != 0:
                    block_means[counter] = grads_sum_block / last_block_size

                gradient[:] = gmom_njit(block_means)

            return grad
