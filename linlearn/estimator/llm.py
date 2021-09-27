# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the ``LLM`` class for the Lecué - Lerasle - Mathieu robust
estimator.

``StateLLM`` is a place-holder for the LLM estimator containing:

    gradient: numpy.ndarray
        A numpy array of shape (n_weights,) containing gradients computed by the
        `grad` function returned by the `grad_factory` factory function.

    TODO: fill the missing things
"""

from collections import namedtuple
import numpy as np
from numba import jit
from ._base import Estimator, jit_kwargs
from .._utils import np_float

StateLLM = namedtuple("StateLLM", ["block_means", "sample_indices", "gradient"])


class LLM(Estimator):
    def __init__(self, X, y, loss, fit_intercept, n_blocks):
        # assert n_blocks % 2 == 1
        super().__init__(X, y, loss, fit_intercept)
        # n_blocks must be uneven
        self.n_blocks = n_blocks + ((n_blocks + 1) % 2)
        self.n_samples_in_block = self.n_samples // n_blocks
        # no last block size, the remaining samples are just ignored
        # self.last_block_size = self.n_samples % self.n_samples_in_block
        # if self.last_block_size > 0:
        #     self.n_blocks += 1

    def get_state(self):
        return StateLLM(
            block_means=np.empty(self.n_blocks, dtype=np_float),
            sample_indices=np.arange(self.n_samples, dtype=np.uintp),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
        )

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        deriv_loss = self.loss.deriv_factory()
        n_samples_in_block = self.n_samples_in_block
        loss = self.loss
        value_loss = loss.value_factory()
        deriv_loss = loss.deriv_factory()

        # Better implementation of argmedian ??
        @jit(**jit_kwargs)
        def argmedian(x):
            med = np.median(x)
            id = 0
            for a in x:
                if a == med:
                    return id
                id += 1
            raise ValueError("Failed argmedian")

            # return np.argpartition(x, len(x) // 2)[len(x) // 2]

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                objectives_sum_block = 0.0
                # Block counter
                counter = 0
                for i, idx in enumerate(sample_indices):
                    objectives_sum_block += value_loss(y[idx], inner_products[idx])
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = objectives_sum_block / n_samples_in_block
                        counter += 1
                        objectives_sum_block = 0.0

                argmed = argmedian(block_means)

                deriv = 0.0
                if j == 0:
                    for i in sample_indices[
                        argmed * n_samples_in_block : (argmed + 1) * n_samples_in_block
                    ]:
                        deriv += deriv_loss(y[i], inner_products[i])
                else:
                    for i in sample_indices[
                        argmed * n_samples_in_block : (argmed + 1) * n_samples_in_block
                    ]:
                        deriv += deriv_loss(y[i], inner_products[i]) * X[i, j - 1]

                deriv /= n_samples_in_block
                return deriv

            return partial_deriv

        else:
            # Same function without an intercept
            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                objectives_sum_block = 0.0
                # Block counter
                counter = 0
                for i, idx in enumerate(sample_indices):
                    objectives_sum_block += value_loss(y[idx], inner_products[idx])
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = objectives_sum_block / n_samples_in_block
                        counter += 1
                        objectives_sum_block = 0.0

                argmed = argmedian(block_means)

                deriv = 0.0
                for i in sample_indices[
                    argmed * n_samples_in_block : (argmed + 1) * n_samples_in_block
                ]:
                    deriv += deriv_loss(y[i], inner_products[i]) * X[i, j]

                deriv /= n_samples_in_block
                return deriv

            return partial_deriv

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        value_loss = loss.value_factory()
        deriv_loss = loss.deriv_factory()
        n_samples_in_block = self.n_samples_in_block
        n_blocks = self.n_blocks

        # Better implementation of argmedian ??
        @jit(**jit_kwargs)
        def argmedian(x):
            med = np.median(x)
            id = 0
            for a in x:
                if a == med:
                    return id
                id += 1
            raise ValueError("Failed argmedian")

            # return np.argpartition(x, len(x) // 2)[len(x) // 2]

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                gradient = state.gradient
                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                objectives_sum_block = 0.0
                # Block counter
                counter = 0
                for i, idx in enumerate(sample_indices):
                    objectives_sum_block += value_loss(y[idx], inner_products[idx])
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = objectives_sum_block / n_samples_in_block
                        counter += 1
                        objectives_sum_block = 0.0

                argmed = argmedian(block_means)

                gradient.fill(0.0)
                deriv = 0.0
                for i in sample_indices[
                    argmed * n_samples_in_block : (argmed + 1) * n_samples_in_block
                ]:
                    deriv = deriv_loss(y[i], inner_products[i])
                    gradient[0] += deriv
                    gradient[1:] += deriv * X[i]
                gradient /= n_samples_in_block

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                gradient = state.gradient
                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                objectives_sum_block = 0.0
                # Block counter
                counter = 0
                for i, idx in enumerate(sample_indices):
                    objectives_sum_block += value_loss(y[idx], inner_products[idx])
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[counter] = objectives_sum_block / n_samples_in_block
                        counter += 1
                        objectives_sum_block = 0.0

                argmed = argmedian(block_means)

                gradient.fill(0.0)
                for i in sample_indices[
                    argmed * n_samples_in_block : (argmed + 1) * n_samples_in_block
                ]:
                    gradient += deriv_loss(y[i], inner_products[i]) * X[i]
                gradient /= n_samples_in_block

            return grad
