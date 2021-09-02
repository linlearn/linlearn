# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from numba import jit, vectorize

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, np_float


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


vectorize_kwargs = {
    "nopython": NOPYTHON,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


################################################################
# Generic estimator
################################################################


class Estimator(ABC):
    def __init__(self, X, y, loss, fit_intercept):
        self.X = X
        self.y = y
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_samples = y.shape[0]

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def partial_deriv_factory(self):
        pass


################################################################
# Empirical risk minimizer (ERM)
################################################################

StateERM = namedtuple("StateERM", [])


class ERM(Estimator):
    def __init__(self, X, y, loss, fit_intercept):
        Estimator.__init__(self, X, y, loss, fit_intercept)

    # NB: Do not make this method static
    def get_state(self):
        return StateERM()

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_sum = 0.0
                if j == 0:
                    for i in range(n_samples):
                        deriv_sum += deriv_loss(y[i], inner_products[i])
                    return deriv_sum / n_samples
                else:
                    for i in range(n_samples):
                        deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                    return deriv_sum / n_samples

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_sum = 0.0
                for i in range(y.shape[0]):
                    deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j]
                return deriv_sum / n_samples

            return partial_deriv


################################################################
# Median of means estimator (MOM)
################################################################


StateMOM = namedtuple("StateMOM", ["block_means", "sample_indices"])


class MOM(Estimator):
    """MOM (Median-of-Means) estimator.

    """

    def __init__(self, X, y, loss, fit_intercept, n_samples_in_block):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block

        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block

        if self.last_block_size > 0:
            self.n_blocks += 1

    def get_state(self):
        return StateMOM(
            block_means=np.empty(self.n_blocks, dtype=np_float),
            sample_indices=np.arange(0, self.n_samples, dtype=np.intp),
        )

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        deriv_loss = self.loss.deriv_factory()
        n_samples = self.n_samples
        n_samples_in_block = self.n_samples_in_block
        n_blocks = self.n_blocks
        last_block_size = self.last_block_size

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                # We shuffle the sample indices each time we compute a partial
                # derivative to end up with different blocks
                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                if j == 0:
                    for idx, i in enumerate(sample_indices):
                        derivatives_sum_block += deriv_loss(y[i], inner_products[i])
                        if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
                            block_means[n_block] = (
                                derivatives_sum_block / n_samples_in_block
                            )
                            n_block += 1
                            derivatives_sum_block = 0.0

                    if last_block_size != 0:
                        block_means[n_block] = derivatives_sum_block / last_block_size

                else:
                    for idx, i in enumerate(sample_indices):
                        derivatives_sum_block += (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )
                        if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
                            block_means[n_block] = (
                                derivatives_sum_block / n_samples_in_block
                            )
                            n_block += 1
                            derivatives_sum_block = 0.0

                    if last_block_size != 0:
                        block_means[n_block] = derivatives_sum_block / last_block_size

                return np.median(block_means)

            return partial_deriv

        else:
            # Same function without an intercept
            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                sample_indices = state.sample_indices
                block_means = state.block_means
                for i in range(n_samples):
                    sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                for idx, i in enumerate(sample_indices):
                    derivatives_sum_block += (
                        deriv_loss(y[i], inner_products[i]) * X[i, j]
                    )
                    if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
                        block_means[n_block] = (
                            derivatives_sum_block / n_samples_in_block
                        )
                        n_block += 1
                        derivatives_sum_block = 0.0

                if last_block_size != 0:
                    block_means[n_blocks - 1] = derivatives_sum_block / last_block_size
                return np.median(block_means)

            return partial_deriv


################################################################
# Trimmed means estimator (TMEAN)
################################################################


StateTMean = namedtuple("StateTMean", ["deriv_samples"])


class TMean(Estimator):
    """Trimmed-mean estimator"""

    def __init__(self, X, y, loss, fit_intercept, percentage):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.percentage = percentage
        # Number of samples excluded from both tails (left and right)
        self.n_excluded_tails = int(self.n_samples * percentage / 2)

    def get_state(self):
        return StateTMean(deriv_samples=np.empty(self.n_samples, dtype=np_float))

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                if j == 0:
                    for i in range(n_samples):
                        deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                else:
                    for i in range(n_samples):
                        deriv_samples[i] = (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )

                # TODO: Try out different sorting mechanisms, since at some point the
                #  sorting order won't change much...
                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i]) * X[i, j]

                deriv_samples.sort()
                return np.mean(deriv_samples[n_excluded_tails:-n_excluded_tails])

            return partial_deriv


################################################################
# "Catoni-Holland" robust estimator
################################################################


@vectorize(**vectorize_kwargs)
def catoni(x):
    return np.sign(x) * np.log(1 + np.sign(x) * x + x * x / 2)
    # if x > 0 else -np.log(1 - x + x*x/2)


@vectorize(**vectorize_kwargs)
def khi(x):
    return 0.62 - 1 / (1 + x * x)  # np.log(0.5 + x*x)#


@vectorize(**vectorize_kwargs)
def gud(x):
    return 2 * np.arctan(np.exp(x)) - np.pi / 2 if x < 12 else np.pi / 2


@jit(**jit_kwargs)
def estimate_sigma(x, eps=0.001):
    sigma = 1.0
    x_mean = x.mean()
    delta = 1
    khi0 = khi(0.0)
    while delta > eps:
        tmp = sigma * np.sqrt(1 - (khi((x - x_mean) / sigma)).mean() / khi0)
        delta = np.abs(tmp - sigma)
        sigma = tmp
    return sigma


@jit(**jit_kwargs)
def holland_catoni_estimator(x, eps=0.001):
    # if the array is constant, do not try to estimate scale
    # the following condition is supposed to reproduce np.allclose() behavior
    # TODO: clean this in some way
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]

    s = estimate_sigma(x) * np.sqrt(len(x) / np.log(1 / eps))
    m = 0.0
    diff = 1.0
    while diff > eps:
        tmp = m + s * gud((x - m) / s).mean()
        diff = np.abs(tmp - m)
        m = tmp
    return m


StateCatoni = namedtuple("StateCatoni", ["deriv_samples"])


class Catoni(Estimator):
    def __init__(self, X, y, loss, fit_intercept, eps=0.001):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.eps = eps

    def get_state(self):
        return StateCatoni(deriv_samples=np.empty(self.n_samples, dtype=np_float))

    def partial_deriv_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        eps = self.eps

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                if j == 0:
                    for i in range(n_samples):
                        deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                else:
                    for i in range(n_samples):
                        deriv_samples[i] = (
                            deriv_loss(y[i], inner_products[i]) * X[i, j - 1]
                        )

                return holland_catoni_estimator(deriv_samples, eps)

            return partial_deriv
        else:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                deriv_samples = state.deriv_samples
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i]) * X[i, j]

                return holland_catoni_estimator(deriv_samples, eps)

            return partial_deriv
