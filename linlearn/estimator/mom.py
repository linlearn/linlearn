# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module implement the ``MOM`` class for the median-of-means robust estimator.

`StateMOM` is a place-holder for the MOM estimator containing:

    block_means : numpy.ndarray
        A numpy array of shape (n_blocks,) containing the mean partial derivatives in
        MOM's blocks.

    sample_indices : numpy.ndarray
        A numpy array of shape (n_samples,) containing the shuffled indices
        corresponding to the block samples.

    gradient : numpy.ndarray
        A numpy array of shape (n_weights,) containing gradients computed by the
        `grad` function returned by the `grad_factory` factory function.
"""

from collections import namedtuple
import numpy as np
from numba import jit
from ._base import Estimator, jit_kwargs
from .._utils import np_float


StateMOM = namedtuple("StateMOM", ["block_means", "sample_indices", "gradient"])


class MOM(Estimator):
    """Median of means estimator. This estimator is robust with respect to outliers
    and heavy-tails. It computes means in blocks, and returns the median value of the
    blocks. Blocks are obtained through a shuffle of the sample indices. This estimator
    mainly allows to compute fast and robust estimations of the partial derivatives
    of the goodness-of-fit.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X.

    loss : Loss
        A loss class for which gradients will be estimated by the estimator.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    n_samples_in_block : int
        Number of samples used in the blocks. Note that the last block can be smaller
        than that.

    Attributes
    ----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_weights : int
        This is `n_features` if `fit_intercept=False` and `n_features` otherwise.

    n_blocks : int
        Number of blocks used.

    last_block_size : int
        Size of the last block
    """

    def __init__(self, X, y, loss, fit_intercept, n_samples_in_block):
        super().__init__(X, y, loss, fit_intercept)
        self.n_samples_in_block = n_samples_in_block
        self.n_blocks = self.n_samples // n_samples_in_block
        self.last_block_size = self.n_samples % n_samples_in_block
        if self.last_block_size > 0:
            self.n_blocks += 1

    def get_state(self):
        """Returns the state of the MOM estimator, which is a place-holder used for
        computations.

        Returns
        -------
        output : StateMOM
            State of the MOM estimator
        """

        return StateMOM(
            block_means=np.empty(self.n_blocks, dtype=np_float),
            sample_indices=np.empty(self.n_samples, dtype=np.intp),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
        )

    def partial_deriv_factory(self):
        """Partial derivatives factory. This returns a jit-compiled function allowing to
        compute partial derivatives of the considered goodness-of-fit.

        Returns
        -------
        output : function
            A jit-compiled function allowing to compute partial derivatives.
        """
        X = self.X
        y = self.y
        deriv_loss = self.loss.deriv_factory()
        n_samples = self.n_samples
        n_samples_in_block = self.n_samples_in_block
        last_block_size = self.last_block_size

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def partial_deriv(j, inner_products, state):
                """Computes the partial derivative of the goodness-of-fit with
                respect to coordinate `j`, given the value of the `inner_products` and
                `state`.

                Parameters
                ----------
                j : int
                    Partial derivative is with respect to this coordinate

                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateMOM
                    The state of the MOM estimator.

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
                sample_indices = state.sample_indices
                block_means = state.block_means

                for i in range(n_samples):
                    sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                if j == 0:
                    for i, idx in enumerate(sample_indices):
                        derivatives_sum_block += deriv_loss(y[idx], inner_products[idx])
                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                            block_means[n_block] = (
                                derivatives_sum_block / n_samples_in_block
                            )
                            n_block += 1
                            derivatives_sum_block = 0.0
                else:
                    for i, idx in enumerate(sample_indices):
                        derivatives_sum_block += (
                            deriv_loss(y[idx], inner_products[idx]) * X[idx, j - 1]
                        )
                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
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
                """Computes the partial derivative of the goodness-of-fit with
                respect to coordinate `j`, given the value of the `inner_products` and
                `state`.

                Parameters
                ----------
                j : int
                    Partial derivative is with respect to this coordinate

                inner_products : numpy.array
                    A numpy array of shape (n_samples,), containing the inner
                    products (decision function) X.dot(w) + b where w is the weights
                    and b the (optional) intercept.

                state : StateMOM
                    The state of the MOM estimator.

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
                sample_indices = state.sample_indices
                block_means = state.block_means
                for i in range(n_samples):
                    sample_indices[i] = i

                np.random.shuffle(sample_indices)
                # Cumulative sum in the block
                derivatives_sum_block = 0.0
                # Block counter
                n_block = 0
                for i, idx in enumerate(sample_indices):
                    derivatives_sum_block += (
                        deriv_loss(y[idx], inner_products[idx]) * X[idx, j]
                    )
                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        block_means[n_block] = (
                            derivatives_sum_block / n_samples_in_block
                        )
                        n_block += 1
                        derivatives_sum_block = 0.0

                if last_block_size != 0:
                    block_means[n_block] = derivatives_sum_block / last_block_size

                return np.median(block_means)

            return partial_deriv

    def grad_factory(self):
        """Gradient factory. This returns a jit-compiled function allowing to
        compute the gradient of the considered goodness-of-fit.

        Returns
        -------
        output : function
            A jit-compiled function allowing to compute gradients.
        """
        X = self.X
        fit_intercept = self.fit_intercept
        partial_deriv = self.partial_deriv_factory()

        @jit(**jit_kwargs)
        def grad(inner_products, state):
            """Computes the gradient of the goodness-of-fit, given the value of the
             `inner_products` and `state`.

            Parameters
            ----------
            inner_products : numpy.array
                A numpy array of shape (n_samples,), containing the inner
                products (decision function) X.dot(w) + b where w is the weights
                and b the (optional) intercept.

            state : StateMOM
                The state of the MOM estimator.

            Returns
            -------
            output : numpy.array
                A numpy array of shape (n_weights,) containing the gradient.
            """
            gradient = state.gradient

            for j in range(X.shape[1] + int(fit_intercept)):
                gradient[j] = partial_deriv(j, inner_products, state)

        return grad
