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
from .._utils import np_float, fast_median


StateMOM = namedtuple(
    "StateMOM",
    [
        "block_means",
        "sample_indices",
        "gradient",
        "loss_derivative",
        "partial_derivative",
    ],
)


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

    def __init__(self, X, y, loss, n_classes, fit_intercept, n_samples_in_block):
        super().__init__(X, y, loss, n_classes, fit_intercept)
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
            block_means=np.empty(
                (self.n_blocks, self.n_classes), dtype=np_float, order="F"
            ),
            sample_indices=np.arange(self.n_samples, dtype=np.intp),
            gradient=np.empty(
                (self.n_features + int(self.fit_intercept), self.n_classes),
                dtype=np_float,
            ),
            loss_derivative=np.empty(self.n_classes, dtype=np_float),
            partial_derivative=np.empty(self.n_classes, dtype=np_float),
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
        n_classes = self.n_classes
        n_samples_in_block = self.n_samples_in_block
        last_block_size = self.last_block_size
        n_blocks = self.n_blocks

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

                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                deriv = state.loss_derivative
                # Cumulative sum in the block
                derivatives_sum_block = state.partial_derivative
                for k in range(n_classes):
                    derivatives_sum_block[k] = 0.0
                # Block counter
                n_block = 0
                if j == 0:
                    for i, idx in enumerate(sample_indices):
                        deriv_loss(y[idx], inner_products[idx], deriv)
                        for k in range(n_classes):
                            derivatives_sum_block[k] += deriv[k]
                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                            for k in range(n_classes):
                                block_means[n_block, k] = (
                                    derivatives_sum_block[k] / n_samples_in_block
                                )
                                derivatives_sum_block[k] = 0.0
                            n_block += 1
                else:
                    for i, idx in enumerate(sample_indices):
                        deriv_loss(y[idx], inner_products[idx], deriv)
                        Xij = X[idx, j - 1]
                        for k in range(n_classes):
                            derivatives_sum_block[k] += deriv[k] * Xij

                        if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                            for k in range(n_classes):
                                block_means[n_block, k] = (
                                    derivatives_sum_block[k] / n_samples_in_block
                                )
                                derivatives_sum_block[k] = 0.0
                            n_block += 1

                if last_block_size != 0:
                    for k in range(n_classes):
                        block_means[n_block, k] = (
                            derivatives_sum_block[k] / last_block_size
                        )

                for k in range(n_classes):
                    derivatives_sum_block[k] = fast_median(block_means[:, k], n_blocks)
                    # derivatives_sum_block[k] = np.median(block_means[:, k])
                # return np.median(block_means)

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
                # for i in range(n_samples):
                #     sample_indices[i] = i

                np.random.shuffle(sample_indices)
                deriv = state.loss_derivative
                # Cumulative sum in the block
                derivatives_sum_block = state.partial_derivative
                for k in range(n_classes):
                    derivatives_sum_block[k] = 0.0
                # Block counter
                n_block = 0
                for i, idx in enumerate(sample_indices):
                    deriv_loss(y[idx], inner_products[idx], deriv)
                    Xij = X[idx, j]
                    for k in range(n_classes):
                        derivatives_sum_block[k] += deriv[k] * Xij

                    if (i != 0) and ((i + 1) % n_samples_in_block == 0):
                        for k in range(n_classes):
                            block_means[n_block, k] = (
                                derivatives_sum_block[k] / n_samples_in_block
                            )
                            derivatives_sum_block[k] = 0.0
                        n_block += 1

                if last_block_size != 0:
                    for k in range(n_classes):
                        block_means[n_block, k] = (
                            derivatives_sum_block[k] / last_block_size
                        )

                for k in range(n_classes):
                    derivatives_sum_block[k] = fast_median(block_means[:, k], n_blocks)
                    # derivatives_sum_block[k] = np.median(block_means[:, k])
                # return np.median(block_means)

            return partial_deriv

    def grad_factory(self):
        """Gradient factory. This returns a jit-compiled function allowing to
        compute the gradient of the considered goodness-of-fit.

        Returns
        -------
        output : function
            A jit-compiled function allowing to compute gradients.
        """
        n_features = self.n_features
        n_classes = self.n_classes
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
            partial_derivative = state.partial_derivative

            for j in range(n_features + int(fit_intercept)):
                partial_deriv(j, inner_products, state)
                for k in range(n_classes):
                    gradient[j, k] = partial_derivative[k]
            return 0

        return grad
