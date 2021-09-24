# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from numba import jit, vectorize, prange

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, np_float


# Options passed to the @jit decorator within this module
jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


# Options passed to the @vectorize decorator within this module
vectorize_kwargs = {
    "nopython": NOPYTHON,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


################################################################
# Generic estimator
################################################################


class Estimator(object):
    """Base abstract estimator class for internal use only.

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

    Attributes
    ----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_weights : int
        This is `n_features` if `fit_intercept=False` and `n_features` otherwise.
    """

    def __init__(self, X, y, loss, fit_intercept):
        self.X = X
        self.y = y
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_samples, self.n_features = X.shape
        self.n_weights = self.n_features + int(self.fit_intercept)

    def get_state(self):
        pass

    def partial_deriv_factory(self):
        pass

    def grad_factory(self):
        pass


################################################################
# Empirical risk minimizer (ERM)
################################################################


"""
`StateERM` is a place-holder for the ERM estimator containing:

    gradient: numpy.ndarray
        A numpy array of shape (n_weights,) containing gradients computed by the
        `grad` function returned by the `grad_factory` factory function.
"""
StateERM = namedtuple("StateERM", ["gradient"])


class ERM(Estimator):
    """Empirical risk minimization estimator. This is the standard statistical
    learning approach, that corresponds to gradients equal to the average of each
    individual sample loss. Using this estimator should match the results of standard
    linear methods from other libraries.

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

    Attributes
    ----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_weights : int
        This is `n_features` if `fit_intercept=False` and `n_features` otherwise.

    """

    def __init__(self, X, y, loss, fit_intercept):
        Estimator.__init__(self, X, y, loss, fit_intercept)

    def get_state(self):
        """Returns the state of the ERM estimator, which is a place-holder used for
        computations.

        Returns
        -------
        output : StateERM
            State of the ERM estimator
        """
        return StateERM(gradient=np.empty(self.n_weights, dtype=np_float))

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
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

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

                state : StateERM
                    The state of the ERM estimator (not used here, but this allows
                    all estimators to have the same prototypes for `partial_deriv`).

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
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

                state : StateERM
                    The state of the ERM estimator (not used here, but this allows
                    all estimators to have the same prototypes for `partial_deriv`).

                Returns
                -------
                output : float
                    The value of the partial derivative
                """
                deriv_sum = 0.0
                for i in range(y.shape[0]):
                    deriv_sum += deriv_loss(y[i], inner_products[i]) * X[i, j]
                return deriv_sum / n_samples

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
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples

        if self.fit_intercept:

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

                state : StateERM
                    The state of the ERM estimator, which contains a place-holder for
                    the returned gradient.

                Returns
                -------
                output : numpy.array
                    A numpy array of shape (n_weights,) containing the gradient.
                """
                gradient = state.gradient
                gradient.fill(0.0)
                for i in range(n_samples):
                    deriv = deriv_loss(y[i], inner_products[i])
                    gradient[0] += deriv
                    gradient[1:] += deriv * X[i]
                gradient /= n_samples

            return grad
        else:

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

                state : StateERM
                    The state of the ERM estimator, which contains a place-holder for
                    the returned gradient.

                Returns
                -------
                output : numpy.array
                    A numpy array of shape (n_weights,) containing the gradient.
                """
                gradient = state.gradient
                gradient.fill(0.0)
                for i in range(n_samples):
                    gradient += deriv_loss(y[i], inner_products[i]) * X[i]
                gradient /= n_samples

            return grad


################################################################
# Median of means estimator (MOM)
################################################################

"""
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


################################################################
# Trimmed means estimator (TMEAN)
################################################################


# TODO: add docstrings for the next estimators

StateTMean = namedtuple(
    "StateTMean", ["deriv_samples", "deriv_samples_outer_prods", "gradient"]
)


class TMean(Estimator):
    """Trimmed-mean estimator"""

    def __init__(self, X, y, loss, fit_intercept, percentage):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.percentage = percentage
        # Number of samples excluded from both tails (left and right)
        self.n_excluded_tails = int(self.n_samples * percentage / 2)

    def get_state(self):
        return StateTMean(
            deriv_samples=np.empty(self.n_samples, dtype=np_float),
            deriv_samples_outer_prods=np.empty(self.n_samples, dtype=np_float),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
        )

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

                # TODO: Hand-made mean ?
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

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        n_excluded_tails = self.n_excluded_tails

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)

                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])

                deriv_samples_outer_prods[:] = deriv_samples
                deriv_samples_outer_prods.sort()

                gradient[0] = np.mean(
                    deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                )
                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    deriv_samples_outer_prods.sort()
                    gradient[j + 1] = np.mean(
                        deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                    )

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)

                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])

                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    deriv_samples_outer_prods.sort()
                    gradient[j] = np.mean(
                        deriv_samples_outer_prods[n_excluded_tails:-n_excluded_tails]
                    )

            return grad


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


from scipy.optimize import brentq


def standard_catoni_estimator(x, eps=0.001):
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]
    s = estimate_sigma(x)
    res = brentq(lambda u: s * catoni((x - u) / s).mean(), np.min(x), np.max(x))
    return res


################################################################
# Holland Catoni (Holland et al.)
################################################################

StateHollandCatoni = namedtuple(
    "StateHollandCatoni", ["deriv_samples", "deriv_samples_outer_prods", "gradient"]
)


class HollandCatoni(Estimator):
    def __init__(self, X, y, loss, fit_intercept, eps=0.001):
        Estimator.__init__(self, X, y, loss, fit_intercept)
        self.eps = eps

    def get_state(self):
        return StateHollandCatoni(
            deriv_samples=np.empty(self.n_samples, dtype=np_float),
            deriv_samples_outer_prods=np.empty(self.n_samples, dtype=np_float),
            gradient=np.empty(
                self.X.shape[1] + int(self.fit_intercept), dtype=np_float
            ),
        )

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

    def grad_factory(self):
        X = self.X
        y = self.y
        loss = self.loss
        deriv_loss = loss.deriv_factory()
        n_samples = self.n_samples
        eps = self.eps

        if self.fit_intercept:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                gradient[0] = holland_catoni_estimator(deriv_samples, eps)
                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    gradient[j + 1] = holland_catoni_estimator(
                        deriv_samples_outer_prods, eps
                    )

            return grad
        else:

            @jit(**jit_kwargs)
            def grad(inner_products, state):
                deriv_samples = state.deriv_samples
                deriv_samples_outer_prods = state.deriv_samples_outer_prods
                gradient = state.gradient

                gradient.fill(0.0)
                for i in range(n_samples):
                    deriv_samples[i] = deriv_loss(y[i], inner_products[i])
                for j in range(X.shape[1]):
                    deriv_samples_outer_prods[:] = deriv_samples * X[:, j]
                    gradient[j] = holland_catoni_estimator(
                        deriv_samples_outer_prods, eps
                    )

            return grad


################################################################
# Lecue et al. (Implicit)
################################################################

StateImplicit = namedtuple(
    "StateImplicit", ["block_means", "sample_indices", "gradient"]
)


class Implicit(Estimator):
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
        return StateImplicit(
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


################################################################
# Prasad et al. (GMOM)
################################################################

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


# C5 = 0.01
# C = lambda p: C5
# print(
#     "WARNING : importing implementation of outlier robust gradient by (Prasad et al.) with arbitrary constant C(p)=%.2f"
#     % C5
# )

#
#
# def SSI(samples, subset_cardinality):
#     """original name of this function is smallest_subset_interval"""
#     if subset_cardinality < 2:
#         raise ValueError("subset_cardinality must be at least 2")
#     sorted_array = np.sort(samples)
#     differences = (
#         sorted_array[subset_cardinality - 1 :] - sorted_array[: -subset_cardinality + 1]
#     )
#     argmin = np.argmin(differences)
#     return sorted_array[argmin : argmin + subset_cardinality]
#
#
# def alg2(X, eps, delta=0.001):
#     # from Prasad et al. 2018
#     X_tilde = alg4(X, eps, delta)
#
#     n, p = X_tilde.shape
#
#     if p == 1:
#         return np.mean(X_tilde)
#
#     S = np.cov(X.T)
#     _, V = np.linalg.eigh(S)
#     PW = V[:, : p // 2] @ V[:, : p // 2].T
#
#     est1 = np.mean(X_tilde @ PW, axis=0, keepdims=True)
#
#     QV = V[:, p // 2 :]
#     est2 = alg2(X_tilde @ QV, eps, delta)
#     est2 = QV.dot(est2.T)
#     est2 = est2.reshape((1, p))
#     est = est1 + est2
#
#     return est
#
#
# def alg4(X, eps, delta=0.001):
#     # from Prasad et al. 2018
#     n, p = X.shape
#     if p == 1:
#         X_tilde = SSI(
#             X.flatten(),
#             max(
#                 2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))
#             ),
#         )
#         return X_tilde[:, np.newaxis]
#
#     a = np.array([alg2(X[:, i : i + 1], eps, delta / p) for i in range(p)])
#     dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
#     asort = np.argsort(dists)
#     X_tilde = X[
#         asort[
#             : ceil(
#                 n
#                 * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n))
#                 * (1 - eps)
#             )
#         ],
#         :,
#     ]
#     return X_tilde
#
#


@jit(**jit_kwargs, parallel=True)
def row_squared_norm_dense(model):
    n_samples, n_features = model.X.shape
    if model.fit_intercept:
        norms_squared = np.ones(n_samples, dtype=model.X.dtype)
    else:
        norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
    for i in prange(n_samples):
        for j in range(n_features):
            norms_squared[i] += model.X[i, j] * model.X[i, j]
    return norms_squared


def row_squared_norm(model):
    # TODO: for C and F order with aliasing
    return row_squared_norm_dense(model.no_python)


@jit(**jit_kwargs, parallel=True)
def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    if fit_intercept:
        norms_squared = np.zeros(n_features + 1, dtype=X.dtype)
        # First squared norm is n_samples
        norms_squared[0] = n_samples
        for j in prange(1, n_features + 1):
            for i in range(n_samples):
                norms_squared[j] += X[i, j - 1] * X[i, j - 1]
    else:
        norms_squared = np.zeros(n_features, dtype=X.dtype)
        for j in prange(n_features):
            for i in range(n_samples):
                norms_squared[j] += X[i, j] * X[i, j]
    return norms_squared


def col_squared_norm(model):
    # TODO: for C and F order with aliasing
    return col_squared_norm_dense(model.no_python)


#
# @njit
# def grad_batch(model, w, out):
#     out.fill(0)
#     if model.fit_intercept:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[1:] += c * model.X[i]
#             out[0] += c
#     else:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[:] += c * model.X[i]
#     return out
#
#
# @njit(parallel=True)
# def row_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.ones(n_samples, dtype=model.X.dtype)
#     else:
#         norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
#     for i in prange(n_samples):
#         for j in range(n_features):
#             norms_squared[i] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def row_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return row_squared_norm_dense(model.no_python)
#
#
# @njit(parallel=True)
# def col_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.zeros(n_features + 1, dtype=model.X.dtype)
#         # First squared norm is n_samples
#         norms_squared[0] = n_samples
#         for j in prange(1, n_features + 1):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j - 1] * model.X[i, j - 1]
#     else:
#         norms_squared = np.zeros(n_features, dtype=model.X.dtype)
#         for j in prange(n_features):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def col_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return col_squared_norm_dense(model.no_python)
