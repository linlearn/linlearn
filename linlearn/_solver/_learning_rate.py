"""
This modules contains functions allowing to compute the learning rates used by solvers.

Functions are named with the template

    `<func>_<estimator>_<matrix_type>_factory`

where:

- <func> is the type of learning rate(s) computed. This includes `learning_rates`
 which corresponds to the learning rates used by the `CGD` solver, `learning_rate_best`
 which is used by the `GD` and `learning_rate_sto` which is used by `SVRG` and `SAGA`.
- <estimator> is the estimator used ('erm', 'mom', etc.)
- <matrix_type> is the type of matrix used, among 'f' (F-major matrix), 'c' (C-major
 matrix), 'csc' (sparse CSC matrix) and 'csr' (sparse CSR matrix).

"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause


import numpy as np
from numba import jit, prange

from ._base import _jit_kwargs
from .._estimator import median_of_means, holland_catoni_estimator, ERM, MOM, CH, TMean
from .._utils import matrix_type


# TODO: parallelize computations of learning rates everywhere


def learning_rates_erm_f_factory(X, fit_intercept):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    ERM (empirical risk minimizer) estimator with a F-major matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features. Ideally this ndarray is F-major.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    n_samples, n_features = X.shape
    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            out[0] = 1.0 / lip_const
            for j in range(1, n_features + 1):
                col_j_squared_norm = 0.0
                for i in range(n_samples):
                    col_j_squared_norm += X[i, j - 1] ** 2
                out[j] = n_samples / (lip_const * col_j_squared_norm)

            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            for j in range(n_features):
                col_j_squared_norm = 0.0
                for i in range(n_samples):
                    col_j_squared_norm += X[i, j] ** 2
                out[j] = n_samples / (lip_const * col_j_squared_norm)

            return out

        return learning_rates


def learning_rates_erm_c_factory(X, fit_intercept):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    ERM (empirical risk minimizer) estimator with a C-major matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features. Ideally this ndarray is C-major.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    n_samples, n_features = X.shape
    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            out[0] = 1.0 / lip_const
            for i in range(n_samples):
                for j in range(1, n_features + 1):
                    out[j] += X[i, j - 1] ** 2
            for j in range(1, n_features + 1):
                out[j] = n_samples / (lip_const * out[j])

            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            for i in range(n_samples):
                for j in range(n_features):
                    out[j] += X[i, j] ** 2
            for j in range(n_features):
                out[j] = n_samples / (lip_const * out[j])

            return out

        return learning_rates


def learning_rates_erm_csc_factory(X, fit_intercept):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    ERM (empirical risk minimizer) estimator with a sparse CSC matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : scipy.sparse.csc_matrix of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    n_samples, n_features = X.shape
    X_data = X.data
    X_indptr = X.indptr

    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            out[0] = 1.0 / lip_const
            for j in range(1, n_features + 1):
                col_j_squared_norm = 0.0
                col_start = X_indptr[j - 1]
                col_end = X_indptr[j]
                for idx in range(col_start, col_end):
                    col_j_squared_norm += X_data[idx] ** 2

                out[j] = n_samples / (lip_const * col_j_squared_norm)

            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs,)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            for j in range(n_features):
                col_j_squared_norm = 0.0
                col_start = X_indptr[j]
                col_end = X_indptr[j + 1]
                for idx in range(col_start, col_end):
                    col_j_squared_norm += X_data[idx] ** 2

                out[j] = n_samples / (lip_const * col_j_squared_norm)

            return out

        return learning_rates


def learning_rates_erm_csr_factory(X, fit_intercept):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    ERM (empirical risk minimizer) estimator with a sparse CSC matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : scipy.sparse.csr_matrix of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    n_samples, n_features = X.shape
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            out[0] = 1.0 / lip_const
            for i in range(n_samples):
                row_start = X_indptr[i]
                row_end = X_indptr[i + 1]
                for idx in range(row_start, row_end):
                    j = X_indices[idx] + 1
                    out[j] += X_data[idx] ** 2
            for j in range(1, n_features + 1):
                out[j] = n_samples / (lip_const * out[j])
            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            out.fill(0.0)
            for i in range(n_samples):
                row_start = X_indptr[i]
                row_end = X_indptr[i + 1]
                for idx in range(row_start, row_end):
                    j = X_indices[idx]
                    out[j] += X_data[idx] ** 2
            for j in range(0, n_features):
                out[j] = n_samples / (lip_const * out[j])
            return out

        return learning_rates


def learning_rates_mom_f_factory(X, fit_intercept, n_samples_in_block):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    MOM (median-of-means) estimator with a F-major matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            out[0] = 1 / lip_const
            for j in range(1, n_features + 1):
                out[j] = 1 / (
                    max(
                        median_of_means(X[:, j - 1] * X[:, j - 1], n_samples_in_block),
                        1e-8,
                    )
                    * lip_const
                )
            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            for j in range(n_features):
                out[j] = 1 / (
                    max(median_of_means(X[:, j] * X[:, j], n_samples_in_block), 1e-8)
                    * lip_const
                )
            return out

        return learning_rates


def learning_rates_mom_c_factory(X, fit_intercept, n_samples_in_block):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    MOM (median-of-means) estimator with a C-major matrix X. This
    returns a jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    # TODO: Can we do better with major C ?
    return learning_rates_mom_f_factory(X, fit_intercept, n_samples_in_block)


def learning_rates_mom_csr_factory(X, fit_intercept, n_samples_in_block):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_mom_csc_factory(X, fit_intercept, n_samples_in_block):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_ch_f_factory(X, fit_intercept, eps):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    CH (Catoni-Holland) estimator with a F-major matrix X. This returns a
    jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            squared_coordinates = np.empty(n_samples, dtype=X.dtype)
            out[0] = 1 / lip_const
            for j in range(n_features):
                squared_coordinates[:] = X[:, j] * X[:, j]
                out[j + 1] = 1 / (
                    holland_catoni_estimator(squared_coordinates, eps) * lip_const
                )
            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            squared_coordinates = np.zeros(n_samples, dtype=X.dtype)
            for j in range(n_features):
                squared_coordinates[:] = X[:, j] * X[:, j]
                out[j] = 1 / (
                    holland_catoni_estimator(squared_coordinates, eps) * lip_const
                )

            return out

        return learning_rates


def learning_rates_ch_c_factory(X, fit_intercept, eps):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    CH (Catoni-Holland) estimator with a C-major matrix X. This returns a
    jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    return learning_rates_ch_f_factory(X, fit_intercept, eps)


def learning_rates_ch_csc_factory(X, fit_intercept, eps):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_ch_csr_factory(X, fit_intercept, eps):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_tmean_f_factory(X, fit_intercept, percentage):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    TMean (trimmed-mean) estimator with a F-major matrix X. This returns a
    jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    if fit_intercept:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            n_excluded_tails = int(n_samples * percentage / 2)
            squared_coordinates = np.zeros(n_samples, dtype=X.dtype)
            out[0] = 1 / lip_const
            for j in range(n_features):
                squared_coordinates[:] = X[:, j] * X[:, j]
                squared_coordinates.sort()
                out[j + 1] = 1 / (
                    np.mean(squared_coordinates[n_excluded_tails:-n_excluded_tails])
                    * lip_const
                )

            return out

        return learning_rates
    else:

        @jit(**_jit_kwargs)
        def learning_rates(lip_const, out):
            n_samples, n_features = X.shape
            n_excluded_tails = int(n_samples * percentage / 2)
            squared_coordinates = np.zeros(n_samples, dtype=X.dtype)
            for j in range(n_features):
                squared_coordinates[:] = X[:, j] * X[:, j]
                squared_coordinates.sort()
                out[j] = 1 / (
                    np.mean(squared_coordinates[n_excluded_tails:-n_excluded_tails])
                    * lip_const
                )

            return out

        return learning_rates


def learning_rates_tmean_c_factory(X, fit_intercept, percentage):
    """Learning rates factory for CGD (coordinate gradient descent) when using the
    TMean (trimmed-mean) estimator with a C-major matrix X. This returns a
    jit-compiled function allowing to compute the learning rates (one for
    each coordinate).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) should be added to the
        decision function.

    Returns
    -------
    output : function
        A jit-compiled function allowing to compute the learning rates.

    """
    return learning_rates_tmean_f_factory(X, fit_intercept, percentage)


def learning_rates_tmean_csc_factory(X, fit_intercept, percentage):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_tmean_csr_factory(X, fit_intercept, percentage):
    # TODO: implement this
    raise NotImplementedError()


def learning_rates_factory(X, fit_intercept, estimator):
    X_type = matrix_type(X)
    if isinstance(estimator, ERM):
        if X_type == "f":
            return learning_rates_erm_f_factory(X, fit_intercept)
        elif X_type == "c":
            return learning_rates_erm_c_factory(X, fit_intercept)
        elif X_type == "csc":
            return learning_rates_erm_csc_factory(X, fit_intercept)
        else:
            return learning_rates_erm_csr_factory(X, fit_intercept)
    elif isinstance(estimator, MOM):
        if X_type == "f":
            return learning_rates_mom_f_factory(
                X, fit_intercept, estimator.n_samples_in_block
            )
        elif X_type == "c":
            return learning_rates_mom_c_factory(
                X, fit_intercept, estimator.n_samples_in_block
            )
        elif X_type == "csc":
            return learning_rates_mom_csc_factory(
                X, fit_intercept, estimator.n_samples_in_block
            )
        else:
            return learning_rates_mom_csr_factory(
                X, fit_intercept, estimator.n_samples_in_block
            )
    elif isinstance(estimator, CH):
        if X_type == "f":
            return learning_rates_ch_f_factory(X, fit_intercept, estimator.eps)
        elif X_type == "c":
            return learning_rates_ch_c_factory(X, fit_intercept, estimator.eps)
        elif X_type == "csc":
            return learning_rates_ch_csc_factory(X, fit_intercept, estimator.eps)
        else:
            return learning_rates_ch_csr_factory(X, fit_intercept, estimator.eps)
    elif isinstance(estimator, TMean):
        if X_type == "f":
            return learning_rates_tmean_f_factory(
                X, fit_intercept, estimator.percentage
            )
        elif X_type == "c":
            return learning_rates_tmean_c_factory(
                X, fit_intercept, estimator.percentage
            )
        elif X_type == "csc":
            return learning_rates_tmean_csc_factory(
                X, fit_intercept, estimator.percentage
            )
        else:
            return learning_rates_tmean_csr_factory(
                X, fit_intercept, estimator.percentage
            )
    else:
        raise ValueError("learning_rates_factory for not support this estimator")


def learning_rate_best(X, fit_intercept):
    # Use np.linalg.norm
    # and scipy.sparse ? or svds from scikit-learn
    pass



def learning_rate_sto(X, fit_intercept):

    pass