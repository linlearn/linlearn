"""
This module contains some utilities used throughout linlearn.
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

import numpy as np
from numpy.random import randint
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc
from numba import jit, void, uintp, prange, float64


# Numba flags applied to all jit decorators
NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
FASTMATH = True


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}

nb_float = float64
np_float = np.float64


def get_type(class_):
    """Gives the numba type of an object if numba.jit decorators are enabled and None
    otherwise. This helps to get correct coverage of the code

    Parameters
    ----------
    class_ : `object`
        A class

    Returns
    -------
    output : `object`
        A numba type of None
    """
    class_type = getattr(class_, "class_type", None)
    if class_type is None:
        return lambda *_: None
    else:
        return class_type.instance_type


@jit(
    void(uintp[:], uintp[:]),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={"n_samples": uintp, "population_size": uintp, "i": uintp, "j": uintp},
)
def sample_without_replacement(pool, out):
    """Samples integers without replacement from pool into out inplace.

    Parameters
    ----------
    pool : ndarray of size population_size
        The array of integers to sample from (it containing [0, ..., n_samples-1]

    out : ndarray of size n_samples
        The sampled subsets of integer
    """
    # We sample n_samples elements from the pool
    n_samples = out.shape[0]
    population_size = pool.shape[0]
    # Initialize the pool
    for i in range(population_size):
        pool[i] = i

    for i in range(n_samples):
        j = randint(population_size - i)
        out[i] = pool[j]
        pool[j] = pool[population_size - i - 1]


@jit(
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH
    # locals={"size": uintp, "left": intp, "right": intp, "middle": intp},
)
def is_in_sorted(i, v):
    """Tests if i value i is in v.

    Parameters
    ----------
    i : int
        The index

    v : ndarray
        array of shape (size,)

    Returns
    -------
    output : bool
        True if i is in v, False otherwise
    """
    size = v.size
    if size == 0:
        return False
    elif size == 1:
        return v[0] == i
    else:
        # We perform a binary search
        left, right = 0, size - 1
        while left <= right:
            middle = (left + right) // 2
            if v[middle] == i:
                return True
            else:
                if v[middle] < i:
                    left = middle + 1
                else:
                    right = middle - 1
        return False


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
def whereis_sorted(i, v):
    size = v.size
    if size == 0:
        return -1
    elif size == 1:
        if v[0] == i:
            # i is the only element in v at index 0
            return 0
        else:
            # i is not in v
            return -1
    else:
        # We perform a binary search
        left, right = 0, size - 1
        while left <= right:
            middle = (left + right) // 2
            if v[middle] == i:
                return middle
            else:
                if v[middle] < i:
                    left = middle + 1
                else:
                    right = middle - 1
        return -1


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
def csr_get(indptr, indices, data, i, j):
    """

    Parameters
    ----------
    data
    indices
    indptr
    i
    j

    Returns
    -------

    """
    row_start = indptr[i]
    row_end = indptr[i + 1]
    idx = whereis_sorted(j, indices[row_start:row_end])
    if idx >= 0:
        return data[row_start + idx]
    else:
        return 0.0


def matrix_type(X):
    """
    Returns the 'matrix type' of the input matrix in the form of a string, indicating
    if it is dense F-major, dense C-major, sparse CSC or sparse CSR. Other types will
    raise an error.

    This function must be used internally only.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------
    output : {'csc', 'csr', 'f', 'c'}
        The output is
         * 'csc' if X is sparse CSC
         * 'csr' if X is sparse CSR
         * 'c' if it is dense (a 2D numpy array) and C-major
         * 'f' if it is dense (a 2D numpy array) and F-major
    """
    if issparse(X):
        if isspmatrix_csc(X):
            return "csc"
        elif isspmatrix_csr(X):
            return "csr"
        else:
            raise ValueError("Only sparse CSC and CSR matrices are supported.")
    else:
        if X.flags.c_contiguous:
            return "c"
        elif X.flags.f_contiguous:
            return "f"
        else:
            raise ValueError("Only C and F-major numpy arrays are supported.")


@jit(**jit_kwargs, parallel=True)
def col_sq_sum_f(X, out):
    """Computes the sums of squares of the columns of the F-major matrix X.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input F-major matrix

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    n_samples, n_features = X.shape
    for j in prange(n_features):
        col_j_squared_norm = 0.0
        for i in range(n_samples):
            col_j_squared_norm += X[i, j] ** 2
        out[j] = col_j_squared_norm


@jit(**jit_kwargs, parallel=True)
def col_sq_sum_c(X, out):
    """Computes the sums of squares of the columns of the C-major matrix X.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input C-major matrix

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    n_samples, n_features = X.shape
    out.fill(0.0)
    for i in prange(n_samples):
        for j in range(n_features):
            out[j] += X[i, j] ** 2

    return out


@jit(**jit_kwargs, parallel=True)
def col_sq_sum_csc(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the columns of the F-major matrix X.

    Parameters
    ----------
    n_samples : int
        Number of rows in the input matrix

    n_features : int
        Number of columns in the input matrix


    X : scipy.sparse.csc_matrix of shape (n_samples, n_features)
        Input sparse CSC matrix

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    for j in prange(n_features):
        col_j_squared_norm = 0.0
        col_start = X_indptr[j]
        col_end = X_indptr[j + 1]
        for i in range(n_samples):
            col_j_squared_norm += X[i, j] ** 2
        out[j] = col_j_squared_norm



@jit(**jit_kwargs, parallel=True)
def col_sq_sum_csr(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the columns of the F-major matrix X.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input F-major matrix

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    # for j in prange(n_features):
    #     col_j_squared_norm = 0.0
    #     for i in range(n_samples):
    #         col_j_squared_norm += X[i, j] ** 2
    #     out[j] = col_j_squared_norm

    pass


@jit(**jit_kwargs, parallel=True)
def row_sq_sum_f(X, out):
    """Computes the sums of squares of the rows of the F-major matrix X.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input F-major matrix

    out : numpy.array of shape (n_samples,)
        Array containing the sums of squares of the rows of X
    """
    n_samples, n_features = X.shape
    out.fill(0.0)
    for j in prange(n_features):
        for i in range(n_samples):
            out[i] += X[i, j] ** 2


@jit(**jit_kwargs, parallel=True)
def row_sq_sum_c(X, out):
    """Computes the sums of squares of the rows of the C-major matrix X.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input C-major matrix

    out : numpy.array of shape (n_samples,)
        Array containing the sums of squares of the rows of X
    """
    n_samples, n_features = X.shape
    for i in prange(n_samples):
        row_i_squared_norm = 0.0
        for j in range(n_features):
            row_i_squared_norm += X[i, j] ** 2
        out[i] = row_i_squared_norm

    return out



def sum_sq(X, axis, out=None):
    n_samples, n_features = X.shape
    mtype = matrix_type(X)
    if axis == 0:
        if out is None:
            out = np.empty(n_features)
        if mtype == "f":
            col_sq_sum_f(X, out)
        elif mtype == "c":
            col_sq_sum_c(X, out)
        else:
            raise NotImplementedError()
    elif axis == 1:
        if out is None:
            out = np.empty(n_samples)
        if mtype == "f":
            row_sq_sum_f(X, out)
        elif mtype == "c":
            row_sq_sum_c(X, out)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Only axis=0 and axis=1 are supported")

    return out
