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
PARALLEL = False

jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}

nb_float = float64
np_float = np.float64

@jit(**jit_kwargs)
def omega(th, p, C):
    return C * ((np.power(np.abs(th), p).sum()) ** (2/p))

@jit(**jit_kwargs)
def grad_omega(th, p, C):
    if th.shape[1] <= 1:
        return 2 * C * ((np.linalg.norm(th.flatten(), p))**(2-p)) * np.sign(th) * np.power(np.abs(th), p-1)
    d, k = th.shape
    euc_norms = np.zeros(d)

    for j in range(th.shape[1]):
        for i in range(d):
            euc_norms[i] += th[i, j] ** 2

    np.power(euc_norms, (2-p)/2, euc_norms)
    # for i in range(d):
    #     euc_norms[i] = np.sqrt(euc_norms[i])

    #euc_norms = np.sqrt(np.power(th, 2).sum(axis=1))
    scaled_th = th.copy()
    # for i in range(d):
    #     if euc_norms[i] > 0:
    #         for j in range(k):
    #             scaled_th[i, j] /= euc_norms[i]

    scale = 0.0
    for j in range(k):
        for i in range(d):
            if euc_norms[i] > 0:
                scaled_th[i, j] /= euc_norms[i]

    for i in range(d):
        scale += (euc_norms[i] ** (p / (2 - p)))

    scale = 2 * C * (scale ** ((2-p)/p))

    for j in range(k):
        for i in range(d):
            scaled_th[i, j] *= scale

    return scaled_th


@jit(**jit_kwargs)
def h(uu, lamda, p):
    if uu.shape[1] > 1:
        stu = np.maximum(np.sqrt((uu ** 2).sum(axis=1)) - lamda, 0)# softthresh(np.sqrt((uu ** 2).sum(axis=1)), lamda)
    else:
        stu = np.maximum(np.abs(uu) - lamda, 0).sum(axis=1)# softthresh(uu, lamda)
    a = np.power(stu, 1 / (p - 1)).sum()
    b = np.power(stu, p / (p - 1)).sum() ** (1 - 2 / p)
    return a * b

# @jit(**jit_kwargs)
# def prox(u, R, p, C):
#     # first figure out lambda
#
#     lamda1, lamda2 = 0, np.max(np.abs(u))
#     while np.abs(lamda2 - lamda1) > 1e-5:
#         mid = (lamda1 + lamda2) / 2
#         if h(u, mid, p)/(2*C) > R:
#             lamda1 = mid
#         else:
#             lamda2 = mid
#     lamda = lamda1
#     if u.shape[1] > 1:
#         stu = np.maximum(np.sqrt((u ** 2).sum(axis=1)) - lamda, 0)# softthresh(np.sqrt((uu ** 2).sum(axis=1)), lamda)
#     else:
#         stu = np.maximum(np.abs(u) - lamda, 0)# softthresh(uu, lamda)
#
#     # stu = softthresh(u, lamda)
#     # return - np.sign(u) * np.power(stu, 1 / (p - 1)) / (
#     #         (2 * C) * (np.linalg.norm(stu.flatten(), p / (p - 1)) ** ((2 - p) / (p - 1))))
#


@jit(**jit_kwargs)
def prox(u, R, p, C):
    # first figure out lambda

    lamda1, lamda2 = 0, np.max(np.abs(u))
    while np.abs(lamda2 - lamda1) > 1e-5:
        mid = (lamda1 + lamda2) / 2
        if h(u, mid, p)/(2*C) > R:
            lamda1 = mid
        else:
            lamda2 = mid
    lamda = lamda1
    if u.shape[1] > 1:
        stu = np.maximum(np.sqrt((u ** 2).sum(axis=1)) - lamda, 0.)# softthresh(np.sqrt((uu ** 2).sum(axis=1)), lamda)
        th = -u.copy()
        for i in range(u.shape[0]):
            if stu[i] <= 0.:
                th[i, :] = 0.
            else:
                th[i, :] /= max(1e-16, stu[i] ** ((p-2)/(p-1)) + lamda * (stu[i] ** (-1/(p-1))))
        return th / (2 * C * (np.power(stu, p / (p - 1)).sum() ** ((2 - p) / p)))
    else:
        stu = np.maximum(np.abs(u) - lamda, 0)# softthresh(uu, lamda)
        th = -np.power(stu, 1/(p-1)) * np.sign(u)
        return th / (2 * C * (np.power(stu, p / (p - 1)).sum() ** ((2 - p) / p)))

    #return th / (2 * C * (np.power(stu, p/(p-1)).sum() ** ((2-p)/p)))
    #return th / (2 * C * (np.power(stu, p / (p - 1)).sum() ** ((2 - p) / p)))

    # stu = softthresh(u, lamda)
    # return - np.sign(u) * np.power(stu, 1 / (p - 1)) / (
    #         (2 * C) * (np.linalg.norm(stu.flatten(), p / (p - 1)) ** ((2 - p) / (p - 1))))



@jit(**jit_kwargs)
def softthresh(u, lamda):
    return np.maximum(np.abs(u) - lamda, 0)

@jit(**jit_kwargs)
def hardthresh(u, k):

    if u.shape[1] <= 1:
        abs_u = np.abs(u.flatten())
    else:
        abs_u = np.sqrt(np.power(u, 2).sum(axis=1))

    thresh = np.partition(abs_u, -k)[-k]
    u_s = u.copy()
    for i in range(u.shape[0]):
        if abs_u[i] < thresh:
            u_s[i, :] = 0
    return u_s


@jit(**jit_kwargs)
def partition(A, p, r, ind):
    A[r], A[ind] = A[ind], A[r]

    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[j], A[i] = A[i], A[j]

    A[i+1], A[r] = A[r], A[i+1]
    return i + 1



@jit(**jit_kwargs)
def findKth_QS(A, n, k):
    N = n
    AA = A
    K = k

    while N > 1:

        kk1 = partition(AA, 0, N-1, N//2)+1
        if kk1 == K:
            break
        elif K < kk1:
            # AA = AA[:kk1-1]
            N = kk1-1
        else:
            AA = AA[kk1:]
            N = N - kk1
            K = K - kk1


@jit(**jit_kwargs)
def findK2(A, n, k1, k2):

    if n <= 1:
        return# k-1
    N = n
    AA = A
    K1 = k1
    K2 = k2

    while True:
        kk1 = partition(AA, 0, N-1, N//2)+1
        if kk1 < K1:
            AA = AA[kk1:]
            N -= kk1
            K1 -= kk1
            K2 -= kk1
        elif kk1 == K1:
            findKth_QS(AA[kk1:], N - kk1, K2 - kk1)
            break
        elif kk1 < K2:
            findKth_QS(AA[:kk1-1], kk1-1, K1)
            findKth_QS(AA[kk1:], N - kk1, K2 - kk1)
            break
        elif kk1 == K2:
            findKth_QS(AA[:kk1-1], kk1-1, K1)
            break
        else:
            AA = AA[:kk1-1]
            N = kk1 - 1


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

@jit(**jit_kwargs)
def trimmed_mean(x, n_samples, n_excluded_tails):  # , percentage):
    # n_excluded_tails = max(1, int(n_samples * percentage))
    n_excluded_tails = int(n_excluded_tails)
    partitioned = np.partition(x, [n_excluded_tails, n_samples - n_excluded_tails - 1])
    result = 0.0
    for i in range(n_excluded_tails, n_samples - n_excluded_tails):
        result += partitioned[i]
    result += partitioned[n_excluded_tails] * n_excluded_tails
    result += partitioned[n_samples - n_excluded_tails-1] * n_excluded_tails
    result /= n_samples
    return result


@jit(**jit_kwargs)
def fast_trimmed_mean(x, n_samples, n_excluded_tails):  # , percentage):

    # n_excluded_tails = max(1, int(n_samples * percentage))
    n_excluded_tails = int(n_excluded_tails)
    # findKth_QS(x, n_samples, n_excluded_tails)
    # findKth_QS(x[n_excluded_tails:], n_samples - n_excluded_tails, n_samples - 2*n_excluded_tails + 1)
    findK2(x, n_samples, n_excluded_tails+1, n_samples - n_excluded_tails)

    result = 0.0
    for i in range(n_excluded_tails, n_samples - n_excluded_tails):
        result += x[i]
    result += x[n_excluded_tails] * n_excluded_tails
    result += x[n_samples-n_excluded_tails-1] * n_excluded_tails
    result /= n_samples

    # alternative code for true theoretical definition

    # half = n_samples // 2
    # n_excluded_tails = max(1, int(half * percentage))
    #
    # findK2(x[:half], half, n_excluded_tails, half - n_excluded_tails + 1)
    # a = x[n_excluded_tails-1]
    # b = x[half - n_excluded_tails]
    # result = 0.0
    # for i in range(half, n_samples):
    #     if x[i] < a:
    #         result += a
    #     elif x[i] < b:
    #         result += x[i]
    #     else:
    #         result += b
    # result /= (n_samples - half)

    return result

@jit(**jit_kwargs)
def fast_median(A, n):
    n2 = n//2
    if n%2 == 1:
        findKth_QS(A, n, n2+1)
        return A[n2]
    else:
        N = n
        AA = A
        K = n2
        upper = n
        while N > 1:
            kk1 = partition(AA, 0, N-1, N//2)+1
            if kk1 == K:
                break
            elif K < kk1:
                # AA = AA[:kk1-1]
                if kk1 > 2:
                    upper -= N - kk1 + 1
                N = kk1-1
            else:
                AA = AA[kk1:]
                N = N - kk1
                K = K - kk1

        mini = A[n2]
        for e in A[n2+1:upper]:
            if e < mini:
                mini = e
        return (A[n2-1] + mini)/2

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
    """Returns the matrix type of the input matrix in the form of a string, indicating
    if it is dense F-major, dense C-major, sparse CSC or sparse CSR. Other types will
    raise an error.

    This function must be used internally only.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Matrix of training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------
    output : {'csc', 'csr', 'f', 'c'}
        The output is
        - 'csc' if X is sparse CSC
        - 'csr' if X is sparse CSR
        - 'c' if it is dense (a 2D numpy array) and C-major
        - 'f' if it is dense (a 2D numpy array) and F-major
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


@jit(**jit_kwargs, parallel=PARALLEL)
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


@jit(**jit_kwargs, parallel=PARALLEL)
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


@jit(**jit_kwargs, parallel=PARALLEL)
def col_sq_sum_csc(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the columns of a sparse CSC matrix X.

    Parameters
    ----------
    n_samples : int
        Number of rows in the input matrix

    n_features : int
        Number of columns in the input matrix

    X_indptr : ndarray
        Array containing X.indptr

    X_indices : ndarray
        Array containing X.indices

    X_data : ndarray
        Array containing X.data

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    for j in prange(n_features):
        col_j_squared_norm = 0.0
        col_start = X_indptr[j]
        col_end = X_indptr[j + 1]
        for idx in range(col_start, col_end):
            col_j_squared_norm += X_data[idx] ** 2
        out[j] = col_j_squared_norm


@jit(**jit_kwargs, parallel=PARALLEL)
def col_sq_sum_csr(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the columns of a sparse CSR matrix X.

    Parameters
    ----------
    n_samples : int
        Number of rows in the input matrix

    n_features : int
        Number of columns in the input matrix

    X_indptr : ndarray
        Array containing X.indptr

    X_indices : ndarray
        Array containing X.indices

    X_data : ndarray
        Array containing X.data

    out : numpy.array of shape (n_features,)
        Array containing the sums of squares of the columns of X
    """
    out.fill(0.0)
    for i in prange(n_samples):
        row_start = X_indptr[i]
        row_end = X_indptr[i + 1]
        for idx in range(row_start, row_end):
            j = X_indices[idx]
            out[j] += X_data[idx] ** 2


@jit(**jit_kwargs, parallel=PARALLEL)
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


@jit(**jit_kwargs, parallel=PARALLEL)
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


@jit(**jit_kwargs, parallel=PARALLEL)
def row_sq_sum_csc(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the rows of a sparse CSC matrix X.

    Parameters
    ----------
    n_samples : int
        Number of rows in the input matrix

    n_features : int
        Number of columns in the input matrix

    X_indptr : ndarray
        Array containing X.indptr

    X_indices : ndarray
        Array containing X.indices

    X_data : ndarray
        Array containing X.data

    out : numpy.array of shape (n_samples,)
        Array containing the sums of squares of the rows of X
    """
    out.fill(0.0)
    for j in prange(n_features):
        col_start = X_indptr[j]
        col_end = X_indptr[j + 1]
        for idx in range(col_start, col_end):
            i = X_indices[idx]
            out[i] += X_data[idx] ** 2


@jit(**jit_kwargs, parallel=PARALLEL)
def row_sq_sum_csr(n_samples, n_features, X_indptr, X_indices, X_data, out):
    """Computes the sums of squares of the rows of a sparse CSR matrix X.

    Parameters
    ----------
    n_samples : int
        Number of rows in the input matrix

    n_features : int
        Number of columns in the input matrix

    X_indptr : ndarray
        Array containing X.indptr

    X_indices : ndarray
        Array containing X.indices

    X_data : ndarray
        Array containing X.data

    out : numpy.array of shape (n_samples,)
        Array containing the sums of squares of the rows of X
    """
    for i in prange(n_samples):
        row_i_squared_norm = 0.0
        row_start = X_indptr[i]
        row_end = X_indptr[i + 1]
        for idx in range(row_start, row_end):
            row_i_squared_norm += X_data[idx] ** 2
        out[i] = row_i_squared_norm


def sum_sq(X, axis, out=None):
    """Computes the sum of squares (squared Euclidean norm) along given axis. It
    supports efficient implementations both for F and C-major numpy arrays and sparse
    CSC and CSR matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Matrix of training vectors, where n_samples is the number of samples and
        n_features is the number of features. This function supports F-major and
        C-major ndarray, CSC and CSR sparse matrices.

    axis : {0, 1}
        Axis along which with compute the sum of squares.

    out : ndarray or None
        If ndarray, then the sum of squares are saved in the given vector and is
        returned. Otherwise a new ndarray is allocated. Note that ``out`` has shape
        (n_features,) if ``axis=0`` and (n_samples,) if ``axis=1 ``.

    Returns
    -------
    output : ndarray
        An array containing the sum of squares along axis. Note that ``out`` has shape
        (n_features,) if ``axis=0`` and (n_samples,) if ``axis=1``. If ``out`` is
        used, this is the same ndarray as the one passed.
    """
    n_samples, n_features = X.shape
    mtype = matrix_type(X)
    if axis == 0:
        if out is None:
            out = np.empty(n_features)
        if mtype == "f":
            col_sq_sum_f(X, out)
        elif mtype == "c":
            col_sq_sum_c(X, out)
        elif mtype == "csc":
            col_sq_sum_csc(n_samples, n_features, X.indptr, X.indices, X.data, out)
        elif mtype == "csr":
            col_sq_sum_csr(n_samples, n_features, X.indptr, X.indices, X.data, out)
        else:
            raise NotImplementedError()
    elif axis == 1:
        if out is None:
            out = np.empty(n_samples)
        if mtype == "f":
            row_sq_sum_f(X, out)
        elif mtype == "c":
            row_sq_sum_c(X, out)
        elif mtype == "csc":
            row_sq_sum_csc(n_samples, n_features, X.indptr, X.indices, X.data, out)
        elif mtype == "csr":
            row_sq_sum_csr(n_samples, n_features, X.indptr, X.indices, X.data, out)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Only axis=0 and axis=1 are supported")

    return out


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
    # void(uintp, float64[:], uintp[:]),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={"csum": float64[:], "i": uintp},
)
def rand_choice_nb(size, csum_probs, out):
    for i in range(size):
        out[i] = np.searchsorted(csum_probs, np.random.random(), side="right")

@jit(**jit_kwargs)
def numba_seed_numpy(rnd_state):
    np.random.seed(rnd_state)
