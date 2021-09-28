"""
This module contains unittests for some functions from _utils
"""

# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

import pytest

from linlearn._utils import is_in_sorted, whereis_sorted, csr_get, matrix_type, sum_sq


@pytest.mark.parametrize(
    "v", [np.array([]), np.array([3]), np.array([3, 4]), np.array([3, 4, 8]),],
)
@pytest.mark.parametrize(
    "i", range(13),
)
def test_is_in_sorted(v, i):
    in_partition = i in v
    assert is_in_sorted(i, v) == in_partition


@pytest.mark.parametrize(
    "v",
    [
        np.array([]),
        np.array([3]),
        np.array([3, 4]),
        np.array([3, 4, 8], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "i", range(13),
)
def test_whereis_sorted(v, i):
    def argwhere(i, v):
        w = np.argwhere(v == i).ravel()
        if w.size > 0:
            return w[0]
        else:
            return -1

    assert whereis_sorted(i, v) == argwhere(i, v)


@pytest.mark.parametrize("sparsify", ("no", "random", "row", "col"))
def test_csr_get(sparsify):
    n_samples, n_features = 11, 4
    rng = np.random.RandomState(42)
    X_dense = rng.randn(n_samples, n_features)
    if sparsify == "random":
        X_dense[X_dense < 0.0] = 0.0
    elif sparsify == "row":
        X_dense[::2, :] = 0.0
    elif sparsify == "col":
        X_dense[:, ::2] = 0.0

    X_csr = csr_matrix(X_dense)
    rows = []
    for i in range(n_samples):
        row = []
        for j in range(n_features):
            w = csr_get(X_csr.indptr, X_csr.indices, X_csr.data, i, j)
            row.append(w)
        rows.append(row)

    assert X_dense == pytest.approx(np.array(rows), abs=1e-15)


@pytest.mark.parametrize("mtype", ("c", "f", "csr", "csc"))
def test_matrix_type(mtype):
    n_samples, n_features = 11, 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    if mtype == "c":
        X = np.ascontiguousarray(X)
    elif mtype == "f":
        X = np.asfortranarray(X)
    elif mtype == "csc":
        X = csc_matrix(X)
    else:
        X = csr_matrix(X)
    assert mtype == matrix_type(X)


@pytest.mark.parametrize("mtype", ("c", "f", "csr", "csc"))
@pytest.mark.parametrize("sparsify", ("no", "random", "row", "col"))
@pytest.mark.parametrize("axis", (0, 1))
def test_sum_sq(mtype, sparsify, axis):
    n_samples, n_features = 11, 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    if sparsify == "random":
        X[X < 0.0] = 0.0
    elif sparsify == "row":
        X[::2, :] = 0.0
    elif sparsify == "col":
        X[:, :2] = 0.0
    norms = np.linalg.norm(X, 2, axis=axis) ** 2
    if mtype == "c":
        X = np.ascontiguousarray(X)
    elif mtype == "f":
        X = np.asfortranarray(X)
    elif mtype == "csc":
        X = csc_matrix(X)
    else:
        X = csr_matrix(X)
    if axis == 0:
        out = np.empty(n_features)
    else:
        out = np.empty(n_samples)
    sum_sq(X, axis=axis, out=out)
    tol = 1e-12
    assert norms == pytest.approx(out, abs=tol, rel=tol)
    assert norms == pytest.approx(sum_sq(X, axis=axis), abs=tol, rel=tol)
