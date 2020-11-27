import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.special import expit
import pytest

import logging

# logging.basicConfig(level=logging.DEBUG)


from numba import njit

from linlearn.strategy import col_squared_norm_sparse

from linlearn import BinaryClassifier

from sklearn.preprocessing import StandardScaler


def simulate_true_logistic(
    n_samples=150, n_features=5, fit_intercept=True, sparse=False, random_state=42
):
    rng = np.random.RandomState(random_state)
    coef0 = rng.randn(n_features)
    if fit_intercept:
        intercept0 = 0.1
    else:
        intercept0 = 0.0
    X = rng.randn(n_samples, n_features)
    X = StandardScaler().fit_transform(X)
    # X[X < -1] = 0
    # X[X > 1] = 0
    if sparse:
        X = csc_matrix(X)
    logits = X.dot(coef0)
    logits += intercept0
    p = expit(logits)
    y = rng.binomial(1, p, size=n_samples)
    return X, y


n_samples = 500
n_features = 3
fit_intercept = True
X_dense, y = simulate_true_logistic(
    n_samples=n_samples,
    n_features=n_features,
    fit_intercept=fit_intercept,
    sparse=False,
)
# print("simulated dense")
# print(X_dense)
# print(y)

br_dense = BinaryClassifier(tol=1e-20, max_iter=1000).fit(X_dense, y)

print("coef_dense: ", br_dense.coef_)
print("intercept_dense: ", br_dense.intercept_)

X_sparse, y = simulate_true_logistic(
    n_samples=n_samples, n_features=n_features, sparse=True
)
# print("simulated sparse")
# print(X_sparse)
# print(y)

from linlearn.solver import plot_history


#
# @njit
# def f_sparse(indices, indptr, data):
#     for i in range(indices.size):
#         print(indices[i])
#
#     for i in range(indptr.size):
#         print(indices[i])
#     for i in range(data.size):
#         print(indices[i])
#
#
# @njit
# def f_dense(X):
#     print("X dense")
#
#
# def factory(X):
#
#     if issparse(X):
#         X_indices = X.indices
#         X_indptr = X.indptr
#         X_data = X.data
#
#         @njit
#         def f_inner():
#             return f_sparse(X_indices, X_indptr, X_data)
#
#         return f_inner
#
#     else:
#
#         @njit
#         def f_inner():
#             return f_dense(X)
#
#         return f_inner
#
#
# # f = factory(X_dense)
#
# f = factory(X_sparse)
#
# f()


# f(X_sparse.indices, X_sparse.indptr, X_sparse.data)

br_sparse = BinaryClassifier(tol=1e-4, max_iter=1000).fit(X_sparse, y)

print("coef_sparse: ", br_sparse.coef_)
print("intercept_sparse: ", br_sparse.intercept_)

plot_history([br_dense, br_sparse], x="epoch", y="obj", dist_min=True, log_scale=True)

print(br_dense.history_.values["obj"])


# for j in range(n_features):
#     col_start = X.indptr[j]
#     col_end = X.indptr[j + 1]
#     for idx in range(col_start, col_end):
#         i = X.indices[idx]
#         print(i, j, X.data[idx])
#
# col_norms = col_squared_norm_sparse(
#     n_samples, n_features, X_sparse.indptr, X_sparse.data, fit_intercept
# )

# print(col_norms)

# print((X_dense ** 2).sum(axis=0))

# is the standard CSC representation where the row indices for column i are stored
# in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
# data[indptr[i]:indptr[i+1]]. If the shape parameter is not supplied, the matrix
# dimensions are inferred from the index arrays.
