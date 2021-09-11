import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.special import expit
from time import time

from linlearn import BinaryClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import safe_sparse_dot

from linlearn._solvers import plot_history

from linlearn._loss import decision_function_factory, Logistic
from linlearn._utils import np_float, is_in_sorted, csr_get, matrix_type
from linlearn._estimator import ERM

from numba import jit


X = np.random.randn(11, 4)
X_csc = csc_matrix(X)


def steps_factory(X):
    mtype = matrix_type(X)

    if mtype in {'csc', 'csr'}:
        X_indptr = X.indptr
        X_indices = X.indices
        X_data = X.data

        @jit(nopython=True, nogil=True, boundscheck=True, fastmath=True)
        def steps(lip_const):
            return lip_const * X_data[X_indptr[0]]

        return steps
    else:
        @jit(nopython=True, nogil=True, boundscheck=True, fastmath=True)
        def steps(lip_const):
            return lip_const * X[0, 0]

        return steps


steps = steps_factory(X)
print(steps(3.14))
steps = steps_factory(X_csc)
print(steps(3.14))

exit(0)

from linlearn._loss import steps_erm_factory


fit_intercept = True
lip_const = 0.25

# Pre-compilation step
n_samples, n_features = 11, 4
n_weights = n_features + int(fit_intercept)
X_small = np.random.randn(n_samples, n_features)

X_c = np.ascontiguousarray(X_small)
X_f = np.asfortranarray(X_small)
X_csc = csc_matrix(X_small)
X_csr = csr_matrix(X_small)

steps_func_1 = steps_erm_factory(X_c, fit_intercept)
steps_func_2 = steps_erm_factory(X_f, fit_intercept)
steps_func_3 = steps_erm_factory(X_csc, fit_intercept)
steps_func_4 = steps_erm_factory(X_csr, fit_intercept)

steps1 = steps_func_1(lip_const, X_c)
steps2 = steps_func_2(lip_const, X_f)
# steps3 = steps_func_3(lip_const, X_csc)
# steps4 = steps_func_4(lip_const, X_csr)


n_samples, n_features = 1_000_000, 100
n_weights = n_features + int(fit_intercept)
X_big = np.random.randn(n_samples, n_features)

X_c = np.ascontiguousarray(X_big)
X_f = np.asfortranarray(X_big)
X_csc = csc_matrix(X_big)
X_csr = csr_matrix(X_big)


steps_func_1 = steps_erm_factory(X_c, fit_intercept)
steps_func_2 = steps_erm_factory(X_f, fit_intercept)
# steps_func_3 = steps_erm_factory(X_csc, fit_intercept)
# steps_func_4 = steps_erm_factory(X_csr, fit_intercept)


tic = time()
steps = steps_func_1(lip_const, X_c)
toc = time()
print("c c")
print(toc - tic)

tic = time()
steps = steps_func_1(lip_const, X_f)
toc = time()
print("c f")
print(toc - tic)

tic = time()
steps = steps_func_2(lip_const, X_c)
toc = time()
print("f c")
print(toc - tic)

tic = time()
steps = steps_func_2(lip_const, X_f)
toc = time()
print("f f")
print(toc - tic)
# print(steps)

exit(0)


fit_intercept = True

n_samples, n_features = 11, 4
n_weights = n_features + int(fit_intercept)
X_dense = np.random.randn(n_samples, n_features)
# X_dense[X_dense < 0.0] = 0.0
X_dense[::2, :] = 0.0
# X_dense[:, ::2] = 0.0
X_csc = csc_matrix(X_dense)
X_csr = csr_matrix(X_dense)

print("dense")
print(X_dense)

rows = []
for i in range(n_samples):
    row = []
    for j in range(n_features):
        w = csr_get(X_csr.indptr, X_csr.indices, X_csr.data, i, j)
        row.append(w)
    rows.append(row)

print(np.array(rows))


exit(0)
print("CSC")
print(X_csc.indptr)
print(X_csc.indices)
print(X_csc.data)

print("CSR")
print(X_csr.indptr)
print(X_csr.indices)
print(X_csr.data)

exit(0)

X_csc = csc_matrix(X_dense)
print(X_csc.shape)
print(X_csc.nnz)
y = np.ones(n_samples, dtype=np_float)
y[: (n_samples // 2)] *= -1
np.random.shuffle(y)
loss = Logistic()
estimator_dense = ERM(X_dense, y, loss, fit_intercept)
estimator_csr = ERM(X_csr, y, loss, fit_intercept)
estimator_csc = ERM(X_csc, y, loss, fit_intercept)
w = np.random.randn(n_weights)

state_dense = estimator_dense.get_state()
state_csc = estimator_csc.get_state()

partial_deriv_dense = estimator_dense.partial_deriv_factory()
# partial_deriv_csr = estimator_csr.partial_deriv_factory()
partial_deriv_csc = estimator_csc.partial_deriv_factory()

inner_products = np.random.randn(n_samples)

for j in range(n_weights):
    deriv_dense = partial_deriv_dense(1, inner_products, state_dense)
    deriv_csc = partial_deriv_csc(1, inner_products, state_dense)
    assert deriv_dense == pytest.approx(deriv_csc, abs=1e-10)


exit(0)


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
    if sparse == "csc":
        X = csc_matrix(X)
    elif sparse == "csr":
        X = csr_matrix(X)

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

X_dense, y = simulate_true_logistic(
    n_samples=n_samples, n_features=n_features, sparse=False
)


print(type(X_dense))

decision_function = decision_function_factory(X_dense, fit_intercept)

w = np.random.randn(n_features + int(fit_intercept))
out = np.empty(n_samples)
decision_function(w, out)

X_csr, y = simulate_true_logistic(
    n_samples=n_samples, n_features=n_features, sparse="csr"
)

decision_function = decision_function_factory(X_csr, fit_intercept)

w = np.random.randn(n_features + int(fit_intercept))
out = np.empty(n_samples)
decision_function(w, out)

print(out)

# v = safe_sparse_dot(X_sparse, np.random.randn(3))
# print(v)

# v = X_sparse.dot(np.random.randn(3))
# print(v)

exit(0)
# print("simulated sparse")
# print(X_sparse)
# print(y)


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
