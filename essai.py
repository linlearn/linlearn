
import numpy as np
from linlearn.model import LeastSquares

from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz


n_samples = 10_000_000
epoch_size = n_samples
n_features = 5
fit_intercept = True

coef0 = np.random.randn(n_features)
intercept0 = -2.

cov = toeplitz(0.5 ** np.arange(0, n_features))
X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)


y = X.dot(coef0) + 0.1 * np.random.randn(n_samples)
if fit_intercept:
    y += intercept0

X = np.zeros((42, 3))
y = np.zeros((42, ))

print("0: ", id(y))

ls = LeastSquares(fit_intercept=True).set(X, y)

# print(X is ls.X)
#
# print(id(X), id(ls.X))
# print(id(y), id(ls.y))



# 1.
# X = np.zeros((42, 3))
# y = np.zeros((41, ))
# ls = LeastSquares(fit_intercept=True).set(X, y)

# 2.
# X = np.zeros((42, 3))
# y = np.zeros((42, 2))
# ls = LeastSquares(fit_intercept=True).set(X, y)

# 3.
# X = np.zeros((42, 3))
# X[0, 0] = np.nan
# y = np.zeros(42,)
# ls = LeastSquares(fit_intercept=True).set(X, y)

# 4.
# X = np.zeros((42, 3))
# y = np.zeros(42,)
# y[0] = np.nan
# ls = LeastSquares(fit_intercept=True).set(X, y)

# 5.
# X = np.zeros((42, 3))
# y = np.zeros((42, ))
# sample_weight = np.zeros(41)
# ls = LeastSquares(fit_intercept=True).set(X, y, sample_weight)

# 6.
# X = np.zeros((42, 3))
# y = np.zeros((42, ))
# sample_weight = np.zeros(42)
# sample_weight[0] = np.nan
# ls = LeastSquares(fit_intercept=True).set(X, y, sample_weight)


# NB : no dtype checks, no warnings

# Verifier que quand on passe des dtypes float32 et integer ou meme object, ca
# passe quand même dans le model ???
# # 7.
# X = np.random.binomial(1, 0.5, size=(42, 3))
# y = np.zeros((42,), dtype=np.int64)
# ls = LeastSquares(fit_intercept=True).set(X, y)
# print(ls.X.dtype)
# print(ls.y.dtype)
#
# # 8.
# X = np.zeros((42, 3), dtype=np.float32)
# y = np.zeros((42,), dtype=np.int64)
# ls = LeastSquares(fit_intercept=True).set(X, y)
# print(ls.X.dtype)
# print(ls.y.dtype)

# 9.
# X = np.zeros((42, 3), dtype=np.float32)
# y = np.array(21 * ['boo'] + 21 * ['baa'])
# ls = LeastSquares(fit_intercept=True).set(X, y)
# # ValueError: could not convert string to float: 'boo'
# print(ls.X.dtype)
# print(ls.y.dtype)

# 10.
# X = np.array(42 * 3 * ['boo']).reshape(42, 3)
# y = np.zeros((42,), dtype=np.int64)
# ls = LeastSquares(fit_intercept=True).set(X, y)
# # ValueError: could not convert string to float: 'boo'



# X = np.((42, 3), dtype=np.int)
# y = np.zeros((42, ))
# sample_weight = np.zeros(42)
# sample_weight[0] = -1.0
# ls = LeastSquares(fit_intercept=True).set(X, y, sample_weight)


# X = np.zeros((42, 3))
# y = np.zeros((42, ))
# sample_weight = np.zeros(42)
# ls = LeastSquares(fit_intercept=True).set(X, y, sample_weight)

# ls.fit_intercept = False
#
# print(ls)
#
#
# print(np.empty(0))
#
# n_samples = 200
#
# print(np.linspace(0, n_samples-1, num=n_samples, dtype=np.float64))

