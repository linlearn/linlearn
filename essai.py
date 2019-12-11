
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

ls = LeastSquares(fit_intercept=True).set(X, y)

ls.fit_intercept = False

print(ls)
