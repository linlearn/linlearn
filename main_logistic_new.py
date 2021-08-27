import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz

from linlearn._loss import get_loss, steps_coordinate_descent, sigmoid
from linlearn._estimator import get_estimator
from linlearn._solver import coordinate_gradient_descent, History


from sklearn.preprocessing import StandardScaler


def simulate(n_samples, w0, b0=None):
    n_features = w0.shape[0]
    cov = toeplitz(0.5 ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    X = StandardScaler().fit_transform(X)
    logits = X.dot(w0)
    if b0 is not None:
        logits += b0
    p = sigmoid(logits)
    y = np.random.binomial(1, p, size=n_samples).astype("float64")
    y[:] = 2 * y - 1
    y = y.astype("float64")
    return X, y


n_samples = 100_000
# n_samples = 1_000
# n_features = 5
n_features = 100
fit_intercept = True

coef0 = np.random.randn(n_features)
if fit_intercept:
    intercept0 = -2.0
else:
    intercept0 = None

X, y = simulate(n_samples, coef0, intercept0)

if fit_intercept:
    w = np.zeros(n_features + 1)
else:
    w = np.zeros(n_features)


block_size = 500

loss = get_loss("logistic")
estimator = get_estimator("mom", n_samples=n_samples, block_size=block_size)
steps = steps_coordinate_descent(loss.state.lip, X, fit_intercept)
max_iter = 10
tol = 1e-7
verbose = True

history = History("CGD", max_iter, verbose)

coordinate_gradient_descent(
    loss, estimator, None, w, X, y, fit_intercept, steps, max_iter, tol, history,
)
