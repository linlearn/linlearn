import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz

from linlearn.model.logistic import sigmoid


def simulate_log_reg(n_samples, coef, intercept=None):
    n_features = coef.shape[0]
    cov = toeplitz(0.5 ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    logits = X.dot(coef)
    if intercept is not None:
        logits += intercept
    p = sigmoid(logits)
    y = np.random.binomial(1, p, size=n_samples).astype("float64")
    y[:] = 2 * y - 1
    return X, y
