import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz

from linlearn.model import Logistic
from linlearn.model.logistic import sigmoid
from linlearn.solver import SVRG
from linlearn.prox import ProxL2Sq


def simulate(n_samples, w0, b0=None):
    n_features = w0.shape[0]
    cov = toeplitz(0.5 ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    logits = X.dot(w0)
    if b0 is not None:
        logits += b0
    p = sigmoid(logits)
    y = np.random.binomial(1, p, size=n_samples).astype('float64')
    y[:] = 2 * y - 1
    return X, y


n_samples = 1_000_000
n_features = 5
fit_intercept = True

coef0 = np.random.randn(n_features)
if fit_intercept:
    intercept0 = -2.
else:
    intercept0 = None

X, y = simulate(n_samples, coef0, intercept0)

if fit_intercept:
    w = np.zeros(n_features + 1)
else:
    w = np.zeros(n_features)

max_epochs = 10
step = 1e-2


# lr = LogisticRegression(fit_intercept=fit_intercept, max_iter=max_epochs,
#                         step=step, smp=True, verbose=True)
# lr.fit(X, y)
#
# # lr.predict_proba(X)

logistic = Logistic(fit_intercept).set(X, y)
prox = ProxL2Sq(strength=0.)


max_iter = 10

svrg = SVRG(step=step, max_iter=max_iter).set(model=logistic, prox=prox)
svrg.solve(w)


if fit_intercept:
    print(intercept0, coef0)
    print(w)
else:
    print(coef0)
    print(w)
