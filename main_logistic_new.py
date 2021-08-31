import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz

from linlearn._loss_old import steps_coordinate_descent, sigmoid, Logistic
from linlearn._estimator_old import ERM
# from linlearn._penalty import get_penalty
from linlearn._solver_old import CGD, History


from sklearn.preprocessing import StandardScaler

random_state = 42

np.random.seed(random_state)


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
# n_samples = 50_000
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


w0 = np.concatenate(([intercept0], coef0))
# print("w0:", w0)


def run(
    loss,
    estimator,
    penalty,
    w,
    X,
    y,
    fit_intercept,
    max_iter=100,
    tol=1e-7,
    verbose=True,
):
    steps = steps_coordinate_descent(loss.lip, X, fit_intercept)
    cgd = CGD(X, y, loss, fit_intercept, estimator, penalty, max_iter, tol,
              random_state, steps)

    cgd.solve()

    # result = coordinate_gradient_descent(
    #     loss,
    #     estimator,
    #     penalty,
    #     w,
    #     X,
    #     y,
    #     fit_intercept,
    #     steps,
    #     max_iter,
    #     tol,
    #     history,
    #     random_state=42,
    # )
    # w = result.w
    # print("w:", w)
    # return result.w


# loss = get_loss("logistic")
# estimator = get_estimator("erm")
# penalty = get_penalty("none")
# run(loss, estimator, penalty, w, X, y, fit_intercept)


loss = Logistic()
estimator = ERM(X, y, loss, fit_intercept)
penalty = None
# loss = get_loss("logistic")
# estimator = get_estimator("erm")
# penalty = get_penalty("l2", strength=1e-2)

run(loss, estimator, penalty, w, X, y, fit_intercept)


# loss = get_loss("logistic")
# estimator = get_estimator("erm")
# penalty = get_penalty("l1", strength=1e-2)
# run(loss, estimator, penalty, w, X, y, fit_intercept)
#
# loss = get_loss("logistic")
# estimator = get_estimator("erm")
# penalty = get_penalty("elasticnet", strength=1e-2, l1_ratio=1.0)
# run(loss, estimator, penalty, w, X, y, fit_intercept)


# block_size = 500


# estimator = get_estimator("mom", n_samples=n_samples, block_size=block_size)
# penalty = get_penalty("none")

