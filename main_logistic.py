import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, csc_matrix

# from linlearn.model import Logistic
# from linlearn.model.logistic import sigmoid
# from linlearn.solver_old import SVRG
# from linlearn.prox_old import ProxL2Sq


np.random.seed(42)

from linlearn._loss import (
    # logistic_value_single,
    # logistic_value_batch,
    sigmoid,
    # logistic_derivative,
    # logistic_lip,
    # steps_coordinate_descent,
)

# from linlearn._penalty import l2sq_apply_single, l2sq_value, l1_apply_single, l1_value
# from linlearn.solver import coordinate_gradient_descent
# from linlearn.solver import History

from sklearn.preprocessing import StandardScaler


def simulate(n_samples, w0, b0=None, matrix_type="f", sparsify=True):
    n_features = w0.shape[0]
    cov = toeplitz(0.5 ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    if sparsify:
        X[X < 0.0] = 0.0
    if matrix_type == "f":
        X = np.asfortranarray(X)
    elif matrix_type == "c":
        X = np.ascontiguousarray(X)
    elif matrix_type == "csc":
        X = csc_matrix(X)
    else:
        X = csr_matrix(X)

    logits = X.dot(w0)
    if b0 is not None:
        logits += b0
    p = sigmoid(logits)
    y = np.random.binomial(1, p, size=n_samples).astype("float64")
    y[:] = 2 * y - 1
    y = y.astype("float64")
    return X, y


# n_samples = 100_000
# n_samples = 1_000
n_samples = 1000
n_features = 5
fit_intercept = True

coef0 = np.random.randn(n_features)
if fit_intercept:
    intercept0 = -2.0
else:
    intercept0 = None

X, y = simulate(n_samples, coef0, intercept0, matrix_type="csc", sparsify=True)

# if fit_intercept:
#     w = np.zeros(n_features + 1)
# else:
#     w = np.zeros(n_features)
#
# steps = steps_coordinate_descent(logistic_lip, X, fit_intercept)
# print(steps)
#
# exit(0)
# step = 1e-2
fit_intercept = True


np.set_printoptions(precision=4)

print("Ground truth")
if fit_intercept:
    print(np.array([intercept0]), coef0)
else:
    print(coef0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


penalty = "l2"
C = 10
tol = 1e-13
max_iter = 100
verbose = True

args = {
    "loss": "logistic",
    # "estimator": "mom",
    # "estimator": "erm",
    # "estimator": "tmean",
    # "estimator": "catoni",
    # "block_size": 0.5,
    # "percentage": 0.01,
    "solver": "cgd",
    "penalty": penalty,
    "l1_ratio": 1.0,
    "tol": tol,
    "max_iter": max_iter,
    "C": C,
    "verbose": verbose,
    "fit_intercept": True,
    "random_state": 42,
}

# TODO: ca a l'air OK pour l2 mais pas pour l1 grrrrr

# For l2:
# # for solver in ["saga", "sag", "lbfgs"]:
# for solver in ["saga"]:
#     clf = LogisticRegression(solver=solver, **args).fit(X, y)
#     print(clf)
#     # print("scikit-learn LogisticRegression with solver = %s" % solver)
#     print(clf.intercept_, clf.coef_.ravel())
#     # print("log-loss:", log_loss(y, clf.predict_proba(X)[:, 1]))
#     # print(clf.n_iter_)
# # # TODO: check that the log-likelihood is exactly the same as scikit's

#
# print("clf.n_iter_: ", clf.n_iter_)
# print("clf.classes_: ", clf.classes_)
# print("clf.n_iter_: ", clf.n_iter_)
# print("clf.n_iter_: ", clf.n_iter_)

from linlearn.learner import BinaryClassifier

# TOD0: pour l1on arrete trop trop les iterations...a cause du critere d'arret
# args["tol"] = 0.0

learners = []

# estimators = ["erm", "mom", "catoni", "tmean"]
estimators = ["erm"]

for estimator in estimators:
    args["estimator"] = estimator
    clf = BinaryClassifier(**args).fit(X, y)
    print("estimator:", estimator)
    print(clf.intercept_, clf.coef_.ravel())
    learners.append(clf)


# args["estimator"] = "mom"
# clf = BinaryClassifier(**args).fit(X, y)
# print(clf)
# print(clf.intercept_, clf.coef_.ravel())

# from linlearn._solver import plot_history
#
# plot_history(
#     learners, x="epoch", y="obj", log_scale=True, labels=estimators, dist_min=True
# )
