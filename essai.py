from numpy.random.mtrand import multivariate_normal
import pytest

from linlearn import BinaryClassifier

#
# def simulate_true_logistic(n_samples=150, n_features=5, fit_intercept=True, corr=0.5):
#     rng = np.random.RandomState(42)
#     coef0 = rng.randn(n_features)
#     if fit_intercept:
#         intercept0 = -2.0
#     else:
#         intercept0 = 0.0
#
#     cov = toeplitz(corr ** np.arange(0, n_features))
#     X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
#     logits = X.dot(coef0)
#     logits += intercept0
#     p = expit(logits)
#     y = rng.binomial(1, p, size=n_samples)
#     return X, y
#
#
# n_samples = 128
# n_features = 5
# verbose = False
#
# X, y = simulate_true_logistic(
#     n_samples=n_samples, n_features=n_features, fit_intercept=True,
# )

# X = pd.DataFrame(X)
# print(X.head())
#
#
# # TODO: ca ne passe pas quand y est une Serie ou un pandas alors que ca raise juste un
# #  warning avec scikit
# # y = pd.DataFrame(y)
#
# LogisticRegression().fit(X, y)
# # BinaryClassifier().fit(X, y)
#
# # TODO: juste une colonne de pandas dataframe
#
# print(np.unique(y))
# # br = BinaryClassifier()
# # br.fit(X, y)
#
# weird = {0: "neg", 1: "pos"}
#
# y_weird = [weird[yi] for yi in y]
#
# LogisticRegression().fit(X, y_weird)
# BinaryClassifier().fit(X, y_weird)
#
# print(y_weird)
#
# # TODO: test sparse
# # TODO: test weird label


# args = {
#     "tol": tol,
#     "max_iter": max_iter,
#     "verbose": verbose,
#     "fit_intercept": fit_intercept,
#     "random_state": 42,
# }

# X, y = make_classification()

#
# def make_linear(n_samples):
#     X, y = make_classification(
#         n_samples=150,
#         n_features=2,
#         n_redundant=0,
#         n_informative=2,
#         random_state=42,
#         shift=1.0,
#         n_clusters_per_class=1,
#     )
#     # rng = np.random.RandomState(2)
#     # X += 2 * rng.uniform(size=X.shape)
#     return X, y


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def simulate_linear(n_samples, random_state=2):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


n_samples = 20
X, y = simulate_linear(n_samples)
X_df = pd.DataFrame(X)

weird = {0: "neg", 1: "pos"}
y_weird = [weird[yi] for yi in y]

br = BinaryClassifier(tol=1e-17, max_iter=200).fit(X_df, y_weird)
lr = LogisticRegression(tol=1e-17, max_iter=200).fit(X_df, y_weird)

assert br.intercept_ == pytest.approx(lr.intercept_, abs=1e-4)
assert br.coef_ == pytest.approx(lr.coef_, abs=1e-4)

# And test prediction methods
assert lr.decision_function(X) == pytest.approx(br.decision_function(X), abs=1e-4)
assert lr.predict_proba(X) == pytest.approx(br.predict_proba(X), abs=1e-4)
assert lr.predict_log_proba(X) == pytest.approx(br.predict_log_proba(X), abs=1e-4)

print(br.predict(X))
print(lr.predict(X))

assert (lr.predict(X) == br.predict(X)).any()

# assert lr.score(X, y) == br.score(X, y)

classes = {0: "truc", 1: "machin"}
#print(classes[np.array([0, 0, 1, 0, 0], dtype="int")])
