import numbers
import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression

# from sklearn.datasets import make_classification

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


def make_linear(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        # linearly separable
        # random_state=1,
        # not linearly separable
        random_state=2,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def plot_data(ax, X, y, xy_labels=True, **kwargs):
    X_1 = X[y == 1]
    X_0 = X[y == 0]
    plt.scatter(X_1[:, 0], X_1[:, 1], c="blue", s=30, label=r"$y_i=1$", **kwargs)
    plt.scatter(X_0[:, 0], X_0[:, 1], c="red", s=30, label=r"$y_i=-1$", **kwargs)
    ax.set_xticks(())
    ax.set_yticks(())
    if xy_labels:
        ax.set_xlabel(r"$x_{i,1}$", fontsize=15)
        ax.set_ylabel(r"$x_{i,2}$", fontsize=15)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)


n_samples = 20

X, y = make_linear(n_samples)
ax = plt.subplot(1, 1, 1)

br = BinaryClassifier(tol=1e-17, max_iter=200).fit(X, y)
lr = LogisticRegression(tol=1e-17, max_iter=200).fit(X, y)


print(br.coef_)
print(lr.coef_)

print(br.intercept_)
print(lr.intercept_)

# br.predict(X)

plot_data(ax, X, y)

plt.show()
