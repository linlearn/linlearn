import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])


def make_linear(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


n_samples = 150

datasets = [
    (
        make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
        "circles",
    ),
    (make_moons(n_samples=n_samples, noise=0.2, random_state=0), "moons"),
    (make_linear(n_samples=n_samples), "linear"),
]

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])


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


#
# n_datasets = len(datasets)
# plt.figure(figsize=(10, 3))
#
# for i, ((X, y), name) in enumerate(datasets):
#     ax = plt.subplot(1, n_datasets, i + 1)
#     plot_data(ax, X, y, alpha=0.5)
#     plt.title(name, fontsize=18)
#
# plt.legend(fontsize=14, loc="center right", bbox_to_anchor=(1.7, 0.5))
# plt.tight_layout()
#
# plt.show()

# plt.savefig("session04_toy_samples.pdf")


clf = LogisticRegression()

n_datasets = len(datasets)
plt.figure(figsize=(12, 3.5))
h = 0.02
levels = 20


def plot_hyperplane(clf, x_min=-10.0, x_max=10.0):
    if isinstance(clf, Pipeline):
        w1, w2 = clf["logreg"].coef_[0]
        b = clf["logreg"].intercept_
    else:
        w1, w2 = clf.coef_[0]
        b = clf.intercept_

    x = np.linspace(-10, 10, 1000)
    y = -(b + w1 * x) / w2
    plt.plot(
        x, y, lw=2, ls="--", color="black", alpha=0.6, label=r"$w \cdot x + b = 0$"
    )


def plot_decision(clf, ax, X, y, h=0.02, levels=20):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ct = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.7, levels=levels)
    cbar = plt.colorbar(ct)
    cbar.ax.set_xlabel(r"$x \cdot w + b$")


for i, ((X, y), name) in enumerate(datasets):
    ax = plt.subplot(1, n_datasets, i + 1)
    clf.fit(X, y)
    plot_decision(clf, ax, X, y, h=h, levels=levels)
    plot_hyperplane(clf, x_min=X[:, 0].min(), x_max=X[:, 0].max())
    plot_data(ax, X, y, xy_labels=False, alpha=0.5)
    plt.title(name, fontsize=18)

#  lgd = plt.legend(fontsize=14)

# lgd = plt.legend(fontsize=14, loc="lower center", ncol=3, bbox_to_anchor=(-0.9, -0.35))
# plt.tight_layout()
plt.show()

# plt.savefig("session04_logistic_decision_functions.pdf",
#             bbox_extra_artists=(lgd,), bbox_inches='tight')
