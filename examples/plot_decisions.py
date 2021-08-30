# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
"""
Comparisons of decision functions
=================================

This example allows to compare the decision functions of several random forest types
of estimators. The following classifiers are used:

- **AMF** stands for `AMFClassifier` from `onelearn`
- **MF** stands for `MondrianForestClassifier` from `scikit-garden`
- **RF** stands for `RandomForestClassifier` from `scikit-learn`
- **ET** stands for `ExtraTreesClassifier` from `scikit-learn`
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.model_selection import train_test_split

# from skgarden import MondrianForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


sys.path.extend([".", ".."])

from linlearn import BinaryClassifier

from plot import (
    get_mesh,
    plot_contour_binary_classif,
    plot_scatter_binary_classif,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)

# n_samples = 1000
n_samples = 200
random_state = 42
h = 0.01
levels = 20

use_aggregation = True
split_pure = True
n_estimators = 100
step = 1.0
dirichlet = 0.5

norm = plt.Normalize(vmin=0.0, vmax=1.0)


def simulate_data(dataset="moons"):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)

        n_samples_outliers = 20
        X_outlier = 0.2 * np.random.randn(n_samples_outliers, 2)
        X_outlier[:, 0] -= 3
        X_outlier[:, 1] += 1
        y_outlier = np.ones(n_samples_outliers)
        X = np.concatenate((X, X_outlier), axis=0)
        y = np.concatenate((y, y_outlier), axis=0)

    elif dataset == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state
        )
    elif dataset == "linear":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=random_state,
            n_clusters_per_class=1,
            flip_y=0.001,
            class_sep=2.0,
        )
        rng = np.random.RandomState(random_state)
        X += 1.5 * rng.uniform(size=X.shape)

        n_samples_outliers = 30
        X_outlier = 0.5 * np.random.randn(n_samples_outliers, 2)
        X_outlier[:, 0] += 20
        X_outlier[:, 1] += 2
        y_outlier = np.zeros(n_samples_outliers)
        X = np.concatenate((X, X_outlier), axis=0)
        y = np.concatenate((y, y_outlier), axis=0)

    else:
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    return X, y


# datasets = [simulate_data("moons"), simulate_data("circles"), simulate_data("linear")]

datasets = [simulate_data("moons"), simulate_data("linear")]

# TODO: polynomial logistic regression
# TODO: different values for the block_size

n_classifiers = 3
n_datasets = 2
_ = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))


def get_classifiers():
    # kwargs = {"tol": 1e-15, "max_iter": 1000, "fit_intercept": False}
    kwargs = {"fit_intercept": True}
    return [
        ("Logistic Regression", LogisticRegression(**kwargs)),
        ("Binary Classifier ERM", BinaryClassifier(**kwargs),),
        ("Binary Classifier MOM", BinaryClassifier(estimator="mom", **kwargs),),
    ]


i = 1

for ds_cnt, ds in enumerate(datasets):
    print("-" * 80)
    X, y = ds
    # y[y == 0] = -1
    # print(y)
    xx, yy, X_mesh = get_mesh(X, h=h, padding=0.2)
    ax = plt.subplot(n_datasets, n_classifiers + 1, i)
    if ds_cnt == 0:
        title = "Input data"
    else:
        title = None

    plot_scatter_binary_classif(ax, xx, yy, X, y, s=20, alpha=0.7, title=title)
    i += 1
    classifiers = get_classifiers()
    for name, clf in classifiers:
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)
        if hasattr(clf, "clear"):
            clf.clear()
        if hasattr(clf, "partial_fit"):
            clf.partial_fit(X, y)
        else:
            clf.fit(X, y)

        print(name)
        print("intercept_: ", clf.intercept_, "coef_: ", clf.coef_)
        Z = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
        if ds_cnt == 0:
            plot_contour_binary_classif(
                ax, xx, yy, Z, levels=levels, title=name, norm=norm
            )
        else:
            plot_contour_binary_classif(ax, xx, yy, Z, levels=levels, norm=norm)
            # plot_contour_binary_classif(ax, xx, yy, Z, levels=levels, norm=None)
        i += 1

plt.tight_layout()
plt.show()

# plt.savefig("decisions.pdf")
# logging.info("Saved the decision functions in 'decision.pdf")
