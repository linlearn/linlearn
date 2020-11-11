from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LRScikit
from linlearn.learner_old import LogisticRegression as LRLinLearn
import numpy as np
from matplotlib.colors import ListedColormap

# from tick.linear_model import LogisticRegression as TickLogisticRegression
# from smp import LogisticRegression
# from tick.linear_model import SimuLogReg
# from tick.linear_model import LogisticRegression

from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score

from scipy.optimize import check_grad

import logging

import matplotlib.pyplot as plt


# from tick.plot import stems

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# np.set_printoptions(precision=2)


def plot_decision_classification(classifiers, datasets):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    h = 0.2
    fig = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # print('=' * 64)
        # preprocess dataset, split into training and test part
        X, y = ds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        # iterate over classifiers
        for name, clf in classifiers:
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
            clf.set(X_train, y_train)
            truc = np.empty((xx.ravel().shape[0], 2))
            truc[:, 0] = xx.ravel()
            truc[:, 1] = yy.ravel()
            # truc = np.array([xx.ravel(), yy.ravel()]).T
            print(truc.flags)
            Z = clf.predict_proba(truc)[:, 1]

            # score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            #         size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()


# Simulation of datasets
n_samples = 200
n_samples_half = int(n_samples / 2)
n_features = 2
n_classes = 2
random_state = 42

X = 0.5 * np.random.randn(n_samples, n_features)

X[:n_samples_half] += np.array([1.0, 2.0])
X[n_samples_half:] += np.array([-2.0, -1.0])

y = np.zeros(n_samples)
y[n_samples_half:] = 1
y[:n_samples_half] = -1

# X, y = SimuLogReg(weights=np.array([1., 1.]), intercept=0., features=X).simulate()

# X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                            n_redundant=0, n_informative=2,
#                            random_state=random_state,
#                            n_clusters_per_class=1)
#
# rng = np.random.RandomState(random_state)
# X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)


# def simu2():
#     n_samples = 10000
#     w0 = np.array([-2., 1.])
#     b0 = 3.
#     n_features = w0.shape[0]
#     cov = toeplitz(0.5 ** np.arange(0, n_features))
#     X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
#     p = LogisticRegression._sigmoid(X.dot(w0) + b0)
#     y = np.random.binomial(1, p, size=n_samples)
#     y[:] = 2 * y - 1
#     return X, y


datasets = [
    # make_moons(n_samples=n_samples, noise=0.1, random_state=0),
    # make_circles(n_samples=n_samples, noise=0.1, factor=0.5,
    #              random_state=random_state),
    linearly_separable,
    # simu2()
]

x0 = np.random.randn(n_features + 1)

# for X, y in datasets:
#     truc = check_grad(
#         func=lambda w: LogisticRegression._loss(w, X, y),
#         x0=x0,
#         grad=lambda w: LogisticRegression._grad(w, X, y),
#     )
#     # print(truc)

# X, y = datasets[-1]
# logreg = LogisticRegression().fit(X, y)

# print(logreg.coefs_)
# print(logreg.intercept_)

# exit(0)


classifiers = [
    # ('LR tick', TickLogisticRegression(penalty='none', tol=0, solver_old='bfgs', max_iter=100)),
    ("LR scikit", LRScikit(C=1e6, solver="lbfgs", max_iter=100)),
    ("LR linlearn", LRLinLearn(C=1e4, max_iter=100, step=1e-2)),
    ("LR linlearn", LRLinLearn(C=1e4, max_iter=100, step=1e-2, smp=True)),
    # ('LR SMP', LogisticRegression(penalty='none', tol=0, max_iter=100, smp=True)),
    # ('LR SMP', LogisticRegression(random_state=123, penalty='none', tol=0, solver_old='svrg', smp=True)),
    # ('MF', MondrianForestClassifier(n_estimators=n_trees)),
    # ('RF', RandomForestClassifier(n_estimators=n_trees)),
    # ('ET', ExtraTreesClassifier(n_estimators=n_trees))
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

plot_decision_classification(classifiers, datasets)

plt.savefig("decisions.pdf")
logging.info("Saved the decision functions in 'decision.pdf")


# for _, clf in classifiers:
#     print(clf.weights)

# for clf_name, clf in classifiers:
#     print('-' * 32)
#     print(clf_name)
#     print(clf.predict_proba(X_test))
#     if hasattr(clf, 'coefs_'):
#         print(clf.coefs_)
#     else:
#         print(clf.weights)

# from tick.linear_model import ModelLogReg


# for X, y in datasets:
#     print('-' * 32)
#     print(np.unique(y))
#     X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=.4, random_state=42)
#
#     ww = np.random.randn(n_features + 1)
#
#     model = ModelLogReg(fit_intercept=True).fit(X_train, y_train)
#
#     print('----')
#     print(model.grad(ww))
#     print(LogisticRegression._grad(ww, X_train, y_train))
#     print('----')
#     print(model.loss(ww))
#     print(LogisticRegression._loss(ww, X_train, y_train))
#     print('----')
#     logreg_tick = TickLogisticRegression(tol=0, max_iter=100, fit_intercept=True,
#                                          verbose=0, solver_old='bfgs', penalty='none')
#     logreg_tick.fit(X_train, y_train)
#     logreg_smp = LogisticRegression(tol=0, max_iter=100, fit_intercept=True)
#     logreg_smp.fit(X_train, y_train)
#
#     print(logreg_tick.weights)
#     print(logreg_smp.coefs_)
#     print('----')
#     print(logreg_tick.intercept)
#     print(logreg_smp.intercept_)
