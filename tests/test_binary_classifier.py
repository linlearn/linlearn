# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
import pytest

# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

# from . import parameter_test_with_min, parameter_test_with_type, approx

import numbers
from linlearn import BinaryClassifier
import pytest


# TODO: parameter_test_with_type does nothing !!!

# TODO: va falloir furieusement tester plein de types de données d'entrée, avec labels
#  non-contigus, et avec labels strings par exemple.

# TODO: verifier que fit avec strategy="erm" amène exactement au meme coef_ et intercept_ que sklearn

# TODO: test the __repr__ (even if it's the one from sklearn


# class TestBinaryClassifierProperties(object):
def test_keyword_args_only():
    with pytest.raises(TypeError) as exc_info:
        _ = BinaryClassifier("l2")
    assert exc_info.type is TypeError
    match = "__init__() takes 1 positional argument but 2 were given"
    assert exc_info.value.args[0] == match


def test_penalty():
    clf = BinaryClassifier()
    assert clf.penalty == "l2"

    for penalty in BinaryClassifier._penalties:
        clf.penalty = penalty
        assert clf.penalty == penalty

    penalty = "stuff"
    with pytest.raises(ValueError) as exc_info:
        clf.penalty = penalty
    assert exc_info.type is ValueError
    match = "penalty must be one of %r; got (penalty=%r)" % (
        BinaryClassifier._penalties,
        penalty,
    )
    assert exc_info.value.args[0] == match

    penalty = "stuff"
    with pytest.raises(ValueError) as exc_info:
        _ = BinaryClassifier(penalty=penalty)
    assert exc_info.type is ValueError
    match = "penalty must be one of %r; got (penalty=%r)" % (
        BinaryClassifier._penalties,
        penalty,
    )
    assert exc_info.value.args[0] == match

    setattr(clf, "penalty", "l1")
    assert getattr(clf, "penalty") == "l1"


def test_C():
    clf = BinaryClassifier()
    assert isinstance(clf.C, float)
    assert clf.C == 1.0

    clf.C = 42e1
    assert isinstance(clf.C, float)
    assert clf.C == 420.0

    clf.C = 0
    assert isinstance(clf.C, float)
    assert clf.C == 0.0

    for C in [-1, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            clf.C = C
        assert exc_info.type is ValueError
        match = "C must be a positive number; got (C=%r)" % C
        assert exc_info.value.args[0] == match

    for C in [-1, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            BinaryClassifier(C=C)
        assert exc_info.type is ValueError
        match = "C must be a positive number; got (C=%r)" % C
        assert exc_info.value.args[0] == match

    setattr(clf, "C", 3.140)
    assert getattr(clf, "C") == 3.14


def test_loss():
    clf = BinaryClassifier()
    assert clf.loss == "logistic"

    for loss in BinaryClassifier._losses:
        clf.loss = loss
        assert clf.loss == loss

    loss = "stuff"
    with pytest.raises(ValueError) as exc_info:
        clf.loss = loss
    assert exc_info.type is ValueError
    match = "loss must be one of %r; got (loss=%r)" % (BinaryClassifier._losses, loss,)
    assert exc_info.value.args[0] == match

    loss = "stuff"
    with pytest.raises(ValueError) as exc_info:
        _ = BinaryClassifier(loss=loss)
    assert exc_info.type is ValueError
    match = "loss must be one of %r; got (loss=%r)" % (BinaryClassifier._losses, loss,)
    assert exc_info.value.args[0] == match

    setattr(clf, "loss", "logistic")
    assert getattr(clf, "loss") == "logistic"


def test_fit_intercept():
    clf = BinaryClassifier()
    assert isinstance(clf.fit_intercept, bool)
    assert clf.fit_intercept is True

    clf.fit_intercept = False
    assert isinstance(clf.fit_intercept, bool)
    assert clf.fit_intercept is False

    for fit_intercept in [0, 1, -1, complex(1.0, 1.0), "1.0", "true"]:
        with pytest.raises(ValueError) as exc_info:
            clf.fit_intercept = fit_intercept
        assert exc_info.type is ValueError
        match = "fit_intercept must be True or False; got (C=%r)" % fit_intercept
        assert exc_info.value.args[0] == match

    for fit_intercept in [0, 1, -1, complex(1.0, 1.0), "1.0", "true"]:
        with pytest.raises(ValueError) as exc_info:
            BinaryClassifier(fit_intercept=fit_intercept)
        assert exc_info.type is ValueError
        match = "fit_intercept must be True or False; got (C=%r)" % fit_intercept
        assert exc_info.value.args[0] == match

    setattr(clf, "fit_intercept", True)
    assert getattr(clf, "fit_intercept") is True


def test_strategy():
    clf = BinaryClassifier()
    assert clf.strategy == "erm"

    for strategy in BinaryClassifier._strategies:
        clf.strategy = strategy
        assert clf.strategy == strategy

    strategy = "stuff"
    with pytest.raises(ValueError) as exc_info:
        clf.strategy = strategy
    assert exc_info.type is ValueError
    match = "strategy must be one of %r; got (strategy=%r)" % (
        BinaryClassifier._strategies,
        strategy,
    )
    assert exc_info.value.args[0] == match

    strategy = "stuff"
    with pytest.raises(ValueError) as exc_info:
        _ = BinaryClassifier(strategy=strategy)
    assert exc_info.type is ValueError
    match = "strategy must be one of %r; got (strategy=%r)" % (
        BinaryClassifier._strategies,
        strategy,
    )
    assert exc_info.value.args[0] == match

    setattr(clf, "strategy", "mom")
    assert getattr(clf, "strategy") == "mom"


def test_solver():
    clf = BinaryClassifier()
    assert clf.solver == "cgd"

    for solver in BinaryClassifier._solvers:
        clf.solver = solver
        assert clf.solver == solver

    solver = "stuff"
    with pytest.raises(ValueError) as exc_info:
        clf.solver = solver
    assert exc_info.type is ValueError
    match = "solver must be one of %r; got (solver=%r)" % (
        BinaryClassifier._solvers,
        solver,
    )
    assert exc_info.value.args[0] == match

    solver = "stuff"
    with pytest.raises(ValueError) as exc_info:
        _ = BinaryClassifier(solver=solver)
    assert exc_info.type is ValueError
    match = "solver must be one of %r; got (solver=%r)" % (
        BinaryClassifier._solvers,
        solver,
    )
    assert exc_info.value.args[0] == match

    setattr(clf, "solver", "cgd")
    assert getattr(clf, "solver") == "cgd"


def test_tol():
    clf = BinaryClassifier()
    assert isinstance(clf.tol, float)
    assert clf.tol == 1e-4

    clf.tol = 3.14e-3
    assert isinstance(clf.tol, float)
    assert clf.tol == 3.14e-3

    for tol in [-1, 0.0, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            clf.tol = tol
        assert exc_info.type is ValueError
        match = "Tolerance for stopping criteria must be positive; got (tol=%r)" % tol
        assert exc_info.value.args[0] == match

    for tol in [-1, 0.0, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            BinaryClassifier(tol=tol)
        assert exc_info.type is ValueError
        match = "Tolerance for stopping criteria must be positive; got (tol=%r)" % tol
        assert exc_info.value.args[0] == match

    setattr(clf, "tol", 3.14)
    assert getattr(clf, "tol") == 3.14


def test_max_iter():
    clf = BinaryClassifier()
    assert isinstance(clf.max_iter, int)
    assert clf.max_iter == 100

    clf.max_iter = 42.0
    assert isinstance(clf.max_iter, int)
    assert clf.max_iter == 42

    for max_iter in [-1, 0, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            clf.max_iter = max_iter
        assert exc_info.type is ValueError
        match = (
            "Maximum number of iteration must be positive; got (max_iter=%r)" % max_iter
        )
        assert exc_info.value.args[0] == match

    for max_iter in [-1, 0.0, complex(1.0, 1.0), "1.0"]:
        with pytest.raises(ValueError) as exc_info:
            BinaryClassifier(max_iter=max_iter)
        assert exc_info.type is ValueError
        match = (
            "Maximum number of iteration must be positive; got (max_iter=%r)" % max_iter
        )
        assert exc_info.value.args[0] == match

    setattr(clf, "max_iter", 123)
    assert getattr(clf, "max_iter") == 123


# ValueError()

# ValueError()

# def test_n_classes(self):
#     parameter_test_with_min(
#         BinaryClassifier,
#         parameter="n_classes",
#         valid_val=3,
#         invalid_type_val=2.0,
#         invalid_val=1,
#         min_value=2,
#         min_value_str="2",
#         mandatory=True,
#         fixed_type=int,
#         required_args={"n_classes": 2},
#     )

# def test_n_features(self):
#     clf = AMFClassifier(n_classes=2)
#     X = np.random.randn(2, 2)
#     y = np.array([0.0, 1.0])
#     clf.partial_fit(X, y)
#     assert clf.n_features == 2
#     with pytest.raises(ValueError, match="`n_features` is a readonly attribute"):
#         clf.n_features = 3
#
# def test_n_estimators(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="n_estimators",
#         valid_val=3,
#         invalid_type_val=2.0,
#         invalid_val=0,
#         min_value=1,
#         min_value_str="1",
#         mandatory=False,
#         fixed_type=int,
#         required_args={"n_classes": 2},
#     )
#
# def test_step(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="step",
#         valid_val=2.0,
#         invalid_type_val=0,
#         invalid_val=0.0,
#         min_value_strict=0.0,
#         min_value_str="0",
#         mandatory=False,
#         fixed_type=float,
#         required_args={"n_classes": 2},
#     )
#
# def test_loss(self):
#     amf = AMFClassifier(n_classes=2)
#     assert amf.loss == "log"
#     amf.loss = "other loss"
#     assert amf.loss == "log"
#
# def test_use_aggregation(self):
#     parameter_test_with_type(
#         AMFClassifier,
#         parameter="step",
#         valid_val=False,
#         invalid_type_val=0,
#         mandatory=False,
#         fixed_type=bool,
#     )
#
# def test_dirichlet(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="dirichlet",
#         valid_val=0.1,
#         invalid_type_val=0,
#         invalid_val=0.0,
#         min_value_strict=0.0,
#         min_value_str="0",
#         mandatory=False,
#         fixed_type=float,
#         required_args={"n_classes": 2},
#     )
#
# def test_split_pure(self):
#     parameter_test_with_type(
#         AMFClassifier,
#         parameter="split_pure",
#         valid_val=False,
#         invalid_type_val=0,
#         mandatory=False,
#         fixed_type=bool,
#     )
#
# def test_random_state(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="random_state",
#         valid_val=4,
#         invalid_type_val=2.0,
#         invalid_val=-1,
#         min_value=0,
#         min_value_str="0",
#         mandatory=False,
#         fixed_type=int,
#         required_args={"n_classes": 2},
#     )
#     amf = AMFClassifier(n_classes=2)
#     assert amf.random_state is None
#     assert amf._random_state == -1
#     amf.random_state = 1
#     amf.random_state = None
#     assert amf._random_state == -1
#
# def test_n_jobs(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="n_jobs",
#         valid_val=4,
#         invalid_type_val=2.0,
#         invalid_val=0,
#         min_value=1,
#         min_value_str="1",
#         mandatory=False,
#         fixed_type=int,
#         required_args={"n_classes": 2},
#     )
#
# def test_n_samples_increment(self):
#     parameter_test_with_min(
#         AMFClassifier,
#         parameter="n_samples_increment",
#         valid_val=128,
#         invalid_type_val=2.0,
#         invalid_val=0,
#         min_value=1,
#         min_value_str="1",
#         mandatory=False,
#         fixed_type=int,
#         required_args={"n_classes": 2},
#     )
#
# def test_verbose(self):
#     parameter_test_with_type(
#         AMFClassifier,
#         parameter="verbose",
#         valid_val=False,
#         invalid_type_val=0,
#         mandatory=False,
#         fixed_type=bool,
#     )
#
# def test_repr(self):
#     amf = AMFClassifier(n_classes=3)
#     assert (
#         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=10, "
#         "step=1.0, loss='log', use_aggregation=True, "
#         "dirichlet=0.01, split_pure=False, n_jobs=1, "
#         "random_state=None, verbose=False)"
#     )
#
#     amf.n_estimators = 42
#     assert (
#         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, "
#         "step=1.0, loss='log', use_aggregation=True, "
#         "dirichlet=0.01, split_pure=False, n_jobs=1, "
#         "random_state=None, verbose=False)"
#     )
#
#     amf.verbose = False
#     assert (
#         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, "
#         "step=1.0, loss='log', use_aggregation=True, "
#         "dirichlet=0.01, split_pure=False, n_jobs=1, "
#         "random_state=None, verbose=False)"
#     )
#
# def test_partial_fit(self):
#     clf = AMFClassifier(n_classes=2)
#     n_features = 4
#     X = np.random.randn(2, n_features)
#     y = np.array([0.0, 1.0])
#     clf.partial_fit(X, y)
#     assert clf.n_features == n_features
#     assert clf.no_python.iteration == 2
#     assert clf.no_python.samples.n_samples == 2
#     assert clf.no_python.n_features == n_features
#
#     with pytest.raises(ValueError) as exc_info:
#         X = np.random.randn(2, 3)
#         y = np.array([0.0, 1.0])
#         clf.partial_fit(X, y)
#     assert exc_info.type is ValueError
#     assert (
#         exc_info.value.args[0] == "`partial_fit` was first called with "
#         "n_features=4 while n_features=3 in this call"
#     )
#
#     with pytest.raises(
#         ValueError, match="All the values in `y` must be non-negative",
#     ):
#         clf = AMFClassifier(n_classes=2)
#         X = np.random.randn(2, n_features)
#         y = np.array([0.0, -1.0])
#         clf.partial_fit(X, y)
#
#     with pytest.raises(ValueError) as exc_info:
#         clf = AMFClassifier(n_classes=2)
#         X = np.random.randn(2, 3)
#         y = np.array([0.0, 2.0])
#         clf.partial_fit(X, y)
#     assert exc_info.type is ValueError
#     assert exc_info.value.args[0] == "n_classes=2 while y.max()=2"
#
# def test_predict_proba(self):
#     clf = AMFClassifier(n_classes=2)
#     with pytest.raises(
#         RuntimeError,
#         match="You must call `partial_fit` before asking for predictions",
#     ):
#         X_test = np.random.randn(2, 3)
#         clf.predict_proba(X_test)
#
#     with pytest.raises(ValueError) as exc_info:
#         X = np.random.randn(2, 2)
#         y = np.array([0.0, 1.0])
#         clf.partial_fit(X, y)
#         X_test = np.random.randn(2, 3)
#         clf.predict_proba(X_test)
#     assert exc_info.type is ValueError
#     assert exc_info.value.args[
#         0
#     ] == "`partial_fit` was called with n_features=%d while predictions are asked with n_features=%d" % (
#         clf.n_features,
#         3,
#     )
#
# def test_performance_on_moons(self):
#     n_samples = 300
#     random_state = 42
#     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, random_state=random_state
#     )
#     clf = AMFClassifier(n_classes=2, random_state=random_state)
#     clf.partial_fit(X_train, y_train)
#     y_pred = clf.predict_proba(X_test)
#     score = roc_auc_score(y_test, y_pred[:, 1])
#     # With this random_state, the score should be exactly 0.9709821428571429
#     assert score > 0.97
#
# def test_predict_proba_tree_match_predict_proba(self):
#     n_samples = 300
#     n_classes = 2
#     n_estimators = 10
#     random_state = 42
#     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, random_state=random_state
#     )
#     clf = AMFClassifier(
#         n_classes=2, n_estimators=n_estimators, random_state=random_state
#     )
#     clf.partial_fit(X_train, y_train)
#     y_pred = clf.predict_proba(X_test)
#     y_pred_tree = np.empty((y_pred.shape[0], n_classes, n_estimators))
#     for idx_tree in range(n_estimators):
#         y_pred_tree[:, :, idx_tree] = clf.predict_proba_tree(X_test, idx_tree)
#
#     assert y_pred == approx(y_pred_tree.mean(axis=2), 1e-6)
#
# def test_random_state_is_consistant(self):
#     n_samples = 300
#     random_state = 42
#     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, random_state=random_state
#     )
#
#     clf = AMFClassifier(n_classes=2, random_state=random_state)
#     clf.partial_fit(X_train, y_train)
#     y_pred_1 = clf.predict_proba(X_test)
#
#     clf = AMFClassifier(n_classes=2, random_state=random_state)
#     clf.partial_fit(X_train, y_train)
#     y_pred_2 = clf.predict_proba(X_test)
#
#     assert y_pred_1 == approx(y_pred_2)
