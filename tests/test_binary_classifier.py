# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz
from scipy.special import expit

import pytest

# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles, make_classification

# from . import parameter_test_with_min, parameter_test_with_type, approx

import numbers
from linlearn import BinaryClassifier
from scipy.special import expit, logit

# TODO: va falloir furieusement tester plein de types de données d'entrée, avec labels
#  non-contigus, et avec labels strings par exemple.

# TODO: test the __repr__ (even if it's the one from sklearn


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


def test_block_size():
    clf = BinaryClassifier()
    assert isinstance(clf.block_size, float)
    assert clf.block_size == 0.07

    clf.block_size = 0.123
    assert isinstance(clf.block_size, float)
    assert clf.block_size == 0.123

    for block_size in [-1, complex(1.0, 1.0), "1.0", 0.0, -1.0, 1.1]:
        with pytest.raises(ValueError) as exc_info:
            clf.block_size = block_size
        assert exc_info.type is ValueError
        match = "block_size must be in (0, 1]; got (block_size=%r)" % block_size
        assert exc_info.value.args[0] == match

    for block_size in [-1, complex(1.0, 1.0), "1.0", 0.0, -1.0, 1.1]:
        with pytest.raises(ValueError) as exc_info:
            _ = BinaryClassifier(block_size=block_size)
        assert exc_info.type is ValueError
        match = "block_size must be in (0, 1]; got (block_size=%r)" % block_size
        assert exc_info.value.args[0] == match

    setattr(clf, "block_size", 0.42)
    assert getattr(clf, "block_size") == 0.42


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


def test_l1_ratio():
    clf = BinaryClassifier()
    assert isinstance(clf.l1_ratio, float)
    assert clf.l1_ratio == 0.5

    clf.l1_ratio = 0.123
    assert isinstance(clf.l1_ratio, float)
    assert clf.l1_ratio == 0.123

    clf.l1_ratio = 0.0
    assert isinstance(clf.l1_ratio, float)
    assert clf.l1_ratio == 0.0

    clf.l1_ratio = 1.0
    assert isinstance(clf.l1_ratio, float)
    assert clf.l1_ratio == 1.0

    for l1_ratio in [-1, complex(1.0, 1.0), "1.0", -1.0, 1.1]:
        with pytest.raises(ValueError) as exc_info:
            clf.l1_ratio = l1_ratio
        assert exc_info.type is ValueError
        match = "l1_ratio must be in (0, 1]; got (l1_ratio=%r)" % l1_ratio
        assert exc_info.value.args[0] == match

    for l1_ratio in [-1, complex(1.0, 1.0), "1.0", -1.0, 1.1]:
        with pytest.raises(ValueError) as exc_info:
            _ = BinaryClassifier(l1_ratio=l1_ratio)
        assert exc_info.type is ValueError
        match = "l1_ratio must be in (0, 1]; got (l1_ratio=%r)" % l1_ratio
        assert exc_info.value.args[0] == match

    setattr(clf, "l1_ratio", 0.42)
    assert getattr(clf, "l1_ratio") == 0.42


def simulate_true_logistic(n_samples=150, n_features=5, fit_intercept=True, corr=0.5):
    rng = np.random.RandomState(42)
    coef0 = rng.randn(n_features)
    if fit_intercept:
        intercept0 = -2.0
    else:
        intercept0 = 0.0

    cov = toeplitz(corr ** np.arange(0, n_features))
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    logits = X.dot(coef0)
    logits += intercept0
    p = expit(logits)
    y = rng.binomial(1, p, size=n_samples)
    return X, y


# Turns out that random_state=1 is linearly separable while it is not for
# random_state=2
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


penalties = BinaryClassifier._penalties


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties)
@pytest.mark.parametrize("C", (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3))
@pytest.mark.parametrize("l1_ratio", (0.1, 0.5, 0.9))
def test_fit_same_sklearn_logistic(fit_intercept, penalty, C, l1_ratio):
    """
    This is a test that checks on many combinations that BinaryClassifier gets the
    same coef_ and intercept_ as scikit-learn on simulated data
    """
    n_samples = 128
    n_features = 5
    tol = 1e-10
    max_iter = 200
    verbose = False

    X, y = simulate_true_logistic(
        n_samples=n_samples, n_features=n_features, fit_intercept=fit_intercept,
    )

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
    }

    if penalty == "none":
        # A single test is required for penalty="none"
        if C != 1.0 or l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, solver="saga", **args)
    elif penalty == "l2":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "l1":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "elasticnet":
        clf_scikit = LogisticRegression(
            penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio, **args
        )
    else:
        raise ValueError("Weird penalty %r" % penalty)

    clf_scikit.fit(X, y)
    # We compare with saga since it supports all penalties
    # clf_scikit = LogisticRegression(solver="saga", **args).fit(X, y)
    clf_linlearn = BinaryClassifier(
        penalty=penalty, C=C, l1_ratio=l1_ratio, solver="cgd", **args
    )
    clf_linlearn.fit(X, y)

    # For some weird reason scikit's intercept_ does not match for "l1" and
    # "elasticnet" with intercept and for small C
    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C < 1e-1):
        # Test the intercept_
        assert clf_scikit.intercept_ == pytest.approx(clf_linlearn.intercept_, abs=1e-7)
        # And test prediction methods
        assert clf_scikit.decision_function(X) == pytest.approx(
            clf_linlearn.decision_function(X), abs=1e-6
        )
        assert clf_scikit.predict_proba(X) == pytest.approx(
            clf_linlearn.predict_proba(X), abs=1e-6
        )
        assert clf_scikit.predict_log_proba(X) == pytest.approx(
            clf_linlearn.predict_log_proba(X), abs=1e-6
        )
        assert (clf_scikit.predict(X) == clf_linlearn.predict(X)).any()
        assert clf_scikit.score(X, y) == clf_linlearn.score(X, y)

    # And always test the coef_
    assert clf_scikit.coef_ == pytest.approx(clf_linlearn.coef_, abs=1e-7)


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties)
@pytest.mark.parametrize("C", (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3))
@pytest.mark.parametrize("l1_ratio", (0.1, 0.5, 0.9))
def test_fit_same_sklearn_moons(fit_intercept, penalty, C, l1_ratio):
    """
    This is a test that checks on many combinations that BinaryClassifier gets the
    same coef_ and intercept_ as scikit-learn on simulated data
    """
    n_samples = 150
    tol = 1e-15
    max_iter = 200
    verbose = False
    random_state = 42

    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
    }

    if penalty == "none":
        if C != 1.0 or l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, solver="saga", **args)
    elif penalty == "l2":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "l1":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "elasticnet":
        clf_scikit = LogisticRegression(
            penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio, **args
        )
    else:
        raise ValueError("Weird penalty %r" % penalty)

    clf_scikit.fit(X, y)
    clf_linlearn = BinaryClassifier(
        penalty=penalty, C=C, l1_ratio=l1_ratio, solver="cgd", **args
    )
    clf_linlearn.fit(X, y)

    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C < 1e-1):
        assert clf_scikit.intercept_ == pytest.approx(clf_linlearn.intercept_, abs=1e-4)

    if not (penalty in ["l1", "elasticnet"] and C == 1e-1 and not fit_intercept):
        assert clf_scikit.coef_ == pytest.approx(clf_linlearn.coef_, abs=1e-4)


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties)
@pytest.mark.parametrize("C", (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3))
@pytest.mark.parametrize("l1_ratio", (0.1, 0.5, 0.9))
def test_fit_same_sklearn_circles(fit_intercept, penalty, C, l1_ratio):
    """
    This is a test that checks on many combinations that BinaryClassifier gets the
    same coef_ and intercept_ as scikit-learn on simulated data
    """
    n_samples = 150
    tol = 1e-15
    max_iter = 200
    verbose = False
    random_state = 42

    X, y = make_circles(n_samples=n_samples, noise=0.2, random_state=random_state)

    def approx(v):
        return pytest.approx(v, abs=1e-4)

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
    }

    if penalty == "none":
        if C != 1.0 or l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, solver="saga", **args)
    elif penalty == "l2":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "l1":
        if l1_ratio != 0.5:
            return
        clf_scikit = LogisticRegression(penalty=penalty, C=C, solver="saga", **args)
    elif penalty == "elasticnet":
        clf_scikit = LogisticRegression(
            penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio, **args
        )
    else:
        raise ValueError("Weird penalty %r" % penalty)

    clf_scikit.fit(X, y)
    clf_linlearn = BinaryClassifier(
        penalty=penalty, C=C, l1_ratio=l1_ratio, solver="cgd", **args
    )
    clf_linlearn.fit(X, y)

    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C <= 1e-1):
        assert clf_scikit.intercept_ == approx(clf_linlearn.intercept_)

    assert clf_scikit.coef_ == approx(clf_linlearn.coef_)


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("C", (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3))
def test_elasticnet_l1_ridge_are_consistent(fit_intercept, C):
    n_samples = 128
    n_features = 5
    tol = 1e-10
    max_iter = 200
    verbose = False

    X, y = simulate_true_logistic(
        n_samples=n_samples, n_features=n_features, fit_intercept=fit_intercept,
    )

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
    }

    def approx(v):
        return pytest.approx(v, abs=1e-7)

    # Test that elasticnet with l1_ratio=0.0 is the same as penalty="l2"
    clf_elasticnet = BinaryClassifier(
        penalty="elasticnet", C=C, l1_ratio=0.0, solver="cgd", **args
    )
    clf_l2 = BinaryClassifier(penalty="l2", C=C, solver="cgd", **args)
    clf_elasticnet.fit(X, y)
    clf_l2.fit(X, y)
    assert clf_elasticnet.intercept_ == approx(clf_l2.intercept_)
    assert clf_elasticnet.coef_ == approx(clf_l2.coef_)

    # Test that elasticnet with l1_ratio=1.0 is the same as penalty="l1"
    clf_elasticnet = BinaryClassifier(
        penalty="elasticnet", C=C, l1_ratio=1.0, solver="cgd", **args
    )
    clf_l1 = BinaryClassifier(penalty="l1", C=C, l1_ratio=0.0, solver="cgd", **args)
    clf_elasticnet.fit(X, y)
    clf_l1.fit(X, y)
    assert clf_elasticnet.intercept_ == approx(clf_l1.intercept_)
    assert clf_elasticnet.coef_ == approx(clf_l1.coef_)


def test_that_array_conversion_is_ok():
    import pandas as pd

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
    assert (lr.predict(X) == br.predict(X)).any()


# TODO: test "mom" strategy works best with outlying data


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
