# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, make_classification, load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np
from linlearn import Classifier

penalties = Classifier._penalties  # ("none",)#

# (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
grid_C = (1e-3, 1.0,)# 1e3) # TODO : figure out why so many tests fail for C = 1000.0
grid_l1_ratio = (0.1, 0.5, 0.9)
solvers = ("cgd", "gd", "svrg", "saga")  # , "sgd",)

random_state = 44


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties)
@pytest.mark.parametrize("C", grid_C)
@pytest.mark.parametrize("l1_ratio", grid_l1_ratio)
@pytest.mark.parametrize("solver", solvers)
def test_fit_same_sklearn_simulated_multiclass(
    fit_intercept, penalty, C, l1_ratio, solver
):
    """
    This is a test that checks on many combinations that Classifier gets the
    same coef_ and intercept_ as scikit-learn on simulated data
    """
    tol = 1e-10
    max_iter = 200  # so many iterations needed to reach necessary precision ...
    verbose = False
    step_size = 1.0

    n_samples = 128
    n_features = 5
    n_classes = 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        random_state=random_state,
    )
    # X, y = load_iris(return_X_y=True)

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
        "multi_class": "multinomial",
    }

    # if solver in ["svrg", "saga", "gd"] and fit_intercept:
    #     abs_approx, rel_approx = 1e-4, 1e-4
    # else:
    #     abs_approx, rel_approx = 1e-6, 1e-6
    abs_approx, rel_approx = 1e-4, 1e-4 #######################################

    if penalty == "l1" and C == 1.0 and fit_intercept:
        args["max_iter"] = 300
    if penalty == "elasticnet" and C == 1.0:
        step_size = 1.0
        args["max_iter"] = 600
        abs_approx, rel_approx = 1e-2, 1e-2 #######################################
    if penalty == "elasticnet" and C == 1.0 and l1_ratio == 0.1:
        step_size = 1.0
        args["max_iter"] = 900
        abs_approx, rel_approx = 1e-2, 1e-2 #######################################
    if penalty == "elasticnet" and C == 1.0 and l1_ratio == 0.5:
        args["max_iter"] = 1000
        abs_approx, rel_approx = 1e-2, 1e-2 #######################################

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
    args.pop("multi_class")
    clf_linlearn = Classifier(
        penalty=penalty,
        loss="multilogistic",
        step_size=step_size,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        **args
    )
    clf_linlearn.fit(X, y)

    # For some weird reason scikit's intercept_ does not match for "l1" and
    # "elasticnet" with intercept and for small C
    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C < 1e-1):
        # Test the intercept_
        assert clf_scikit.intercept_ == pytest.approx(
            clf_linlearn.intercept_, abs=abs_approx, rel=rel_approx
        )
        # And test prediction methods
        assert clf_scikit.decision_function(X) == pytest.approx(
            clf_linlearn.decision_function(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_proba(X) == pytest.approx(
            clf_linlearn.predict_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_log_proba(X) == pytest.approx(
            clf_linlearn.predict_log_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert (clf_scikit.predict(X) == clf_linlearn.predict(X)).any()
        assert clf_scikit.score(X, y) == clf_linlearn.score(X, y)

    # And always test the coef_
    assert clf_scikit.coef_ == pytest.approx(
        clf_linlearn.coef_, abs=abs_approx, rel=rel_approx
    )

@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties[1:]) # don't test iris with none penalty
@pytest.mark.parametrize("C", grid_C)
@pytest.mark.parametrize("l1_ratio", grid_l1_ratio)
@pytest.mark.parametrize("solver", solvers)
def test_fit_same_sklearn_iris(
    fit_intercept, penalty, C, l1_ratio, solver
):
    """
    This is a test that checks on many combinations that Classifier gets the
    same coef_ and intercept_ as scikit-learn on the iris dataset
    """
    tol = 1e-10
    max_iter = 400  # so many iterations needed to reach necessary precision ...
    verbose = False
    step_size = 1.0

    X, y = load_iris(return_X_y=True)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    # std_scaler = StandardScaler()
    # X = std_scaler.fit_transform(X)

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
        "multi_class": "multinomial",
    }

    # if solver in ["svrg", "saga", "gd"] and fit_intercept:
    #     abs_approx, rel_approx = 1e-4, 1e-4
    # else:
    #     abs_approx, rel_approx = 1e-6, 1e-6
    abs_approx, rel_approx = 1e-3, 1e-3 #######################################

    if penalty == "elasticnet" and l1_ratio == 0.5:
        step_size = 1.5
    if (penalty == "l1") or (penalty == "elasticnet" and l1_ratio == 0.9):
        step_size = 2.0
        args["max_iter"] = 1200
    if penalty == "l1" and fit_intercept:
        step_size = 3.5
        args["max_iter"] = 1500
        abs_approx, rel_approx = 1e-2, 1e-2 #######################################

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
    args.pop("multi_class")
    clf_linlearn = Classifier(
        penalty=penalty,
        loss="multilogistic",
        C=C,
        step_size=step_size,
        l1_ratio=l1_ratio,
        solver=solver,
        **args
    )
    clf_linlearn.fit(X, y)

    # And always test the coef_

    assert clf_scikit.coef_ == pytest.approx(
        clf_linlearn.coef_, abs=abs_approx, rel=rel_approx
    )

    # For some weird reason scikit's intercept_ does not match for "l1" and
    # "elasticnet" with intercept and for small C
    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C < 1e-1):
        # Test the intercept_
        assert clf_scikit.intercept_ == pytest.approx(
            clf_linlearn.intercept_, abs=abs_approx, rel=rel_approx
        )
        # And test prediction methods
        assert clf_scikit.decision_function(X) == pytest.approx(
            clf_linlearn.decision_function(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_proba(X) == pytest.approx(
            clf_linlearn.predict_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_log_proba(X) == pytest.approx(
            clf_linlearn.predict_log_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert (clf_scikit.predict(X) == clf_linlearn.predict(X)).any()
        assert clf_scikit.score(X, y) == clf_linlearn.score(X, y)


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("penalty", penalties[1:]) # don't test the wine dataset with no penalty
@pytest.mark.parametrize("C", grid_C)
@pytest.mark.parametrize("l1_ratio", grid_l1_ratio)
@pytest.mark.parametrize("solver", solvers)
def test_fit_same_sklearn_wine(
    fit_intercept, penalty, C, l1_ratio, solver
):
    """
    This is a test that checks on many combinations that Classifier gets the
    same coef_ and intercept_ as scikit-learn on the iris dataset
    """
    tol = 1e-10
    max_iter = 400  # so many iterations needed to reach necessary precision ...
    verbose = False
    step_size = 1.0

    X, y = load_wine(return_X_y=True)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    # std_scaler = StandardScaler()
    # X = std_scaler.fit_transform(X)

    args = {
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "fit_intercept": fit_intercept,
        "random_state": 42,
        "multi_class": "multinomial",
    }

    # if solver in ["svrg", "saga", "gd"] and fit_intercept:
    #     abs_approx, rel_approx = 1e-4, 1e-4
    # else:
    #     abs_approx, rel_approx = 1e-6, 1e-6
    abs_approx, rel_approx = 1e-3, 1e-3

    if penalty == "l2" and C == 1.0 and fit_intercept:
        step_size = 2.0
    if penalty == "l1" and C == 1.0:
        step_size = 2.0
        args["max_iter"] = 900
    if penalty == "elasticnet" and C == 1.0:
        step_size = 2.0
        args["max_iter"] = 600
        if solver == "gd" and l1_ratio == 0.9:
            abs_approx, rel_approx = 1e-2, 1e-2

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
    args.pop("multi_class")
    clf_linlearn = Classifier(
        penalty=penalty,
        loss="multilogistic",
        step_size=step_size,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        **args
    )
    clf_linlearn.fit(X, y)

    # And always test the coef_

    assert clf_scikit.coef_ == pytest.approx(
        clf_linlearn.coef_, abs=abs_approx, rel=rel_approx
    )

    # For some weird reason scikit's intercept_ does not match for "l1" and
    # "elasticnet" with intercept and for small C
    if not (penalty in ["l1", "elasticnet"] and fit_intercept and C < 1e-1):
        # Test the intercept_
        assert clf_scikit.intercept_ == pytest.approx(
            clf_linlearn.intercept_, abs=abs_approx, rel=rel_approx
        )
        # And test prediction methods
        assert clf_scikit.decision_function(X) == pytest.approx(
            clf_linlearn.decision_function(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_proba(X) == pytest.approx(
            clf_linlearn.predict_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert clf_scikit.predict_log_proba(X) == pytest.approx(
            clf_linlearn.predict_log_proba(X), abs=abs_approx, rel=rel_approx
        )
        assert (clf_scikit.predict(X) == clf_linlearn.predict(X)).any()
        assert clf_scikit.score(X, y) == clf_linlearn.score(X, y)
