# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

"""
This module contains unittests for the estimators. Can be run using

    > pytest -v
"""

import pytest

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from linlearn import BinaryClassifier
from .utils import simulate_true_logistic


@pytest.mark.parametrize(
    "estimator, solver",
    [
        ("erm", "cgd"),
        ("mom", "cgd"),
        ("ch", "cgd"),
        ("tmean", "cgd"),
        ("llm", "gd"),
        ("gmom", "gd"),
    ],
)
@pytest.mark.parametrize("fit_intercept", (False, True))
def test_estimators_on_simple_data(estimator, solver, fit_intercept):
    n_samples = 1_000
    n_features = 3
    verbose = False
    random_state = 2
    X, y, coef0, intercept0 = simulate_true_logistic(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        fit_intercept=fit_intercept,
        return_coef=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, stratify=y, random_state=42, test_size=0.25
    )
    kwargs = {
        "estimator": estimator,
        "solver": solver,
        "verbose": verbose,
        "random_state": random_state,
    }
    clf = BinaryClassifier(**kwargs).fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, y_score) >= 0.8
    assert coef0 == pytest.approx(clf.coef_.ravel(), abs=0.5, rel=0.5)
    assert intercept0 == pytest.approx(clf.intercept_, abs=0.5, rel=0.5)
