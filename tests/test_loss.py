# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
import numpy as np
import pytest
from scipy.sparse import csr_matrix, csc_matrix


from linlearn._loss import decision_function_factory


@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("sparse", [False, "csr", "csc"])
def test_decision_function(fit_intercept, sparse):
    n_samples, n_features = 11, 4
    n_weights = n_features + int(fit_intercept)
    X = np.random.randn(n_samples, n_features)
    if sparse == "csr":
        X = csr_matrix(X)
    if sparse == "csc":
        X = csc_matrix(X)
    w = np.random.randn(n_weights)
    if fit_intercept:
        z = X.dot(w[1:]) + w[0]
    else:
        z = X.dot(w)
    out = np.empty(n_samples)
    decision_function = decision_function_factory(X, fit_intercept)
    decision_function(w, out)
    assert z == pytest.approx(out, abs=1e-6)
