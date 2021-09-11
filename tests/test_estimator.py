# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
import numpy as np
import pytest
from scipy.sparse import csr_matrix, csc_matrix

from linlearn._loss import Logistic
from linlearn._estimator import ERM
from linlearn._utils import np_float


@pytest.mark.parametrize("Estimator", (ERM,))
@pytest.mark.parametrize("fit_intercept", (False, True))
@pytest.mark.parametrize("sparsify", ("no", "random", "row", "col"))
def test_sparse_matches_dense(Estimator, fit_intercept, sparsify):
    n_samples, n_features = 11, 4
    n_weights = n_features + int(fit_intercept)
    X_dense = np.random.randn(n_samples, n_features)
    # We try out different sparsity patterns just to be sure that CSR and CSC code
    # really works
    if sparsify == "random":
        X_dense[X_dense < 0.0] = 0.0
    elif sparsify == "row":
        X_dense[::2, :] = 0.0
    elif sparsify == "col":
        X_dense[:, ::2] = 0.0
    X_csr = csr_matrix(X_dense)
    X_csc = csc_matrix(X_dense)
    y = np.ones(n_samples, dtype=np_float)
    y[: (n_samples // 2)] *= -1
    np.random.shuffle(y)
    loss = Logistic()
    estimator_dense = Estimator(X_dense, y, loss, fit_intercept)
    estimator_csr = Estimator(X_csr, y, loss, fit_intercept)
    estimator_csc = Estimator(X_csc, y, loss, fit_intercept)
    w = np.random.randn(n_weights)

    state_dense = estimator_dense.get_state()
    state_csc = estimator_csc.get_state()
    state_csr = estimator_csr.get_state()

    partial_deriv_dense = estimator_dense.partial_deriv_factory()
    with pytest.warns(UserWarning):
        partial_deriv_csr = estimator_csr.partial_deriv_factory()
    # TODO: check that the warning message is correct
    partial_deriv_csc = estimator_csc.partial_deriv_factory()

    inner_products = np.random.randn(n_samples)
    for j in range(n_weights):
        deriv_dense = partial_deriv_dense(j, inner_products, state_dense)
        deriv_csc = partial_deriv_csc(j, inner_products, state_csc)
        assert deriv_dense == pytest.approx(deriv_csc, abs=1e-10)
        deriv_csr = partial_deriv_csr(j, inner_products, state_csr)
        assert deriv_dense == pytest.approx(deriv_csr, abs=1e-10)
