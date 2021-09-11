# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
import numpy as np
import pytest
from scipy.sparse import csr_matrix, csc_matrix

from linlearn._solver._learning_rate import (
    learning_rates_erm_f_factory,
    learning_rates_erm_c_factory,
    learning_rates_erm_csc_factory,
    learning_rates_erm_csr_factory,
    learning_rates_mom_f_factory,
    learning_rates_ch_f_factory,
    learning_rates_tmean_f_factory,
)


@pytest.mark.parametrize("sparsify", ("no", "random", "row"))
@pytest.mark.parametrize("fit_intercept", (False, True))
def test_learning_rates_erm_dense_match_sparse(sparsify, fit_intercept):
    n_samples, n_features = 11, 4
    n_weights = n_features + int(fit_intercept)
    lip_const = 0.25
    X = np.random.randn(n_samples, n_features)
    if sparsify == "random":
        X[X < 0.0] = 0.0
    elif sparsify == "row":
        X[::2, :] = 0.0

    X_c = np.ascontiguousarray(X)
    X_f = np.asfortranarray(X)
    X_csc = csc_matrix(X)
    X_csr = csr_matrix(X)

    learning_rates_c = learning_rates_erm_c_factory(X_c, fit_intercept)
    learning_rates_f = learning_rates_erm_f_factory(X_f, fit_intercept)
    learning_rates_csc = learning_rates_erm_csc_factory(X_csc, fit_intercept)
    learning_rates_csr = learning_rates_erm_csr_factory(X_csr, fit_intercept)

    steps = np.empty(n_weights)
    if fit_intercept:
        steps[0] = 1 / lip_const
        steps[1:] = n_samples / (lip_const * (X ** 2).sum(axis=0))
    else:
        steps[:] = n_samples / (lip_const * (X ** 2).sum(axis=0))

    lrs = np.empty(n_weights)
    learning_rates_c(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)
    learning_rates_f(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)
    learning_rates_csc(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)
    learning_rates_csr(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)
    learning_rates_f(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)
    learning_rates_c(lip_const, lrs)
    assert steps == pytest.approx(lrs, abs=1e-13)


@pytest.mark.parametrize("fit_intercept", (False, True))
def test_learning_rates_estimators_are_close(fit_intercept):
    n_samples, n_features = 1000, 4
    n_weights = n_features + int(fit_intercept)
    lip_const = 0.25
    rs = np.random.RandomState(seed=42)
    X = rs.randn(n_samples, n_features)

    X_f = np.asfortranarray(X)

    learning_rates_erm = learning_rates_erm_f_factory(X_f, fit_intercept)
    learning_rates_mom = learning_rates_mom_f_factory(
        X_f, fit_intercept, n_samples_in_block=100
    )
    learning_rates_ch = learning_rates_ch_f_factory(X_f, fit_intercept, eps=0.01)
    learning_rates_tmean = learning_rates_tmean_f_factory(
        X_f, fit_intercept, percentage=0.05
    )
    lrs_erm = np.empty(n_weights)
    learning_rates_erm(lip_const, lrs_erm)
    lrs_mom = np.empty(n_weights)
    learning_rates_mom(lip_const, lrs_mom)
    assert lrs_erm == pytest.approx(lrs_mom, rel=0.05)
    lrs_ch = np.empty(n_weights)
    learning_rates_ch(lip_const, lrs_ch)
    assert lrs_erm == pytest.approx(lrs_ch, rel=0.15)
    lrs_tmean = np.empty(n_weights)
    learning_rates_tmean(lip_const, lrs_tmean)
    assert lrs_erm == pytest.approx(lrs_tmean, rel=0.15)
