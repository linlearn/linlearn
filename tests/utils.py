# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.linalg import toeplitz
from scipy.special import expit
from sklearn.datasets import make_classification
import pytest


def simulate_true_logistic(
    n_samples=150,
    n_features=5,
    fit_intercept=True,
    corr=0.5,
    random_state=0,
    return_coef=False,
):
    rng = np.random.RandomState(random_state)
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
    if return_coef:
        return X, y, coef0, intercept0
    else:
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


def approx(v, abs=1e-15):
    return pytest.approx(v, abs)


def parameter_test_with_min(
    class_tested,
    parameter,
    valid_val,
    invalid_type_val,
    invalid_val,
    min_value=None,
    min_value_strict=None,
    min_value_str=None,
    mandatory=False,
    fixed_type=None,
    required_args=None,
):
    """Tests for an attribute of integer type

    Parameters
    ----------
    valid_val
        A valid value for the parameter

    invalid_type_val
        A value with invalid type

    invalid_val
        A value which is invalid because of its value

    parameter
    min_value
    mandatory

    Returns
    -------

    """

    if required_args is None:
        required_args = {}

    def get_params(param, val):
        """If the parameter is not 'n_classes', we need to specify
        `n_classes`, since it's mandatory to create the class
        """
        required_args[param] = val
        return required_args

    # If the parameter is mandatory, we check that an exception is raised
    # if not passed to the constructor
    if mandatory:
        with pytest.raises(TypeError) as exc_info:
            class_tested()
        assert exc_info.type is TypeError
        assert (
            exc_info.value.args[0] == "__init__() missing 1 required "
            "positional argument: '%s'" % parameter
        )

    if min_value is not None and min_value_strict is not None:
        raise ValueError(
            "You can't set both `min_value` and " "`min_value_strict` at the same time"
        )

    clf = class_tested(**get_params(parameter, valid_val))
    assert getattr(clf, parameter) == valid_val

    # If valid_val is valid, than valid_val + 1 is also valid
    setattr(clf, parameter, valid_val + 1)
    assert getattr(clf, parameter, valid_val + 1)

    with pytest.raises(
        ValueError,
        match="`%s` must be of type `%s`" % (parameter, fixed_type.__name__),
    ):
        setattr(clf, parameter, invalid_type_val)

    with pytest.raises(
        ValueError,
        match="`%s` must be of type `%s`" % (parameter, fixed_type.__name__),
    ):
        class_tested(**get_params(parameter, invalid_type_val))

    if min_value is not None:
        with pytest.raises(
            ValueError, match="`%s` must be >= %s" % (parameter, min_value_str)
        ):
            setattr(clf, parameter, invalid_val)

        with pytest.raises(
            ValueError, match="`%s` must be >= %s" % (parameter, min_value_str)
        ):
            class_tested(**get_params(parameter, invalid_val))

    if min_value_strict is not None:
        with pytest.raises(
            ValueError, match="`%s` must be > %s" % (parameter, min_value_str)
        ):
            setattr(clf, parameter, invalid_val)

        with pytest.raises(
            ValueError, match="`%s` must be > %s" % (parameter, min_value_str)
        ):
            class_tested(**get_params(parameter, invalid_val))

    clf = class_tested(**get_params(parameter, valid_val))
    # TODO: we should not need to change the dtype here
    X = np.random.randn(2, 2)
    y = np.array([0.0, 1.0])
    clf.partial_fit(X, y)
    with pytest.raises(
        ValueError,
        match="You cannot modify `%s` " "after calling `partial_fit`" % parameter,
    ):
        setattr(clf, parameter, valid_val)


def parameter_test_with_type(
    class_tested, parameter, valid_val, invalid_type_val, mandatory, fixed_type
):
    # TODO: code it
    pass
