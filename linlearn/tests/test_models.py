import numpy as np
from itertools import product
from numpy.random.mtrand import multivariate_normal
from scipy.optimize import check_grad
from scipy.linalg import toeplitz
import pytest

from linlearn.model import LeastSquares, Logistic
from linlearn.model.logistic import sigmoid


class TestModel(object):

    model_classes = [
        LeastSquares, Logistic
    ]

    w = np.array([
        -0.86017247, -0.58127151, -0.6116414, 0.23186939, -0.85916332,
        1.6783094, 1.39635801, 1.74346116, -0.27576309, -1.00620197
    ])

    def test_fit_intercept_and_is_set(self):
        for Model in self.model_classes:
            TestModel.fit_intercept(Model)

    def test_repr(self):
        for Model in self.model_classes:
            TestModel.repr(Model)

    def test_set(self):
        for Model in self.model_classes:
            TestModel.set(Model)

    @staticmethod
    def get_coef_intercept(n_features, fit_intercept):
        coef = np.random.randn(n_features)
        if fit_intercept:
            intercept = np.random.randn(1)
        else:
            intercept = None
        return coef, intercept

    @staticmethod
    def simulate_log_reg(n_samples, coef0, intercept0=None):
        n_features = coef0.shape[0]
        cov = toeplitz(0.5 ** np.arange(0, n_features))
        X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        logits = X.dot(coef0)
        if intercept0 is not None:
            logits += intercept0
        p = sigmoid(logits)
        y = np.random.binomial(1, p, size=n_samples).astype('float64')
        y[:] = 2 * y - 1
        return X, y

    @staticmethod
    def simulate_lin_reg(n_samples, coef0, intercept0=None):
        n_features = coef0.shape[0]
        cov = toeplitz(0.5 ** np.arange(0, n_features))
        X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        y = X.dot(coef0) + 0.1 * np.random.randn(n_samples)
        if intercept0 is not None:
            y += intercept0
        return X, y

    @staticmethod
    def get_weights(n_features, fit_intercept):
        if fit_intercept:
            return np.random.randn(n_features + 1)
        else:
            return np.random.randn(n_features)

    @staticmethod
    def loss_and_grad_match(Model, simulation_method):
        """A generic method to test that loss and grad match in a model"""
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        sample_weights = [
            None,
            np.linspace(0, n_samples-1, num=n_samples, dtype=np.float64)
        ]

        # Test that for any combinations loss and grad match
        for fit_intercept, sample_weight in product([False, True],
                                                    sample_weights):
            coef, intercept = TestModel.get_coef_intercept(n_features,
                                                           fit_intercept)
            X, y = simulation_method(n_samples, coef, intercept)
            w = TestModel.get_weights(n_features, fit_intercept)
            model = Model(fit_intercept=fit_intercept).set(X, y)

            def f(w):
                return model.no_python.loss_batch(w)

            def f_prime(w):
                out = np.empty(w.shape)
                model.no_python.grad_batch(w, out)
                return out

            assert check_grad(f, f_prime, w) < 1e-6

        def approx(v):
            return pytest.approx(v, abs=1e-15)

        # Test that loss and grad match for no weights given and weights equal
        # to one
        for fit_intercept in [False, True]:
            coef, intercept = TestModel.get_coef_intercept(n_features,
                                                           fit_intercept)
            X, y = simulation_method(n_samples, coef, intercept)
            w = TestModel.get_weights(n_features, fit_intercept)
            model1 = Model(fit_intercept=fit_intercept).set(X, y)
            sample_weight = np.ones(n_samples, dtype=np.float64)
            model2 = Model(fit_intercept=fit_intercept).set(X, y, sample_weight)
            w = TestModel.get_weights(n_features, fit_intercept)

            assert model1.no_python.loss_batch(w) == \
                   approx(model2.no_python.loss_batch(w))

            out1 = np.empty(w.shape)
            out2 = np.empty(w.shape)
            model1.no_python.grad_batch(w, out1)
            model2.no_python.grad_batch(w, out2)
            assert out1 == approx(out2)

    def test_least_squares_loss_and_grad_match(self):
        TestModel.loss_and_grad_match(LeastSquares, TestModel.simulate_lin_reg)

    def test_logistic_loss_and_grad_match(self):
        TestModel.loss_and_grad_match(Logistic, TestModel.simulate_log_reg)

    @staticmethod
    def fit_intercept(Model):
        model = Model()
        assert model.fit_intercept is True
        assert model.no_python.fit_intercept is True

        fit_intercept = False
        model = Model(fit_intercept)
        assert model.fit_intercept is False
        assert model.no_python.fit_intercept is False

        fit_intercept = True
        model.fit_intercept = fit_intercept
        assert model.fit_intercept is True
        assert model.no_python.fit_intercept is True

        with pytest.raises(ValueError, match="'fit_intercept' must be of "
                                             "boolean type"):
            Model(1)

        with pytest.raises(ValueError, match="'fit_intercept' must be of "
                                             "boolean type"):
            model = Model()
            model.fit_intercept = 1

    @staticmethod
    def repr(Model):
        class_name = Model.__name__

        model = Model()
        assert repr(model) == class_name + "(fit_intercept=True, " \
                                           "n_samples=None, n_features=None)"

        model = Model(fit_intercept=False)
        assert repr(model) == class_name + "(fit_intercept=False, " \
                                           "n_samples=None, n_features=None)"

        X = np.zeros((42, 3))
        y = np.zeros(42)

        model = Model().set(X, y)
        assert repr(model) == class_name + "(fit_intercept=True, " \
                                           "n_samples=42, n_features=3)"

    @staticmethod
    def set(Model):
        # First, we check that no copy is performed when not required
        np.random.seed(42)
        X = np.zeros((42, 3), dtype=np.float64)
        y = np.zeros(42, dtype=np.float64)
        model = Model().set(X, y)
        assert model.X is X and model.y is y
        assert model.no_python.X is X and model.no_python.y is y
        assert model.is_set is True
        assert model.no_python.is_set is True

        X = np.zeros((42, 3))
        y = np.zeros((41, ))
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "Found input variables with " \
                                         "inconsistent numbers of " \
                                         "samples: [42, 41]"

        X = np.zeros((42, 3))
        y = np.zeros((42, 2))
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "bad input shape (42, 2)"

        X = np.zeros((42, 3))
        X[0, 0] = np.nan
        y = np.zeros(42,)
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "Input contains NaN, infinity or a " \
                                         "value too large for dtype('float64')."

        X = np.zeros((42, 3))
        y = np.zeros(42,)
        y[0] = np.nan
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "Input contains NaN, infinity or a " \
                                         "value too large for dtype('float64')."

        X = np.zeros((42, 3))
        y = np.zeros((42, ))
        sample_weight = np.zeros(41)
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y, sample_weight)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "sample_weight.shape == (41,), " \
                                         "expected (42,)!"

        X = np.zeros((42, 3))
        y = np.zeros((42, ))
        sample_weight = np.zeros(42)
        sample_weight[0] = np.nan
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y, sample_weight)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "Input contains NaN, infinity or a " \
                                         "value too large for dtype('float64')."

        X = np.random.binomial(1, 0.5, size=(42, 3))
        y = np.zeros((42,), dtype=np.int64)
        ls = Model(fit_intercept=True).set(X, y)
        assert ls.X.dtype == 'float64'
        assert ls.y.dtype == 'float64'

        X = np.zeros((42, 3), dtype=np.float32)
        y = np.zeros((42,), dtype=np.int64)
        ls = Model(fit_intercept=True).set(X, y)
        assert ls.X.dtype == 'float64'
        assert ls.y.dtype == 'float64'

        X = np.zeros((42, 3), dtype=np.float32)
        y = np.array(21 * ['boo'] + 21 * ['baa'])
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "could not convert string to float: " \
                                         "'boo'"

        X = np.array(42 * 3 * ['boo']).reshape(42, 3)
        y = np.zeros((42,), dtype=np.int64)
        with pytest.raises(ValueError) as exc_info:
            Model().set(X, y)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == "could not convert string to float: " \
                                         "'boo'"

        X = np.zeros((42, 3))
        y = np.zeros((42, 1))
        model = Model().set(X, y)
        assert model.y.shape == (42,)
