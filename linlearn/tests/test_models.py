import numpy as np
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

    def test_least_squares_loss_and_grad(self):
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        for fit_intercept in [False, True]:
            coef, intercept = TestModel.get_coef_intercept(n_features,
                                                           fit_intercept)
            X, y = TestModel.simulate_lin_reg(n_samples, coef, intercept)
            w = self.get_weights(n_features, fit_intercept)
            model = LeastSquares(fit_intercept=fit_intercept).set(X, y)

            def f(w):
                return model.no_python.loss_batch(w)

            def f_prime(w):
                out = np.empty(w.shape)
                model.no_python.grad_batch(w, out)
                return out

            assert check_grad(f, f_prime, w) < 1e-6

    def test_logistic_loss_and_grad(self):
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        for fit_intercept in [False, True]:
            coef, intercept = TestModel.get_coef_intercept(n_features,
                                                           fit_intercept)
            X, y = TestModel.simulate_log_reg(n_samples, coef, intercept)
            w = self.get_weights(n_features, fit_intercept)
            model = Logistic(fit_intercept=fit_intercept).set(X, y)

            def f(w):
                return model.no_python.loss_batch(w)

            def f_prime(w):
                out = np.empty(w.shape)
                model.no_python.grad_batch(w, out)
                return out

            assert check_grad(f, f_prime, w) < 1e-6

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
            model = Model(1)

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
        y = np.zeros(3)

        model = Model().set(X, y)
        assert repr(model) == class_name + "(fit_intercept=True, " \
                                           "n_samples=42, n_features=3)"
        # TODO: test error if X.shape[0] != y.shape[0]
        # TODO: test error if not C continuous and all...

    @staticmethod
    def set(Model):
        np.random.seed(42)
        X = np.random.randn(42, 3)
        y = np.random.randn(3)
        model = Model().set(X, y)
        assert model.X is X and model.y is y
        assert model.no_python.X is X and model.no_python.y is y
        assert model.is_set is True
        assert model.no_python.is_set is True
