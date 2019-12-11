import numpy as np
from numpy.linalg import norm
import pytest
from linlearn.prox import ProxL2Sq, ProxL1


class TestProx(object):

    # Prox classes to be tested
    prox_classes = [
        ProxL2Sq, ProxL1
    ]

    w = np.array([
        -0.86017247, -0.58127151, -0.6116414, 0.23186939, -0.85916332,
        1.6783094, 1.39635801, 1.74346116, -0.27576309, -1.00620197
    ])

    def test_strength(self):
        for Prox in self.prox_classes:
            TestProx.strength(Prox)

    def test_start_end(self):
        for Prox in self.prox_classes:
            TestProx.start_end(Prox)

    def test_positive(self):
        for Prox in self.prox_classes:
            TestProx.positive(Prox)

    def test_repr(self):
        for Prox in self.prox_classes:
            TestProx.repr(Prox)

    def test_call(self):
        for Prox in self.prox_classes:
            TestProx.call(Prox)

    def test_value(self):
        for Prox in self.prox_classes:
            TestProx.value(Prox)

    def test_prox_l2sq(self):
        w = self.w.copy()
        strength = 3e-2
        step = 1.7

        def approx(v):
            return pytest.approx(v, abs=1e-16)

        prox = ProxL2Sq(strength)
        out = w.copy()
        out *= 1. / (1. + step * strength)
        assert 0.5 * strength * norm(w) ** 2. == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))

        prox = ProxL2Sq(strength, (3, 8))
        out = w.copy()
        out[3:8] *= 1. / (1. + step * strength)
        assert 0.5 * strength * norm(w[3:8]) ** 2. == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))

        prox = ProxL2Sq(strength, (3, 8), positive=True)
        out = w.copy()
        out[3:8] *= 1. / (1. + step * strength)
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        assert 0.5 * strength * norm(w[3:8]) ** 2. == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))

    def test_prox_l1(self):
        w = self.w.copy()
        strength = 3e-2
        step = 1.7

        def approx(v):
            return pytest.approx(v, abs=1e-15)

        prox = ProxL1(strength)
        thresh = step * strength
        out = np.sign(w) * (np.abs(w) - thresh) * (np.abs(w) > thresh)
        assert strength * np.abs(w).sum() == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))

        prox = ProxL1(strength, (3, 8))
        thresh = step * strength
        w_sub = w[3:8]
        out = w.copy()
        out[3:8] = np.sign(w_sub) * (np.abs(w_sub) - thresh) * (np.abs(w_sub)
                                                                > thresh)
        assert strength * np.abs(w[3:8]).sum() == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))

        prox = ProxL1(strength, (3, 8), positive=True)
        thresh = step * strength
        w_sub = w[3:8]
        out = w.copy()
        out[3:8] = np.sign(w_sub) * (np.abs(w_sub) - thresh) * (np.abs(w_sub)
                                                                > thresh)
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        assert strength * np.abs(w[3:8]).sum() == approx(prox.value(w))
        assert out == approx(prox.call(w, step=step))


    @staticmethod
    def strength(Prox):
        with pytest.raises(TypeError):
            Prox()

        strength = 1e-3
        prox = Prox(strength)
        assert prox.strength == strength
        assert prox.no_python.strength == strength

        strength = 0.
        prox = Prox(strength)
        assert prox.strength == strength
        assert prox.no_python.strength == strength

        strength = 42.0
        prox.strength = strength
        assert prox.strength == strength
        assert prox.no_python.strength == strength

        with pytest.raises(ValueError, match="'strength' must be of "
                                             "float type"):
            prox = Prox(1)

        with pytest.raises(ValueError, match="'strength' must be of "
                                             "float type"):
            prox = Prox(1.0)
            prox.strength = 1

        with pytest.raises(ValueError, match="'strength' must be "
                                             "non-negative"):
            prox = Prox(-1.0)

        with pytest.raises(ValueError, match="'strength' must be "
                                             "non-negative"):
            prox = Prox(1.0)
            prox.strength = -1.0

    @staticmethod
    def start_end(Prox):
        # Check defaults
        strength = 1e-3
        prox = Prox(strength)
        assert prox.start_end is None
        assert prox.no_python.start == 0
        assert prox.no_python.end == 0
        assert prox.no_python.has_start_end is False

        # Check constructor
        prox = Prox(strength, (1, 42))
        assert prox.start_end == (1, 42)
        assert prox.no_python.start == 1
        assert prox.no_python.end == 42
        assert prox.no_python.has_start_end is True

        # Check properties
        prox = Prox(strength, (1, 42))
        prox.start_end = (3, 43)
        assert prox.start_end == (3, 43)
        assert prox.no_python.start == 3
        assert prox.no_python.end == 43
        assert prox.no_python.has_start_end is True

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, 1)

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (0, 42))
            prox.start_end = 1

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, [1])

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (0, 42))
            prox.start_end = [1]

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (1,))

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (0, 42))
            prox.start_end = (1,)

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (1, 2, 3))

        with pytest.raises(ValueError, match="'start_end' must be a tuple with "
                                             "2 elements"):
            prox = Prox(strength, (0, 42))
            prox.start_end = (1, 2, 3)

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (1.0, 2.0))

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (0, 42))
            prox.start_end = (1.0, 2.0)

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (1.0, 2))

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (0, 42))
            prox.start_end = (1.0, 2)

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (1, 2.0))

        with pytest.raises(ValueError, match="'start_end' tuple must contain "
                                             "integers"):
            prox = Prox(strength, (0, 42))
            prox.start_end = (1, 2.0)

    @staticmethod
    def positive(Prox):
        strength = 1e-3
        prox = Prox(strength)
        assert prox.positive is False
        assert prox.no_python.positive is False

        positive = True
        prox = Prox(strength, positive=positive)
        assert prox.positive == positive
        assert prox.no_python.positive == positive

        with pytest.raises(ValueError, match="'positive' must be of boolean "
                                             "type"):
            prox = Prox(strength, positive=1)

        with pytest.raises(ValueError, match="'positive' must be of boolean "
                                             "type"):
            prox = Prox(strength)
            prox.positive = 1

    @staticmethod
    def repr(Prox):
        class_name = Prox.__name__
        prox = Prox(1e-3)
        assert repr(prox) == class_name + "(strength=0.001, start_end=None, " \
                                          "positive=False)"

        prox = Prox(1e-3, start_end=(1, 42))
        assert repr(prox) == class_name + "(strength=0.001, " \
                                          "start_end=(1, 42), positive=False)"

        prox = Prox(1e-3, positive=True)
        assert repr(prox) == class_name + "(strength=0.001, start_end=None, " \
                                          "positive=True)"

        prox = Prox(1e-3, start_end=(1, 42), positive=True)
        assert repr(prox) == class_name + "(strength=0.001, " \
                                          "start_end=(1, 42), positive=True)"

    @staticmethod
    def call(Prox):
        prox = Prox(1e-3, (3, 8))
        w = np.random.randn(7)
        # TODO: this fails with match="'end' is larger than 'w.size[0]'" and I don't understand why
        with pytest.raises(ValueError):
            prox.call(w, step=1.)

    @staticmethod
    def value(Prox):
        prox = Prox(1e-3, (3, 8))
        w = np.random.randn(7)
        # TODO: this fails with match="'end' is larger than 'w.size[0]'" and I don't understand why
        with pytest.raises(ValueError):
            prox.value(w)
