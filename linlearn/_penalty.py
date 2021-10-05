# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause
from abc import ABC, abstractmethod
from math import fabs
from numba import jit
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH


jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


class Penalty(ABC):
    def __init__(self, strength):
        self.strength = strength

    @abstractmethod
    def value_one_unscaled_factory(self):
        pass

    @abstractmethod
    def value_one_scaled_factory(self):
        pass

    @abstractmethod
    def apply_one_unscaled_factory(self):
        pass

    @abstractmethod
    def apply_one_scaled_factory(self):
        pass

    def value_factory(self):
        value_one_unscaled = self.value_one_unscaled_factory()
        strength = self.strength

        @jit(**jit_kwargs)
        def value(w):
            val = 0.0
            for j1 in range(w.shape[0]):
                for j2 in range(w.shape[1]):
                    val += value_one_unscaled(w[j1, j2])
            return strength * val

        return value

    def apply_factory(self):
        apply_one_unscaled = self.apply_one_unscaled_factory()
        strength = self.strength

        @jit(**jit_kwargs)
        def apply(w, t, out):
            thresh = strength * t
            for j1 in range(w.shape[0]):
                for j2 in range(w.shape[1]):
                    out[j1, j2] = apply_one_unscaled(w[j1, j2], thresh)

        return apply


################################################################
# no penalization
################################################################


class NoPen(Penalty):
    def __init__(self, strength):
        Penalty.__init__(self, strength)

    def value_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def value_one_unscaled(x):
            return 0.0

        return value_one_unscaled

    def value_one_scaled_factory(self):
        @jit(**jit_kwargs)
        def value_one_scaled(x):
            return 0.0

        return value_one_scaled

    def apply_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def apply_one_unscaled(x, t):
            return x

        return apply_one_unscaled

    def apply_one_scaled_factory(self):
        @jit(**jit_kwargs)
        def apply_one_scaled(x, t):
            return x

        return apply_one_scaled


################################################################
# L2Sq penalization (l2-squared a.k.a ridge penalization)
################################################################


class L2Sq(Penalty):
    def __init__(self, strength):
        Penalty.__init__(self, strength)

    def value_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def value_one_unscaled(x):
            return 0.5 * x * x

        return value_one_unscaled

    def value_one_scaled_factory(self):
        strength = self.strength

        @jit(**jit_kwargs)
        def value_one_scaled(x):
            return strength * 0.5 * x * x

        return value_one_scaled

    def apply_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def apply_one_unscaled(x, t):
            return x / (1 + t)

        return apply_one_unscaled

    def apply_one_scaled_factory(self):
        strength = self.strength

        @jit(**jit_kwargs)
        def apply_one_scaled(x, t):
            return x / (1 + strength * t)

        return apply_one_scaled


################################################################
# l1 penalization
################################################################


class L1(Penalty):
    def __init__(self, strength):
        Penalty.__init__(self, strength)

    def value_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def value_one_unscaled(x):
            return fabs(x)

        return value_one_unscaled

    def value_one_scaled_factory(self):
        strength = self.strength

        @jit(**jit_kwargs)
        def value_one_scaled(x):
            return strength * fabs(x)

        return value_one_scaled

    def apply_one_unscaled_factory(self):
        @jit(**jit_kwargs)
        def apply_one_unscaled(x, t):
            if x > t:
                return x - t
            elif x < -t:
                return x + t
            else:
                return 0.0

        return apply_one_unscaled

    def apply_one_scaled_factory(self):
        strength = self.strength

        @jit(**jit_kwargs)
        def apply_one_scaled(x, t):
            thresh = strength * t
            if x > thresh:
                return x - thresh
            elif x < -thresh:
                return x + thresh
            else:
                return 0.0

        return apply_one_scaled


################################################################
# elasticnet penalization
################################################################


class ElasticNet(Penalty):
    def __init__(self, strength, l1_ratio):
        Penalty.__init__(self, strength)
        self.l1_ratio = l1_ratio

    def value_one_unscaled_factory(self):
        l1_ratio = self.l1_ratio

        @jit(**jit_kwargs)
        def value_one_unscaled(x):
            return l1_ratio * fabs(x) + (1.0 - l1_ratio) * 0.5 * x * x

        return value_one_unscaled

    def value_one_scaled_factory(self):
        strength = self.strength
        l1_ratio = self.l1_ratio

        @jit(**jit_kwargs)
        def value_one_scaled(x):
            return strength * (l1_ratio * fabs(x) + (1.0 - l1_ratio) * 0.5 * x * x)

        return value_one_scaled

    def apply_one_unscaled_factory(self):
        scale_l1 = self.l1_ratio
        state_l2sq = 1.0 - self.l1_ratio

        @jit(**jit_kwargs)
        def apply_one_unscaled(x, t):
            thresh_l1 = scale_l1 * t
            thresh_l2sq = state_l2sq * t
            if x > thresh_l1:
                return (x - thresh_l1) / (1.0 + thresh_l2sq)
            elif x < -thresh_l1:
                return (x + thresh_l1) / (1.0 + thresh_l2sq)
            else:
                return 0.0

        return apply_one_unscaled

    def apply_one_scaled_factory(self):
        scale_l1 = self.strength * self.l1_ratio
        state_l2sq = self.strength * (1.0 - self.l1_ratio)

        @jit(**jit_kwargs)
        def apply_one_scaled(x, t):
            thresh_l1 = scale_l1 * t
            thresh_l2sq = state_l2sq * t
            if x > thresh_l1:
                return (x - thresh_l1) / (1.0 + thresh_l2sq)
            elif x < -thresh_l1:
                return (x + thresh_l1) / (1.0 + thresh_l2sq)
            else:
                return 0.0

        return apply_one_scaled
