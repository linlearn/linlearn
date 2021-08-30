from collections import namedtuple
from math import fabs
from numba import jit
from numba.experimental import jitclass

from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FLOAT, NP_FLOAT


# Functions associated to a penalty
Penalty = namedtuple("Penalty", ["state", "value_one", "apply_one"])


################################################################
# Generic functions
################################################################

# TODO: we can specialize some generic functions through the state. For instance,
#  we can test something in the state to understand that it's "none" penalization and
#  simply return 0 for instance ? Or we can use @generated_jit for this


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def value(value_one, state, w):
    """Computes the value of the (scaled) penalization function on the whole input
    """
    val = 0.0
    for w_j in w:
        val += value_one(state, w_j)
    return val


# # @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
# # def scaled_value(value_single, state, x):
# #     """Computes the value of the (scaled) penalization function corresponding to a
# #     single coordinate of the input"""
#     return state.strength * value_single(state, x, state)


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def apply(apply_one, state, w, t, out):
    """Applies the (scaled) penalization on the whole input"""
    t_scaled = state.strength * t
    for j in range(w.shape[0]):
        out[j] = apply_one(state, w[j], t_scaled)


################################################################
# no penalization
################################################################


spec_state_none = [("strength", FLOAT)]


@jitclass(spec_state_none)
class StateNone(object):
    def __init__(self):
        self.strength = 0.0


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def value_one_none(state, x):
    return 0.0


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def apply_one_none(state, x, t):
    return x


################################################################
# l2sq penalization (L2 squared a.k.a ridge penalization)
################################################################

spec_state_l2sq = [("strength", FLOAT)]


@jitclass(spec_state_l2sq)
class StateL2Sq(object):
    def __init__(self, strength):
        self.strength = strength


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def value_one_l2sq(state, x):
    return state.strength * x * x / 2.0


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def apply_one_l2sq(state, x, t):
    return x / (1 + t * state.strength)


################################################################
# l1 penalization
################################################################

spec_state_l1 = [("strength", FLOAT)]


@jitclass(spec_state_l1)
class StateL1(object):
    def __init__(self, strength):
        self.strength = strength


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def value_one_l1(state, x):
    return state.strength * fabs(x)


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def apply_one_l1(state, x, t):
    thresh = state.strength * t
    if x > thresh:
        return x - thresh
    elif x < -thresh:
        return x + thresh
    else:
        return 0.0


################################################################
# elasticnet penalization
################################################################


spec_state_elasticnet = [("strength", FLOAT), ("l1_ratio", FLOAT)]


@jitclass(spec_state_elasticnet)
class StateElasticnet(object):
    def __init__(self, strength, l1_ratio):
        self.strength = strength
        self.l1_ratio = l1_ratio


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def value_one_elasticnet(state, x):
    return state.strength * (
        state.l1_ratio * fabs(x) + (1.0 - state.l1_ratio) * x * x / 2.0
    )


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK)
def apply_one_elasticnet(state, x, t):
    thresh_l1 = state.strength * state.l1_ratio * t
    thresh_l2sq = state.strength * (1.0 - state.l1_ratio) * t
    if x > thresh_l1:
        return (x - thresh_l1) / (1.0 + thresh_l2sq)
    elif x < -thresh_l1:
        return (x + thresh_l1) / (1.0 + thresh_l2sq)
    else:
        return 0.0


# TODO: could be faster for elasticnet but what the heck !
#  @njit
# def elasticnet_value(x, strength, l1_ratio):
#     l1, l2sq = 0.0, 0.0
#     for j in range(x.shape[0]):
#         l1 += fabs(x[j])
#         l2sq += x[j] * x[j]
#
#     return strength * (l1_ratio * l1 + (1 - l1_ratio) * l2sq / 2)


def get_penalty(penalty, **kwargs):
    if penalty == "none":
        return Penalty(
            state=StateNone(**kwargs),
            value_one=value_one_none,
            apply_one=apply_one_none,
        )
    elif penalty == "l2":
        return Penalty(
            state=StateL2Sq(**kwargs),
            value_one=value_one_l2sq,
            apply_one=apply_one_l2sq,
        )
    elif penalty == "l1":
        return Penalty(
            state=StateL1(**kwargs), value_one=value_one_l1, apply_one=apply_one_l1
        )
    elif penalty == "elasticnet":
        return Penalty(
            state=StateElasticnet(**kwargs),
            value_one=value_one_elasticnet,
            apply_one=apply_one_elasticnet,
        )
    else:
        raise ValueError("Unknown penalty")
