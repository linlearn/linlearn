import numpy as np

from math import fabs
from numba import njit

from numba.types import int64, float64, boolean

# from linlearn.prox_old.prox_old import Prox
# from linlearn.prox_old.utils import call_with_single, value_with_single

import numpy as np
from collections import namedtuple

# Functions associated to a penalty
Penalty = namedtuple("Penalty", ["value_single", "value", "apply", "apply_single"])


# # Prototype for penalty value functions
# specs_value = [
#     "float32(float32)",
#     "float64(float64)"
# ]
#
# # Prototype for penalty apply functions
# specs_apply = [
#     "float32(float32, float32)",
#     "float64(float64, float64)"
# ]


#
# Let us first start with general purpose functions
#


def penalty_value_factory():
    pass


@njit
def penalty_value(penalty_value_single, x, strength):
    val = 0.0
    for j in range(x.shape[0]):
        val += penalty_value_single(x[j])
    return strength * val


# NB: we could use @vectorize from numba, but it would break everything for code
# coverage when using NUMBA_DISABLE_JIT=1


@njit
def penalty_apply(penalty_apply_single, x, t, out):
    for j in range(x.shape[0]):
        out[j] = penalty_apply_single(x[j], t)


#
# l2sq (ridge) penalization functions
#
@njit
def l2sq_value_single(x):
    return x * x / 2


@njit
def l2sq_apply_single(x, t):
    return x / (1 + t)


@njit
def l2sq_value(x, strength):
    return penalty_value(l2sq_value_single, x, strength)


@njit
def l2sq_apply(x, t, out):
    penalty_apply(l2sq_apply_single, x, t, out)


def l2sq_penalty_factory(strength):
    @njit
    def value_single(x):
        return strength * l2sq_value_single(x)

    @njit
    def value(x):
        return l2sq_value(x, strength)

    @njit
    def apply_single(x, t):
        return l2sq_apply_single(x, strength * t)

    @njit
    def apply(x, t):
        return l2sq_apply(x, strength * t)

    return Penalty(
        value_single=value_single, value=value, apply_single=apply_single, apply=apply,
    )


#
# l1 penalization functions
#
@njit
def l1_value_single(x):
    return fabs(x)


@njit
def l1_apply_single(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.0


@njit
def l1_value(x, strength):
    return penalty_value(l1_value_single, x, strength)


@njit
def l1_apply(x, t, out):
    penalty_apply(l1_apply_single, x, t, out)


def l1_penalty_factory(strength):
    @njit
    def value_single(x):
        return strength * l1_value_single(x)

    @njit
    def value(x):
        return l1_value(x, strength)

    @njit
    def apply_single(x, t):
        return l1_apply_single(x, strength * t)

    @njit
    def apply(x, t):
        return l1_apply(x, strength * t)

    return Penalty(
        value_single=value_single, value=value, apply_single=apply_single, apply=apply,
    )


# l1_penalty = Penalty(
#     value_single=l1_value_single,
#     value=l1_value,
#     apply_single=l1_apply_single,
#     apply=l1_apply,
# )


penalties_factory = {"l1": l1_penalty_factory, "l2": l2sq_penalty_factory}


@njit
def main():
    x = l2sq_value_single(4.0)
    print(x)
    x = l2sq_apply_single(4.0, 2.0)
    print(x)

    x = np.array([1.0, -3.0])

    print(l2sq_value(x, 2.0))

    l2sq_apply(x, 2.0, x)
    print(x)


# main()

# def penalty_l1_call(w, thresh, out, start, end, positive):
#     """
#     TODO: insert docstring
#     """
#     for i in range(start, end):
#         wi = w[i]
#         if wi > 0:
#             if wi > thresh:
#                 out[i] = wi - thresh
#             else:
#                 out[i] = 0
#         else:
#             # If wi is negative and we project onto the non-negative half-plane
#             # we set it to 0
#             if positive:
#                 out[i] = 0
#             else:
#                 if wi < -thresh:
#                     out[i] = wi + thresh
#                 else:
#                     out[i] = 0


#
# @njit
# def apply_single_l2sq(x, t):
#
# @njit
# def penalty_l1_value_single(x):
#     return fabs(x)
#
# @njit
# def penalty_l1_value_single(x):
#     return fabs(x)


# class Prox(object):
#
#     def __init__(self, no_python_class, strength, start_end=None,
#                  positive=False):
#         self.no_python = no_python_class(0., 0, 0, False, False)
#         self.strength = strength
#         self.start_end = start_end
#         self.positive = positive
#
#     @property
#     def strength(self):
#         return self.no_python.strength
#
#     @strength.setter
#     def strength(self, val):
#         if not isinstance(val, float):
#             raise ValueError("'strength' must be of float type")
#         elif val < 0:
#             raise ValueError("'strength' must be non-negative")
#         else:
#             self.no_python.strength = val
#
#     @property
#     def start_end(self):
#         if self.no_python.has_start_end:
#             return self.no_python.start, self.no_python.end
#         else:
#             return None
#
#     @start_end.setter
#     def start_end(self, val):
#         if val is None:
#             self.no_python.start = 0
#             self.no_python.end = 0
#             self.no_python.has_start_end = False
#         elif not isinstance(val, tuple) or len(val) != 2:
#             raise ValueError("'start_end' must be a tuple "
#                              "with 2 elements")
#         else:
#             start, end = val
#             if not (isinstance(start, int) and isinstance(end, int)):
#                 raise ValueError("'start_end' tuple must contain integers")
#             if end <= start:
#                 raise ValueError("'end' must be larger than 'start'")
#             self.no_python.start = start
#             self.no_python.end = end
#             self.no_python.has_start_end = True
#
#     @property
#     def positive(self):
#         return self.no_python.positive
#
#     @positive.setter
#     def positive(self, val):
#         if type(val) is bool:
#             self.no_python.positive = val
#         else:
#             raise ValueError("'positive' must be of boolean type")
#
#     def call(self, w, step, out=None):
#         # TODO: checks on w, step and out
#         if out is None:
#             out = np.empty(w.shape[0])
#         self.no_python.call(w, step, out)
#         return out
#
#     def value(self, w):
#         # TODO: checks on w
#         return self.no_python.value(w)
#
#     def __repr__(self):
#         strength = self.strength
#         start_end = self.start_end
#         positive = self.positive
#         r = self.__class__.__name__
#         r += "(strength={strength}".format(strength=strength)
#         r += ", start_end={start_end}".format(start_end=start_end)
#         r += ", positive={positive})".format(positive=positive)
#         return r
#
#

#
# specs = [
#     ('strength', float64),
#     ('start', int64),
#     ('end', int64),
#     ('positive', boolean),
#     ('has_start_end', boolean)
# ]
# @jitclass(specs)
# class ProxL1NoPython(object):
#
#     def __init__(self, strength, start, end, has_start_end, positive):
#         self.strength = strength
#         self.start = start
#         self.end = end
#         self.has_start_end = has_start_end
#         self.positive = positive
#
#     def call_single(self, x, step):
#         thresh = step * self.strength
#         if x > 0:
#             if x > thresh:
#                 return x - thresh
#             else:
#                 return 0
#         else:
#             # If x is negative and we project onto the non-negative half-plane
#             # we set it to 0
#             if self.positive:
#                 return 0
#             else:
#                 if x < -thresh:
#                     return x + thresh
#                 else:
#                     return 0
#
#     def value_single(self, x):
#         return fabs(x)
#
#     def call(self, w, step, out):
#         call_with_single(self, w, step, out)
#
#     def value(self, w):
#         return value_with_single(self, w)
#
#
# class ProxL1(Prox):
#     """Proximal operator of the L1 norm (Lasso penalization)
#
#     Parameters
#     ----------
#     strength : `float`
#         Level of L1 penalization.
#
#     start_end : `tuple` of two `int`, default=`None`
#         Range on which the prox_old is applied. If `None` then the prox_old is
#         applied on the whole vector.
#
#     positive : `bool`, default=`False`
#         If True, apply ridge penalization together with a projection
#         onto the set of vectors with non-negative entries.
#     """
#     def __init__(self, strength, start_end=None, positive=False):
#         Prox.__init__(self, ProxL1NoPython, strength, start_end, positive)
#
#
# from numba import jitclass
# from numba.types import int64, float64, boolean
# from linlearn.prox_old.prox_old import Prox
# from linlearn.prox_old.utils import call_with_single, value_with_single
#
#
# specs = [
#     ('strength', float64),
#     ('start', int64),
#     ('end', int64),
#     ('positive', boolean),
#     ('has_start_end', boolean)
# ]
# @jitclass(specs)
# class ProxL2NoPython(object):
#
#     def __init__(self, strength, start, end, has_start_end, positive):
#         self.strength = strength
#         self.start = start
#         self.end = end
#         self.has_start_end = has_start_end
#         self.positive = positive
#
#     def call_single(self, x, step):
#         if self.positive and x < 0:
#             return 0
#         else:
#             return x / (1 + step * self.strength)
#
#     def value_single(self, x):
#         return x * x / 2
#
#     def call(self, w, step, out):
#         call_with_single(self, w, step, out)
#
#     def value(self, w):
#         return value_with_single(self, w)
#
#
# class ProxL2Sq(Prox):
#     """Proximal operator of the squared L2 norm (ridge penalization)
#
#     Parameters
#     ----------
#     strength : `float`
#         Level of L2 penalization.
#
#     start_end : `tuple` of two `int`, default=`None`
#         Range on which the prox_old is applied. If `None` then the prox_old is
#         applied on the whole vector.
#
#     positive : `bool`, default=`False`
#         If True, apply ridge penalization together with a projection
#         onto the set of vectors with non-negative entries.
#     """
#
#     def __init__(self, strength, start_end=None, positive=False):
#         Prox.__init__(self, ProxL2NoPython, strength, start_end, positive)
#
# from numba import njit
#
#
# @njit
# def value_with_single(prox_old, w):
#     if prox_old.has_start_end:
#         if prox_old.end > w.shape[0]:
#             raise ValueError("'end' is larger than 'w.size[0]'")
#         start, end = prox_old.start, prox_old.end
#     else:
#         start, end = 0, w.shape[0]
#     val = 0.
#     w_sub = w[start:end]
#     for i in range(w_sub.shape[0]):
#         val += prox_old.value_single(w_sub[i])
#     return prox_old.strength * val
#
#
# @njit
# def call_with_single(prox_old, w, step, out):
#     if w.shape != out.shape:
#         raise ValueError("'w' and 'out' must have the same shape")
#     if prox_old.has_start_end:
#         if prox_old.end > w.shape[0]:
#             raise ValueError("'end' is larger than 'w.size[0]'")
#         start, end = prox_old.start, prox_old.end
#         # out is w changed only in [start, end], so we must put w in out
#         out[:] = w
#     else:
#         start, end = 0, w.shape[0]
#
#     w_sub = w[start:end]
#     out_sub = out[start:end]
#     for i in range(w_sub.shape[0]):
#         out_sub[i] = prox_old.call_single(w_sub[i], step)
#
#
# @njit
# def is_in_range(i, start, end):
#     if i >= start:
#         return False
#     elif i < end:
#         return False
#     else:
#         return True

# from abc import abstractmethod
# import numpy as np
# from math import fabs
# from numba import njit
# from numba.types import uint, int8
#
# from scipy.stats import norm
#
#
# class Penalty(object):
#     def __init__(self, strength, start_end=None, positive=False):
#         self.strength = strength
#         self.start_end = start_end
#         self.positive = positive
#
#     @property
#     def strength(self):
#         return self._strength
#
#     @strength.setter
#     def strength(self, val):
#         if not isinstance(val, float):
#             raise ValueError("'strength' must be of float type")
#         elif val < 0:
#             raise ValueError("'strength' must be non-negative")
#         else:
#             self._strength = val
#
#     @property
#     def start_end(self):
#         return self._start_end
#
#     @start_end.setter
#     def start_end(self, val):
#         if val is None:
#             self._start_end = None
#         elif not isinstance(val, tuple) or len(val) != 2:
#             raise ValueError("'start_end' must be a tuple " "with 2 elements")
#         else:
#             start, end = val
#             if not (isinstance(start, int) and isinstance(end, int)):
#                 raise ValueError("'start_end' tuple must contain integers")
#             if end <= start:
#                 raise ValueError("'end' must be larger than 'start'")
#             self._start_end = val
#
#     @property
#     def positive(self):
#         return self._positive
#
#     @positive.setter
#     def positive(self, val):
#         if type(val) is bool:
#             self._positive = val
#         else:
#             raise ValueError("'positive' must be of boolean type")
#
#     @abstractmethod
#     def _call(self, w, step, out, start, end):
#         return
#
#     def call(self, w, step=1.0, out=None):
#         # TODO: checks on step
#         if out is None:
#             out = np.empty(w.shape[0])
#         else:
#             # TODO: and check dtype
#             if w.shape != out.shape:
#                 raise ValueError("'w' and 'out' must have the same shape")
#
#         if self.start_end is None:
#             start, end = 0, w.shape[0]
#         else:
#             start, end = self.start_end
#             # print("start, end, w.shape[0]:", start, end, w.shape[0])
#             # We won't modify the whole vector, so a copy is required
#             out[:] = w
#             if end > w.shape[0]:
#                 raise ValueError("'end' is larger than 'w.size[0]'")
#
#         self._call(w, step, out, start, end)
#         return out
#
#     @abstractmethod
#     def _value(self, w, start, end):
#         return
#
#     def value(self, w):
#         # TODO: checks on w
#         if self.start_end is None:
#             start, end = 0, w.shape[0]
#         else:
#             start, end = self.start_end
#             # print("start, end, w.shape[0]:", start, end, w.shape[0])
#             # We won't modify the whole vector, so a copy is required
#             if end > w.shape[0]:
#                 raise ValueError("'end' is larger than 'w.size[0]'")
#         return self._value(w, start, end)
#
#     @property
#     def name(self):
#         return self.__class__.__name__
#
#     def __repr__(self):
#         strength = self.strength
#         start_end = self.start_end
#         positive = self.positive
#         r = self.name
#         r += "(strength={strength}".format(strength=strength)
#         r += ", start_end={start_end}".format(start_end=start_end)
#         r += ", positive={positive})".format(positive=positive)
#         return r


# specs = [
#     ("strength", float64),
#     ("start", int64),
#     ("end", int64),
#     ("positive", boolean),
#     ("has_start_end", boolean),
# ]

#
# @njit
# def penalty_l1_call(w, thresh, out, start, end, positive):
#     """
#     TODO: insert docstring
#     """
#     for i in range(start, end):
#         wi = w[i]
#         if wi > 0:
#             if wi > thresh:
#                 out[i] = wi - thresh
#             else:
#                 out[i] = 0
#         else:
#             # If wi is negative and we project onto the non-negative half-plane
#             # we set it to 0
#             if positive:
#                 out[i] = 0
#             else:
#                 if wi < -thresh:
#                     out[i] = wi + thresh
#                 else:
#                     out[i] = 0
#
#
# @njit
# def penalty_l1_value(w, strength, start, end):
#     """
#     TODO: insert docstring
#     """
#     val = 0.0
#     for i in range(start, end):
#         val += fabs(w[i])
#     return strength * val
#
#
# class PenaltyL1(Penalty):
#     """L1 norm penalization (Lasso penalization)
#
#     Parameters
#     ----------
#     strength : `float`
#         Level of L1 penalization.
#
#     start_end : `tuple` of two `int`, default=`None`
#         Range on which the prox_old is applied. If `None` then the prox_old is
#         applied on the whole vector.
#
#     positive : `bool`, default=`False`
#         If True, apply ridge penalization together with a projection
#         onto the set of vectors with non-negative entries.
#     """
#
#     def __init__(self, strength, start_end=None, positive=False):
#         Penalty.__init__(self, strength, start_end, positive)
#
#     def _call(self, w, step, out, start, end):
#         penalty_l1_call(w, step * self.strength, out, start, end, self.positive)
#
#     def _value(self, w, start, end):
#         return penalty_l1_value(w, self.strength, start, end)


# @njit
# def value_with_single(prox_old, w):
#     if prox_old.has_start_end:
#         if prox_old.end > w.shape[0]:
#             raise ValueError("'end' is larger than 'w.size[0]'")
#         start, end = prox_old.start, prox_old.end
#     else:
#         start, end = 0, w.shape[0]
#     val = 0.0
#     w_sub = w[start:end]
#     for i in range(w_sub.shape[0]):
#         val += prox_old.value_single(w_sub[i])
#     return prox_old.strength * val
#
#
# @njit
# def call_with_single(prox_old, w, step, out):
#     if w.shape != out.shape:
#         raise ValueError("'w' and 'out' must have the same shape")
#     if prox_old.has_start_end:
#         if prox_old.end > w.shape[0]:
#             raise ValueError("'end' is larger than 'w.size[0]'")
#         start, end = prox_old.start, prox_old.end
#         # out is w changed only in [start, end], so we must put w in out
#         out[:] = w
#     else:
#         start, end = 0, w.shape[0]
#
#     w_sub = w[start:end]
#     out_sub = out[start:end]
#     for i in range(w_sub.shape[0]):
#         out_sub[i] = prox_old.call_single(w_sub[i], step)
#
#
# @njit
# def is_in_range(i, start, end):
#     if i >= start:
#         return False
#     elif i < end:
#         return False
#     else:
#         return True

#
# // License: BSD 3 clause
#
# #include "tick/prox_old/prox_sorted_l1.h"
#
# template <class T, class K>
# void TProxSortedL1<T, K>::compute_weights(void) {
#   TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
# }
#

# void TProxSortedL1<T, K>::call(const Array<K> &coeffs, T t, Array<K> &out,
#                                ulong start, ulong end) {


# template <class T, class K>
# void TProxSlope<T, K>::compute_weights(void) {
#   if (!weights_ready) {
#     ulong size = end - start;
#     weights = Array<T>(size);
#     for (ulong i = 0; i < size; i++) {
#       // tmp is double as float prevents adequate precision for
#       //  standard_normal_inv_cdf
#       double tmp = false_discovery_rate / (2 * size);
#       weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
#     }
#     weights_ready = true;
#   }
# }

#
# # This piece comes from E. Candes and co-authors from SLOPE Matlab's code
# @njit
# def prox_sorted_l1(y, weights, x):
#     n = y.shape[0]
#     s = np.empty(n)
#     w = np.empty(n)
#     idx_i = np.empty(n, dtype=uint)
#     idx_j = np.empty(n, dtype=uint)
#     i, j, k = 0, 0, 0
#     for i in range(n):
#         idx_i[k] = i
#         idx_j[k] = i
#         s[k] = y[i] - weights[i]
#         w[k] = s[k]
#         while (k > 0) and (w[k - 1] <= w[k]):
#             k -= 1
#             idx_j[k] = i
#             s[k] += s[k + 1]
#             w[k] = s[k] / (i - idx_i[k] + 1)
#         k += 1
#     for j in range(k):
#         d = w[j]
#         if d < 0:
#             d = 0
#         for i in range(idx_i[j], idx_j[j] + 1):
#             x[i] = d
#
#
# @njit
# def penalty_sorted_l1_call(coeffs, t, weights, out, start, end):
#     size = end - start
#     thresh = t
#     sub_coeffs = coeffs[start:end]
#     sub_out = out[start:end]
#     sub_out.fill(0)
#
#     # weights = get_weights_bh(0.05, t)
#     weights_copy = thresh * weights.copy()
#     sub_coeffs_sign = np.empty(size, dtype=int8)
#     sub_coeffs_abs = np.empty(size)
#
#     # Indices that sort sub_coeffs
#     idx = np.argsort(np.abs(sub_coeffs))[::-1]
#     sub_coeffs_sorted = sub_coeffs[idx]
#     sub_coeffs_abs_sort = np.empty(size)
#
#     # Multiply the weights by the threshold
#     for i in range(size):
#         # weights_copy[i] *= thresh
#         sub_coeffs_i = sub_coeffs[i]
#         sub_coeffs_abs_sort[i] = fabs(sub_coeffs_sorted[i])
#         if sub_coeffs_i >= 0:
#             sub_coeffs_sign[i] = 1
#             sub_coeffs_abs[i] = sub_coeffs_i
#         else:
#             sub_coeffs_sign[i] = -1
#             sub_coeffs_abs[i] = -sub_coeffs_i
#
#     # Where do the crossing occurs?
#     crossing = 0
#     for i in range(size):
#         if sub_coeffs_abs_sort[i] > weights_copy[i]:
#             crossing = i
#
#     if crossing > 0:
#         n_sub_coeffs = crossing + 1
#     else:
#         n_sub_coeffs = size
#
#     subsub_coeffs = sub_coeffs_abs_sort[:n_sub_coeffs]
#     subsub_out = np.zeros(n_sub_coeffs)
#
#     prox_sorted_l1(subsub_coeffs, weights_copy, subsub_out)
#
#     for i in range(n_sub_coeffs):
#         sub_out[idx[i]] = subsub_out[i]
#
#     for i in range(size):
#         sub_out[i] = sub_out[i] * sub_coeffs_sign[i]
#
#     return out
#
#
# def get_weights_bh(fdr, size):
#     """Computes the Benjamini-Hochberg weights for the SLOPE penalization
#
#     Parameters
#     ----------
#     fdr : `float`
#         Desired False Discovery Rate. Must be in (0, 1)
#
#     size : `int`
#         Size of the vector
#
#     Returns
#     -------
#     output : `np.array`, shape=(size,)
#         The  Benjamini-Hochberg weights
#     """
#     tmp = fdr / (2 * size)
#     return norm.ppf(1 - tmp * np.arange(1, size + 1))
#
#
# def penalty_slope_value(w, weights, strength, start, end):
#     w_abs = np.abs(w[start:end])
#     # Indices that sort the entries in decreasing order
#     idx = np.argsort(w_abs)[::-1]
#     return strength * w_abs[idx].dot(weights)
#
#
# class PenaltySlope(Penalty):
#     """Slope penalization. This penalization is particularly relevant for feature
#     selection, when features correlation is not too high.
#
#     Parameters
#     ----------
#     strength : `float`
#         Level of penalization
#
#     fdr : `float`, default=0.6
#         Desired False Discovery Rate for detection of non-zeros in
#         the coefficients. Must be between 0 and 1.
#
#     start_end : `tuple` of two `int`, default=`None`
#         Range on which the prox_old is applied. If `None` then the prox_old is
#         applied on the whole vector
#
#     Attributes
#     ----------
#     weights : `np.array`, shape=(n_coeffs,)
#         The weights used in the penalization. They are automatically
#         computed, depending on the ``weights_type`` and ``fdr``
#         parameters. Note that these weights do not include the strength used.
#
#     Notes
#     -----
#     Uses the stack-based algorithm for FastProxL1 from
#
#     * SLOPE--Adaptive Variable Selection via Convex Optimization, by
#       Bogdan, M. and Berg, E. van den and Sabatti, C. and Su, W. and Candes, E. J.
#       arXiv preprint arXiv:1407.3824, 2014
#     """
#
#     def __init__(self, strength, start_end, fdr, positive=False):
#         # For now, start_end argument mandatory in this penalty
#         Penalty.__init__(self, strength, start_end, positive)
#         self.strength = strength
#         self.fdr = fdr
#         # TODO: we can set positive=True for this prox_old
#         self.positive = False
#         self.weights = None
#         self._weights_ready = False
#         self._size = self.start_end[1] - self.start_end[0]
#
#     def _call(self, w, step, out, start, end):
#         if not self._weights_ready:
#             self.weights = get_weights_bh(self.fdr, self._size)
#             self._weights_ready = True
#         penalty_sorted_l1_call(w, step * self.strength, self.weights, out, start, end)
#
#     def _value(self, w, start, end):
#         if not self._weights_ready:
#             self.weights = get_weights_bh(self.fdr, self._size)
#             self._weights_ready = True
#         start, end = self.start_end
#         return penalty_slope_value(w, self.weights, self.strength, start, end)
#
#     @property
#     def fdr(self):
#         return self._fdr
#
#     @fdr.setter
#     def fdr(self, val):
#         if not isinstance(val, float):
#             raise ValueError("'fdr' must be of float type")
#         elif val <= 0:
#             raise ValueError("'fdr' must be positive")
#         elif val >= 1:
#             raise ValueError("'fdr' must be less than 1")
#         else:
#             self._weights_ready = False
#             self._fdr = val
