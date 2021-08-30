from collections import namedtuple
from numba import jit, njit, void, uint32, float64, generated_jit
from numba.experimental import jitclass
import numpy as np
from linlearn._utils import get_type, NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH
from time import time

# Estimator = namedtuple("Estimator", ["partial_deriv", "state"])

from linlearn._loss import decision_function_factory


from linlearn._loss import Logistic
from linlearn._estimator import ERM


X = np.random.randn(3, 3)
y = np.random.randn(3)
inner_products = np.empty(3)
fit_intercept = True

loss = Logistic()
estimator = ERM(X, y, loss, fit_intercept)

partial_deriv = estimator.partial_deriv_factory()

print(partial_deriv(1, inner_products))

exit(0)


@njit
def f():
    print(value(2.0, 1.0))
    print(deriv(2.0, 1.0))


f()

exit(0)

def factory():
    jit_kwargs = {
        "nopython": NOPYTHON,
        "nogil": NOGIL,
        "boundscheck": BOUNDSCHECK,
        "fastmath": FASTMATH,
    }

    @jit(**jit_kwargs)
    def f(x):
        return 2 * x

    @jit(**jit_kwargs)
    def g(x):
        return 3 * x

    return f, g


f, g = factory()

x = 2.0
print(f(x), g(x))

exit(0)


X = np.arange(0, 9, dtype=np.float64).reshape(3, 3)
w = np.arange(2, 5, dtype=np.float)
out = np.zeros((3,))
fit_intercept = False
decision_function = decision_function_factory(X, fit_intercept=fit_intercept)
decision_function(w, out)
print(out)

X = np.arange(0, 9, dtype=np.float64).reshape(3, 3)
w = np.arange(1, 5, dtype=np.float)
out = np.zeros((3,))
fit_intercept = True
decision_function = decision_function_factory(X, fit_intercept=fit_intercept)
decision_function(w, out)
print(out)

exit(0)

spec = [("weights", FLOAT[::1])]


@jitclass(spec)
class State(object):
    def __init__(self, n_samples):
        self.weights = np.empty(n_samples)


def apply_factory(fit_intercept, val):

    jitdec = jit(
        void(get_type(State)),
        nopython=NOPYTHON,
        nogil=NOGIL,
        boundscheck=BOUNDSCHECK,
        fastmath=FASTMATH,
    )

    if fit_intercept:

        @jitdec
        def apply(state):
            weights = state.weights
            for i in range(weights.shape[0]):
                weights[i] = val

    else:

        @jitdec
        def apply(state):
            weights = state.weights
            for i in range(1, weights.shape[0]):
                weights[i] = val

    return apply


state = State(5)
apply = apply_factory(True, 2.15)
apply(state)
print(state.weights)

state = State(5)
apply = apply_factory(False, 42.0)
apply(state)
print(state.weights)


exit(0)


@jitclass(spec)
class State(object):
    def __init__(self, n_samples):
        self.vec = np.empty(n_samples)


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
def f(x):
    return 2 * x


@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
def apply_sum_f(f, state):
    vec = state.vec
    s = 0.0
    for i in range(vec.shape[0]):
        s += f(vec[i])
    return s


# @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
def get_sum_f(f):
    @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
    def sum_f(state):
        vec = state.vec
        s = 0.0
        for i in range(vec.shape[0]):
            s += f(vec[i])
        return s

    return sum_f


def get_repeat1(sum_f):
    @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
    def repeat(state):
        s = 0.0
        for i in range(100_000):
            s += sum_f(f, state)
        return s

    return repeat


def get_repeat2():
    sum_f = get_sum_f(f)

    @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
    def repeat(state):
        s = 0.0
        for i in range(100_000):
            s += sum_f(state)
        return s

    return repeat


# @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
# def repeat1(sum_f, state):
#     s = 0.0
#     for i in range(100_000):
#         s += sum_f(f, state)
#     return s
#
#
# def get_repeat2(sumf):
#     @jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH)
#     def repeat2(state):
#         s = 0.0
#         for i in range(1000):
#             s += sumf(state)
#         return s
#     return repeat2


n_samples = 10
state = State(n_samples)
sumf = get_sum_f(f)

repeat1 = get_repeat1(apply_sum_f)
repeat2 = get_repeat2()

s = repeat1(state)
s = repeat2(state)

n_samples = 10_000
state = State(n_samples)
# sumf = get_sum_f(f, state)

tic = time()
s = repeat1(state)
toc = time()
print(toc - tic)


tic = time()
s = repeat2(state)
toc = time()
print(toc - tic)


exit(0)


@jitclass([])
class A(object):
    def __init__(self):
        pass


@jitclass([])
class B(object):
    def __init__(self):
        pass


@njit
def fa(state):
    print("Found an A")


@njit
def fb(state):
    print("Found a B")


@njit
def fc(state):
    print("Found an unknown")


@generated_jit(nopython=True, nogil=True, boundscheck=False)
def func(state):
    if isinstance(state, A):
        return fa
    elif isinstance(state, B):
        return fb
    else:
        return fc


func(A())
func(B())


exit(0)


spec_state_mom = [
    ("n_blocks", uint32),
    ("block_means", float64[::1]),
]


@jitclass(spec_state_mom)
class StateMOM(object):
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.block_means = np.zeros((n_blocks,), dtype=np.float64)


def get_estimator(**kwargs):
    state = StateMOM(**kwargs)
    print(state)
    return state


estimator = get_estimator(n_blocks=10)

print(estimator.n_blocks)
print(estimator.block_means)


@jitclass([])
class StateERM(object):
    def __init__(self):
        pass


@njit
def partial_deriv_least_squares():
    return 2.78


@njit
def partial_deriv_logistic():
    return 3.14


@njit
def partial_deriv_mom(
    j, partial_deriv_loss, state_solver, state_mom,
):
    state_solver.inner_products[0] = partial_deriv_loss()
    state_mom.block_means[0] = partial_deriv_loss()


@njit
def partial_deriv_erm(
    j, partial_deriv_loss, state_solver, state_erm,
):
    state_solver.inner_products[0] = partial_deriv_loss()
    # state_mom.block_means[0] = partial_deriv_loss()


spec_state_cgd = [
    ("inner_products", float64[::1]),
]


@jitclass(spec_state_cgd)
class StateCGD(object):
    def __init__(self, n_samples):
        self.inner_products = np.zeros((n_samples,), dtype=np.float64)


# mom_state = MOMState(3)
#
# print(mom_state)
#
# cgd_state = CGDState(10)
#
# print(cgd_state)
#

EstimatorMethods = namedtuple("EstimatorMethods", ["partial_deriv"])


@njit
def get_methods_mom():
    return EstimatorMethods(partial_deriv=partial_deriv_mom)


@njit
def get_methods_erm():
    return EstimatorMethods(partial_deriv=partial_deriv_erm)


# @njit
# def factory_mom_estimator(partial_deriv_loss, state_mom):
#     # mom_state = MOMState(n_blocks=n_blocks)
#     # block_means = state_mom.block_means
#     # inner_products = solver_state.inner_products
#
#     @njit
#     def partial_deriv(j, inner_products):
#         return partial_deriv_mom(j, inner_products, partial_deriv_loss, state_mom)
#
#     return Estimator(partial_deriv=partial_deriv)


# @njit
# def partial_deriv_mom(j, inner_products, partial_deriv_loss, estimator_state):
#     inner_products[0] = 3.14
#     block_means[0] = 2.78


# def solve(estimator):
#     partial_deriv = estimator.partial_deriv
#
#     @njit
#     def run():
#         partial_deriv(0)
#
#     run()


# @njit
# def partial_deriv_erm(j, cgd_state, strategy_state):
#     inner_products = cgd_state.inner_products
#     # block_means = strategy_state.block_means
#     inner_products[0] = 42

n_samples = 5
n_blocks = 2

# state_estimator = StateMOM(n_blocks=n_blocks)
state_estimator = StateERM()
state_solver = StateCGD(n_samples=n_samples)
estimator_methods = get_methods_erm()
# estimator_methods = get_methods_mom()


# estimator_methods

# estimator = factory_mom_estimator(partial_deriv_logistic, state_mom)

# estimator.partial_deriv(0, state_cgd.inner_products)


@njit
def solve(
    partial_deriv, state_solver, state_estimator,
):

    partial_deriv(0, partial_deriv_logistic, state_solver, state_estimator)


solve(estimator_methods.partial_deriv, state_solver, state_estimator)


print(state_solver.inner_products)

# print(state_estimator.block_means)

# j, partial_deriv_loss, state_solver, state_mom,
