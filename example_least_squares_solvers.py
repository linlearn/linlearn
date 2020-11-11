import numpy as np

from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz

from linlearn.model import LeastSquares
from linlearn.model.utils import inner_prods
from linlearn.prox_old import ProxL2Sq, ProxL1
from linlearn.solver_old import SVRG, CGD, History

from linlearn.plot import plot_history

from linlearn.model.utils import col_squared_norm

np.set_printoptions(precision=2)


n_samples = 100_000
n_features = 200

coef0 = 2 * np.random.randn(n_features)
intercept0 = -2.0
# intercept0 = None

fit_intercept = intercept0 is not None
cov = toeplitz(0.5 ** np.arange(0, n_features))
X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
y = X.dot(coef0) + 0.1 * np.random.randn(n_samples)
if fit_intercept:
    y += intercept0
    w_start = np.zeros(n_features + 1)
else:
    w_start = np.zeros(n_features)

model = LeastSquares(fit_intercept).set(X, y)

# w = np.random.randn(w_start.shape[0])
# grad1 = np.empty((w.size,), dtype=np.float64)
#
# model.no_python.grad_batch(w, grad1)
#
# print('grad1: ', grad1)
#
# grad2 = np.empty(w.shape, dtype=np.float64)
#
# inner_products = np.empty(model.no_python.n_samples, dtype=np.float64)
# inner_prods(model.no_python.X, model.no_python.fit_intercept, w, inner_products)
#
# for j in range(w.size):
#     grad2[j] = model.no_python.grad_coordinate(j, inner_products)
#
# print('grad2: ', grad2)
#
# # exit(0)

# prox_old = ProxL2Sq(strength=1e-4)
# prox_old = ProxL2Sq(strength=1e-2)

prox = ProxL1(strength=1e-2)

steps = 0.5 * n_samples / col_squared_norm(model)

step = 1 / model.lip_max
max_iter = 50

# Find a first good minimizer
svrg = SVRG(step=step, max_iter=3 * max_iter).set(model, prox)
w = w_start.copy()
w = svrg.solve(w)
obj_opt = model.loss(w) + prox.value(w)


solvers = [
    CGD(steps=steps, max_iter=max_iter).set(model, prox),
    SVRG(step=step, max_iter=max_iter).set(model, prox),
]

for solver in solvers:
    w = w_start.copy()
    solver.solve(w)

labels = ["CGD", "SVRG"]


plot_history(
    solvers,
    x="epoch",
    y="obj",
    labels=labels,
    rendering="bokeh",
    log_scale=True,
    dist_min=obj_opt,
)
