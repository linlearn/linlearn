
import numpy as np

from numpy.random.mtrand import multivariate_normal
from scipy.linalg import toeplitz
from scipy.optimize import check_grad

# from linlearn.model.logistic import LeastSquares
# from linlearn.model.linear import Features, loss_sample, loss_batch

from linlearn.model import LeastSquares

from linlearn.prox import ProxL2Sq
from linlearn.solver import SVRG, History

from time import sleep

np.set_printoptions(precision=2)




n_samples = 1_000
epoch_size = n_samples
n_features = 50
fit_intercept = True

coef0 = np.random.randn(n_features)
intercept0 = -2.

cov = toeplitz(0.5 ** np.arange(0, n_features))
X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)


y = X.dot(coef0) + 0.1 * np.random.randn(n_samples)
if fit_intercept:
    y += intercept0

if fit_intercept:
    w = np.zeros(n_features + 1)
else:
    w = np.zeros(n_features)

if fit_intercept:
    lip_max = (X ** 2).sum(axis=1).max() + 1
else:
    lip_max = (X ** 2).sum(axis=1).max()

# print("shape: ", (X ** 2).sum(axis=1).shape)

step = 1 / lip_max

print("step: ", step)


model = LeastSquares(fit_intercept).set(X, y)

prox = ProxL2Sq(strength=0.)

max_epochs = 10

svrg = SVRG(step=step, verbose=True).set(model=model, prox=prox)

svrg.solve(w)

print(svrg.history.values)

sleep(1)

svrg.history.print()

if fit_intercept:
    print(intercept0, coef0)
    print(w)
else:
    print(coef0)
    print(w)



# @njit
# def f():
#
#     linear_regression = LinearRegression(intercept)
#     linear_regression.fit(X, y)
#
#     # inner_prod(X[1], w, intercept)
#
#     i = 1
#
#     linear_regression.inner_prod(i, w)
#     linear_regression.inner_prods(w)
#     linear_regression.inner_prod(i, w)
#
#     linear_regression.inner_prods(w)
#
#     out = np.empty(y.shape)
#     linear_regression.inner_prods(w, out)
#
#     linear_regression.loss_sample(i, w)
#     linear_regression.loss_batch(w)
#
#     linear_regression.grad_sample_coef(i, w)
#     linear_regression.grad_sample(i, w)
#
#     out = np.empty(w.shape)
#     linear_regression.grad_sample(i, w, out)
#
#     linear_regression.grad_batch(w)
#
#     out = np.empty(w.shape)
#     linear_regression.grad_batch(w, out)
#
#
# f()

# linear_regression = LinearRegression(True)
# linear_regression.fit(X, y)
# print(linear_regression.fit_intercept)
# print(linear_regression.X)
# print(linear_regression.n_samples, linear_regression.n_features)

# err = check_grad(
#     lambda w: loss_batch(X, y, w, intercept=intercept),
#     lambda w: grad_batch(X, y, w, intercept=intercept),
#     w
# )

# err = check_grad(loss, grad, w)
#
#
# print('err: ', err)
#
#
# L = svd(X.T.dot(X) / n_samples, compute_uv=False)[0]
#
# step = 1 / L

# max_epochs = 10
#
#
# lr = LeastSquares(intercept).fit(X, y)
#
# # lr.fit(X, y)
# # gd(lr, w, max_epochs, step)
#
# step = 1e-2
#
# svrg(lr, w, max_epochs, step)
#
# if intercept:
#     print(w[0])
#     print(b0)
#
#     print(w[1:])
#     print(w0)
# else:
#     print(w)
#     print(w0)
#
# svrg(lr, w, max_epochs, step)

#
# gd(loss_batch, grad_batch, w, max_epochs, step)

# callback = Inspector(verbose=False)
#
# callback.update(1.)
# callback.update(2.)
# callback.update(3.)
#
# print(callback.objectives)

# callback = inspector(loss, max_epochs, verbose=True)
#
#
# t1 = time()
# w = gd(loss, grad, w, max_epochs, step)
# t2 = time()
# print("time: ", t2 - t1)

# w = gd(loss, grad, w, max_epochs, step)

#
# @njit
# def loss(w):
#     return loss_batch(X, y, w, intercept)
#
#
# @njit
# def grad(w, out):
#     return grad_batch(X, y, w, intercept, out)
#
#
# @njit
# def grad_sample(i, w, out):
#     xi = X[i]
#     yi = y[i]
#     grad_sample_coef(xi, yi, w, intercept, out)
#
#
# # loss(w)
# # grad(w)
#
# out = np.empty(10)
#
# grad_coef(1, w, out)


# #
# grad(w, out)
#
# print(out)

# step = 1e-2
#
# w = svrg(loss, grad, grad_coef, w, max_epochs, n_samples, step)


# t1 = time()
# w = gd(loss, grad, w, max_epochs, step)
# t2 = time()
#
# print("time: ", t2 - t1)
