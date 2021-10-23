from linlearn import Regressor
import numpy as np
import logging
import pickle
from sklearn.utils import check_array
from datetime import datetime
from sklearn.linear_model import HuberRegressor, RANSACRegressor, LinearRegression
from scipy.optimize import minimize
import scipy
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import os
from math import ceil
import time
import argparse
from noise_generators import gaussian, frechet, loglogistic, lognormal, weibull, student, pareto

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_directory("exp_archives/")
experiment_logfile = "exp_archives/corrupt_regression_exp.log"
experiment_name = "corrupt_regression"

file_handler = logging.FileHandler(filename=experiment_logfile)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

parser = argparse.ArgumentParser()

parser.add_argument("--random_seed", type=int, default=43)
parser.add_argument("--ransac_min_samples", type=int, default=5)
parser.add_argument("--n_repeats", type=int, default=10)
parser.add_argument("--huber_epsilon", type=float, default=1.1)
parser.add_argument("--fixed_epsilon", type=float, default=0.1)
parser.add_argument("--fixed_dim", type=float, default=15)
parser.add_argument("--noise_dist", type=str, default="gaussian", choices=["gaussian", "student", "weibull", "loglogistic", "lognormal", "pareto"])
parser.add_argument('--X_centered', dest='X_centered', action='store_true')
parser.add_argument('--X_not_centered', dest='X_centered', action='store_false')
parser.set_defaults(X_centered=True)
parser.add_argument('--save_results', dest='save_results', action='store_true')
parser.set_defaults(save_results=False)

args = parser.parse_args()

save_results = args.save_results

logging.info(48 * "=")
logging.info("Running new experiment session")
logging.info(48 * "=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = args.n_repeats

Huber_epsilon = args.huber_epsilon
RANSAC_min_samples = args.ransac_min_samples
random_seed = args.random_seed

fit_intercept = False

noise_sigma = {
    "gaussian": 20,
    "lognormal": 1.75,
    "pareto": 30,
    "student": 20,
    "weibull": 20,
    "frechet": 10,
    "loglogistic": 10,
}

X_centered = args.X_centered

w_star_dist = "uniform"
noise_dist = args.noise_dist

logging.info(
    "Lauching experiment with parameters : \n %r \n w_star_dist = %s , noise_dist = %s , sigma = %f"
    % (args, w_star_dist, noise_dist, noise_sigma[noise_dist])
)

epsilon_vals = np.linspace(0.05, 0.2, 6)
dims = np.arange(5, 50, 10)

fixed_epsilon = args.fixed_epsilon
fixed_dim = args.fixed_dim

logging.info("epsilon values = %r , dims = %r \nfixed_epsilon = %.2f , fixed_dim = %d"%(epsilon_vals, dims, fixed_epsilon, fixed_epsilon))

rng = np.random.RandomState(random_seed)  ## Global random generator


def gen_w_star(d, dist="normal"):
    if dist == "normal":
        return rng.multivariate_normal(np.zeros(d), np.eye(d)).reshape(d)
    elif dist == "uniform":
        return (10 * rng.uniform(size=d) - 5)/10
    else:
        raise Exception("Unknown w_star distribution")

def corrupt_data(X, y, epsilon=fixed_epsilon):
    n_samples, n_features = X.shape
    n_corrupted = int(n_samples * epsilon)
    indices = np.random.permutation(n_samples)[:n_corrupted]
    max_Sigma_X = 10
    max_y = np.max(np.abs(y))
    for i in indices:
        type = np.random.randint(6)
        dir = np.random.randn(n_features)
        dir /= np.sqrt((dir * dir).sum())  # random direction
        if type == 0:
            X[i] = max_Sigma_X * np.ones(n_features)
            y[i] = 10 * max_y

        elif type == 1:
            X[i] = 2 * max_Sigma_X * dir + np.random.randn(n_features)
            y[i] = 10 * max_y

        elif type == 2:
            X[i] = max_Sigma_X * np.ones(n_features)
            y[i] = 10 * (2 * np.random.randint(2) - 1) * max_y

        elif type == 3:
            X[i] = np.ones(n_features)
            y[i] = 1
        elif type == 4:
            X[i] = 10 * max_Sigma_X * dir + np.random.randn(n_features)
            y[i] = np.random.randint(2)
        elif type == 5:
            X[i] = np.random.randn(n_features)
            X[i] = 10 * max_Sigma_X * X[i] / np.linalg.norm(X[i])
            y[i] = 10 * max_y * (
                    (2 * np.random.randint(2) - 1)
                    + (2 * np.random.rand() - 1) / 5
            )
    return indices


col_try, col_time, col_algo, col_error, col_epsilon, col_dim = [], [], [], [], [], []

if noise_dist == "gaussian":
    noise_fct = gaussian
elif noise_dist == "lognormal":
    noise_fct = lognormal
elif noise_dist == "pareto":
    noise_fct = pareto
elif noise_dist == "student":
    noise_fct = student
elif noise_dist == "weibull":
    noise_fct = weibull
elif noise_dist == "frechet":
    noise_fct = frechet
elif noise_dist == "loglogistic":
    np.random.seed(seed=random_seed)
    noise_fct = loglogistic
else:
    raise Exception("unknown noise dist")


def gen_data(n_features, corruption_epsilon):
    n_samples = int((1/corruption_epsilon**2)*n_features)
    Sigma_X = np.diag(np.random.randint(10, size=n_features) + 1)
    mu_X = np.zeros(n_features) if X_centered else np.ones(n_features)

    X = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples)

    w_star = gen_w_star(n_features, dist=w_star_dist)
    noise, expect_noise, noise_2nd_moment = noise_fct(
        rng, n_samples, noise_sigma[noise_dist]
    )
    y = X @ w_star + noise

    corrupt_data(X, y, corruption_epsilon)

    return X, y, w_star


C5 = 0.01
C = lambda p:C5
print("WARNING : importing implementation of outlier robust gradient by (Prasad et al.) with arbitrary constant C(p)=%.2f"%C5)

def SSI(samples, subset_cardinality):
    """original name of this function is smallest_subset_interval"""
    if subset_cardinality < 2:
        raise ValueError("subset_cardinality must be at least 2")
    sorted_array = np.sort(samples)
    differences = sorted_array[subset_cardinality - 1:] - sorted_array[:-subset_cardinality + 1]
    argmin = np.argmin(differences)
    return sorted_array[argmin:argmin + subset_cardinality]


def alg2(X, eps, delta=0.01):
    # from Prasad et al. 2018

    X_tilde = alg4(X, eps, delta)

    n, p = X_tilde.shape

    if p == 1:
        return np.mean(X_tilde)

    S = np.cov(X.T)

    _, V = scipy.linalg.eigh(S)
    PW = V[:, :p // 2] @ V[:, :p // 2].T

    est1 = np.mean(X_tilde @ PW, axis=0, keepdims=True)

    QV = V[:, p // 2:]
    est2 = alg2(X_tilde @ QV, eps, delta)
    est2 = QV.dot(est2.T)
    est2 = est2.reshape((1, p))
    est = est1 + est2

    return est


def alg4(X, eps, delta=0.01):
    # from Prasad et al. 2018
    n, p = X.shape
    if p == 1:
        X_tilde = SSI(X.flatten(), max(2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))))
        return X_tilde[:, np.newaxis]

    a = np.array([alg2(X[:, i:i + 1], eps, delta / p) for i in range(p)])
    dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
    asort = np.argsort(dists)
    X_tilde = X[asort[:ceil(n * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n)) * (1 - eps))], :]
    return X_tilde

def LAD_regression(X, y, eps):

    empirical_risk = lambda w : np.abs(X @ w - y).mean()

    def empirical_gradient(w):
        return (np.sign(X @ w - y) @ X)/(X.shape[0])
    t0 = time.time()
    opt = minimize(empirical_risk, np.zeros(X.shape[1]), jac=empirical_gradient)
    fit_time = time.time() - t0
    return opt.x, fit_time

def linlearn_tmean(X, y, eps):
    reg = Regressor(loss="leastsquares", estimator="tmean", fit_intercept=False, solver="cgd", cgd_IS=True, percentage=2*eps + np.log(1/0.001)/X.shape[0])
    # t0 = time.time()
    reg.fit(X, y, dummy_first_step=True)
    # fit_time = time.time() - t0
    # time_record = reg.history_.records[1].record
    # fit_time = time_record[reg.history_.n_updates-1] - time_record[0]
    return reg.coef_, reg.fit_time()

def Huber(X, y, eps):

    reg = HuberRegressor(fit_intercept=False, epsilon=Huber_epsilon)
    t0 = time.time()
    reg.fit(X, y)
    fit_time = time.time() - t0

    return reg.coef_, fit_time

def RANSAC(X, y, eps):

    reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False), min_samples=RANSAC_min_samples)
    t0 = time.time()
    reg.fit(X, y)
    fit_time = time.time() - t0

    return reg.estimator_.coef_, fit_time

def Hubergradient(X, y, eps):
    delta = 0.01
    def grad(w):
        sample_gradients = np.multiply((X @ w - y)[:, np.newaxis], X)

        return alg2(sample_gradients, 2*eps, delta)[0]

    t0 = time.time()
    opt = minimize(lambda w : np.linalg.norm(grad(w)), np.zeros(X.shape[1]), jac=grad)
    fit_time = time.time() - t0
    return opt.x, fit_time


algorithms = [linlearn_tmean, LAD_regression, Huber, RANSAC, Hubergradient]

for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d/%d" % (rep + 1, n_repeats))
    logging.info(64 * "-")

    def parameter_error(w):
        return np.sqrt(((w - w_star)**2).sum())

    for eps in epsilon_vals:
        X, y, w_star = gen_data(fixed_dim, eps)
        check_array(y, ensure_2d=False)
        for algo in algorithms:
            param, fit_time = algo(X, y, eps)
            col_try.append(rep)
            col_algo.append(algo.__name__)
            col_error.append(parameter_error(param))
            col_dim.append(fixed_dim)
            col_epsilon.append(eps)
            col_time.append(fit_time)
    for dim in dims:
        print(dim)
        X, y, w_star = gen_data(dim, fixed_epsilon)

        for algo in algorithms:
            param, fit_time = algo(X, y, fixed_epsilon)
            col_try.append(rep)
            col_algo.append(algo.__name__)
            col_error.append(parameter_error(param))
            col_dim.append(dim)
            col_epsilon.append(fixed_epsilon)
            col_time.append(fit_time)

    logging.info("repetition done")

logging.info("Creating pandas DataFrame")
data = pd.DataFrame(
    {
        "fit_time": col_time,
        "repeat": col_try,
        "algo": col_algo,
        "error": col_error,
        "dim" : col_dim,
        "epsilon" : col_epsilon,
    }
)

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    ensure_directory("exp_archives/" + experiment_name + "/")
    import subprocess

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = experiment_name + "_results_" + now + ".pickle"
    with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "commit": commit, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,6))

sns.lineplot(x="dim", y="error", hue="algo", ax=ax1, data=data.query("epsilon == %f"%fixed_epsilon), legend=False)

sns.lineplot(x="epsilon", y="error", hue="algo", ax=ax2, data=data.query("dim == %d"%fixed_dim), legend=False)

sns.lineplot(x="dim", y="fit_time", hue="algo", ax=ax3, data=data.query("epsilon == %f"%fixed_epsilon), legend="brief")

# g.set_titles(col_template="{col_name}")

# axes = g.axes.flatten()
# axes[0].set_title("Excess empirical risk")
# axes[1].set_title("Excess risk")

# color_palette = []
# for line in axes[0].get_lines():
#     color_palette.append(line.get_c())
# color_palette = color_palette[: len(algorithms) + 1]


# axes[1].legend(
# plt.legend(
#     list(data["algo"].unique()),
#     # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
#     loc="upper right",#"lower left",  #
#     ncol=2,
#     borderaxespad=0.2,
#     columnspacing=1.0,
#     fontsize=10,
# )


# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle(
#     "n=%d , noise=%s , $\\sigma$ = %.2f, block_size=%.2f, w_star_dist=%s , outliers=%r , X_centered=%r"
#     % (
#         n_samples,
#         noise_dist,
#         noise_sigma[noise_dist],
#         MOMreg_block_size,
#         w_star_dist,
#         outliers,
#         X_centered,
#     )
# )

fig.tight_layout()
plt.show()

# save figure
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
specs = "n_rep%d_%s%.2f_w_dist=%s" % (
    n_repeats,
    noise_dist,
    noise_sigma[noise_dist],
    w_star_dist,
)
ensure_directory("exp_archives/" + experiment_name + "/")

fig_file_name = "exp_archives/" + experiment_name + "/" + specs + now + ".pdf"
fig.savefig(fname=fig_file_name, bbox_inches="tight")
logging.info("Saved figure into file : %s" % fig_file_name)
