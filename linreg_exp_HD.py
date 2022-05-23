from linlearn import Regressor
import numpy as np
import logging
import pickle
from datetime import datetime
from sklearn import linear_model
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from liuliu19 import liuliu19_solver
from liuliu18 import liuliu18_solver
from scipy.stats import multivariate_t
from noise_generators import (
    gaussian,
    frechet,
    loglogistic,
    lognormal,
    weibull,
    student,
    pareto,
)
import argparse
from numba import jit, objmode

NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
FASTMATH = True
PARALLEL = False

jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_directory("exp_archives/")
experiment_logfile = "exp_archives/linreg_exp_HD.log"
experiment_name = "linreg_HD"


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

parser.add_argument("--n_samples", type=int, default=500)
parser.add_argument("--n_features", type=int, default=5000)
parser.add_argument("--sparsity", type=int, default=40)
parser.add_argument("--sparsity_ub", type=int, default=50)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--n_repeats", type=int, default=5)
parser.add_argument("--outlier_types", nargs="+", type=int, default=[])
parser.add_argument("--max_iter", type=int, default=520)
parser.add_argument("--step_size", type=float, default=0.1)
parser.add_argument("--stage_length", type=int, default=40)
parser.add_argument("--confidence", type=float, default=0.01)
parser.add_argument("--corruption_rate", type=float, default=0.0)
parser.add_argument(
    "--noise_dist",
    type=str,
    default="student",
    choices=["gaussian", "student", "weibull", "loglogistic", "lognormal", "pareto"],
)
parser.add_argument("--X_centered", dest="X_centered", action="store_true")
parser.add_argument("--X_not_centered", dest="X_centered", action="store_false")
parser.set_defaults(X_centered=True)
parser.add_argument("--save_results", dest="save_results", action="store_true")
parser.set_defaults(save_results=False)
args = parser.parse_args()


logging.info(48 * "=")
logging.info("Running new experiment session")
logging.info(48 * "=")

n_repeats = args.n_repeats
n_samples = args.n_samples
sparsity = args.sparsity
sparsity_ub = args.sparsity_ub
n_features = args.n_features
save_results = args.save_results
corruption_rate = args.corruption_rate

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

fit_intercept = False

confidence = args.confidence
random_seed = args.random_seed

percentage = np.log(4 / confidence) / n_samples + corruption_rate

# print(1 / (4 * (corruption_rate * n_samples)))
logging.info("percentage is %.2f" % percentage)

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
noise_dist = args.noise_dist
step_size = args.step_size
stage_length = args.stage_length
T = args.max_iter
outlier_types = args.outlier_types

w_star_dist = "uniform"

logging.info("Lauching experiment with parameters : \n %r" % args)

# logging.info("mu_X = %r , Sigma_X = %r" % (mu_X, Sigma_X))
logging.info(
    "w_star_dist = %s , noise_dist = %s , sigma = %f"
    % (w_star_dist, noise_dist, noise_sigma[noise_dist])
)

rng = np.random.RandomState(random_seed)  ## Global random generator

min_Sigma = 1.0
max_Sigma = 10.0


def corrupt_data(X, y, types, corruption_rate):
    number = int(n_samples * corruption_rate)
    corrupted_indices = rng.choice(n_samples, size=number, replace=False)

    dir = rng.randn(n_features)
    dir /= np.sqrt((dir * dir).sum())  # random direction
    max_y = np.max(np.abs(y))
    for i in corrupted_indices:
        type = rng.choice(types)

        if type == 1:
            X[i, :] = np.max(Sigma_X) * np.ones(n_features)
            y[i] = 2 * max_y
        elif type == 2:
            X[i, :] = np.max(Sigma_X) * np.ones(n_features)
            y[i] = 2 * (2 * rng.randint(2) - 1) * max_y
        elif type == 3:
            X[i, :] = 100 * np.max(Sigma_X) * dir + rng.randn(n_features)
            y[i] = rng.randint(2)
        elif type == 4:
            vec = rng.randn(n_features)
            vec /= np.sqrt((vec * vec).sum())
            X[i, :] = 100 * np.max(Sigma_X) * vec

            y[i] = max_y * ((2 * rng.randint(2) - 1) + (2 * rng.rand() - 1) / 5)
        elif type == 5:
            X[i, :] = np.ones(n_features)
            y[i] = 1
        elif type == 6:
            X[i, :] = 2 * np.max(Sigma_X) * dir + rng.randn(n_features)
            y[i] = 2 * max_y
        else:
            raise Exception("Unknown outliers type")
    return corrupted_indices


def hardthresh(u, k):

    abs_u = np.abs(u)

    thresh = np.partition(abs_u, -k)[-k]
    for i in range(len(abs_u)):
        if abs_u[i] < thresh:
            u[i] = 0.0


def gen_w_star(d, dist="normal"):
    if dist == "normal":
        w = rng.randn(d)
    elif dist == "uniform":
        w = 10 * rng.uniform(size=d) - 5
    else:
        raise Exception("Unknown w_star distribution")
    hardthresh(w, sparsity)
    return w


def run_Lasso_CDA(X, y, param, max_iter, random_state=None):
    reg = linear_model.Lasso(
        alpha=param,
        max_iter=1,
        warm_start=True,
        fit_intercept=False,
        random_state=random_state,
    )
    coefs = np.zeros((max_iter + 1, X.shape[1]))

    for t in range(max_iter):
        with warnings.catch_warnings():  # silence the warnings about not enough iterations
            warnings.simplefilter("ignore")
            reg.fit(X, y)
        coefs[t + 1, :] = reg.coef_

    return coefs


@jit(**jit_kwargs)
def drv(y1, y2):
    return y1 - y2


def run_liuliu18(X, y, max_iter, theta_star, sigma, C_gamma=100.0, corrupt_lvl=0.0):
    ret = liuliu18_solver(
        X,
        y,
        step_size,
        sparsity_ub,
        max_iter,
        drv,
        theta_star,
        sigma,
        C_gamma=C_gamma,
        corrupt_lvl=corrupt_lvl,
    )

    return ret


def run_liuliu19(X, y, step_size, max_iter, random_seed=random_seed, estim="tmean"):
    # TODO : how to figure out step size ? it blows up if too much
    ret = liuliu19_solver(
        X,
        y,
        step_size,
        sparsity_ub,
        max_iter,
        drv,
        estim,
        random_seed,
        tm_alpha=percentage,
        only_last=False,
    )
    return ret


def run_MD(X, y, max_iter, step_size, sparsity_ub, stage_len):
    reg = Regressor(
        tol=0,
        max_iter=max_iter,
        solver="md",
        estimator="tmean",
        sparsity_ub=sparsity_ub,
        fit_intercept=fit_intercept,
        step_size=step_size,
        stage_length=stage_len,
        penalty="none",
        percentage=percentage,
        random_state=random_seed,
    )
    reg.fit(X, y)
    ret = np.squeeze(reg.history_.records[0].record)

    return ret


def run_DA(X, y, max_iter, step_size, sparsity_ub, stage_len):
    reg = Regressor(
        tol=0,
        max_iter=max_iter,
        solver="da",
        estimator="tmean",
        sparsity_ub=sparsity_ub,
        fit_intercept=fit_intercept,
        step_size=step_size,
        stage_length=stage_len,
        penalty="none",
        percentage=percentage,
        random_state=random_seed,
    )
    reg.fit(X, y)
    ret = np.squeeze(reg.history_.records[0].record)

    return ret


def run_linlearn_CD_lasso(X, y, max_iter, step_size, penalty_strength):
    reg = Regressor(
        tol=0,
        max_iter=max_iter,
        solver="cgd",
        loss="leastsquares",
        estimator="tmean",
        # sparsity_ub=sparsity_ub,
        fit_intercept=fit_intercept,
        step_size=step_size,
        # stage_length=stage_len,
        penalty="l1",
        percentage=percentage,
        C=penalty_strength,
        random_state=random_seed,
    )
    reg.fit(X, y)
    ret = np.squeeze(reg.history_.records[0].record)

    return ret


col_try, col_iter, col_algo, col_metric, col_val = [], [], [], [], []

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


def compute_objective_history(
    out, alg_name, metrics, rep, col_try, col_iter, col_algo, col_metric, col_val
):

    for ind_metric, metric in enumerate(metrics):
        for i in range(T):
            col_try.append(rep)
            col_algo.append(alg_name)
            col_metric.append(metric.__name__)
            col_val.append(metric(out[i]))
            col_iter.append(i)


for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d/%d" % (rep + 1, n_repeats))
    logging.info(64 * "-")

    logging.info("generating data ...")

    Sigma_X = rng.uniform(size=n_features) * (max_Sigma - min_Sigma) + min_Sigma
    mu_X = np.zeros(n_features) if X_centered else np.ones(n_features)

    # X = rng.randn(n_samples, n_features)
    X = multivariate_t.rvs(df=4.1, size=n_samples * n_features, random_state=rng).reshape((n_samples, n_features))
    for j in range(n_features):
        X[:, j] *= Sigma_X[j]
    X += mu_X[np.newaxis, :]
    # rng.multivariate_normal(mu_X, Sigma_X, size=n_samples)

    w_star = gen_w_star(n_features, dist=w_star_dist)
    noise, expect_noise, noise_2nd_moment = noise_fct(
        rng, n_samples, noise_sigma[noise_dist]
    )

    # noise -= expect_noise
    # expect_noise = 0.0

    y = X @ w_star + noise

    # outliers
    if len(outlier_types) > 0:
        corrupted_indices = corrupt_data(X, y, outlier_types, corruption_rate)
    else:
        corrupted_indices = []

    logging.info("generating risks and gradients ...")

    def predict_error(w):
        v = w - w_star
        return np.sqrt(np.einsum("i, i, i->", v, Sigma_X, v))
        # return np.sqrt(np.dot(v, Sigma_X @ v))

    def l2_error(w):
        return np.linalg.norm(w - w_star)

    def l1_error(w):
        return np.sum(np.abs(w - w_star))

    outputs = {}

    logging.info("Running algorithms ...")

    metrics = [l2_error, l1_error, predict_error]  #

    # out = run_liuliu18(X, y, T, w_star, noise_sigma[noise_dist], corrupt_lvl=corruption_rate)
    # compute_objective_history(out, "liuliu18tmean", metrics, rep, col_try, col_iter, col_algo, col_metric, col_val)

    out = run_MD(X, y, T, step_size * 3, sparsity_ub, stage_length)
    compute_objective_history(
        out,
        "linlearn_md",
        metrics,
        rep,
        col_try,
        col_iter,
        col_algo,
        col_metric,
        col_val,
    )
    logging.info("linlearn MD done")
    out = run_DA(X, y, T, step_size * 5, sparsity_ub, stage_length)
    compute_objective_history(
        out,
        "linlearn_da",
        metrics,
        rep,
        col_try,
        col_iter,
        col_algo,
        col_metric,
        col_val,
    )
    logging.info("linlearn DA done")
    out = run_linlearn_CD_lasso(
        X,
        y,
        T,
        1.0,
        1
        / (
            n_samples
            * 2
            * noise_sigma[noise_dist]
            * np.sqrt(2 * np.log(n_features) / n_samples)
        ),
    )
    compute_objective_history(
        out,
        "linlearn_CD_lasso",
        metrics,
        rep,
        col_try,
        col_iter,
        col_algo,
        col_metric,
        col_val,
    )
    logging.info("linlearn Lasso done")

    out = run_liuliu19(X, y, step_size / (20*max_Sigma), T, random_seed=random_seed, estim="mom")
    compute_objective_history(
        out,
        "liuliu19mom",
        metrics,
        rep,
        col_try,
        col_iter,
        col_algo,
        col_metric,
        col_val,
    )
    logging.info("Liuliu 19 MOM done")
    out = run_liuliu19(X, y, step_size / (20*max_Sigma), T, estim="tmean")
    compute_objective_history(
        out,
        "liuliu19tmean",
        metrics,
        rep,
        col_try,
        col_iter,
        col_algo,
        col_metric,
        col_val,
    )
    logging.info("Liuliu 19 TMean done")
    out = run_Lasso_CDA(
        X,
        y,
        2 * noise_sigma[noise_dist] * np.sqrt(2 * np.log(n_features) / n_samples),
        T,
        random_state=random_seed,
    )
    compute_objective_history(
        out, "lasso", metrics, rep, col_try, col_iter, col_algo, col_metric, col_val
    )
    logging.info("Sklearn Lasso done")

    logging.info("computing objective history")

    logging.info("repetition done")

logging.info("Creating pandas DataFrame")
data = pd.DataFrame(
    {
        "t": col_iter,
        "repeat": col_try,
        "algo": col_algo,
        "metric": col_metric,
        "value": col_val,
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

    specs = {
        "n_samples": n_samples,
        "n_rep": n_repeats,
        "noise": noise_dist,
        "sigma": noise_sigma[noise_dist],
        "w_star_dist": w_star_dist,
    }

    with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
        pickle.dump(
            {"datetime": now, "commit": commit, "results": data, "specs": specs}, f
        )

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.2

g = sns.FacetGrid(data, col="metric", height=4, sharey=False)  # , legend_out=True
g.map(sns.lineplot, "t", "value", "algo", lw=line_width, ci=None,).set(
    xlabel="", ylabel=""
)  # .set(yscale="log")
g.add_legend(loc="upper right")


# g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)

# ax = sns.lineplot(
#     x="t",
#     y="value",
#     hue="algo",
#     data=data.query("metric == '%s' & value <= 50" % metrics[0].__name__),
#     #legend=False,
#     lw=line_width,
#     ci=None,
# )


# color_palette = []
# for line in ax.get_lines():
#     color_palette.append(line.get_c())
# color_palette = color_palette[: len(algorithms) + 1]
#


plt.tight_layout()
plt.show()

# save figure
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

ensure_directory("exp_archives/" + experiment_name + "/")

fig_file_name = "exp_archives/" + experiment_name + "/linreg_HD_results_" + now + ".pdf"
# fig = ax.get_figure()
# fig.savefig(fname=fig_file_name, bbox_inches="tight")
g.savefig(fname=fig_file_name, bbox_inches="tight")
logging.info("Saved figure into file : %s" % fig_file_name)
