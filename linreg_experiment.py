from linlearn import Regressor
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.optimize import minimize
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
<<<<<<< HEAD
import time
import os

=======
from collections import namedtuple
import os


>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

<<<<<<< HEAD
ensure_directory('exp_archives/')
=======

ensure_directory("exp_archives/")
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c


file_handler = logging.FileHandler(filename="exp_archives/linreg_exp.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

save_results = False
save_fig = True

logging.info(128 * "=")
logging.info("Running new experiment session")
logging.info(128 * "=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

<<<<<<< HEAD
n_repeats = 10
=======
n_repeats = 30
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c

n_samples = 500
n_features = 5

fit_intercept = False

n_outliers = 10
outliers = False

<<<<<<< HEAD
MOMreg_block_size = 0.02
=======
MOMreg_block_size = 0.07
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c

prasad_delta = 0.01

random_seed = 42

noise_sigma = {
    "gaussian": 20,
    "lognormal": 1.75,
    "pareto": 30,
    "student": 20,
    "weibull": 20,
    "frechet": 10,
    "loglogistic": 10,
}

X_centered = True
Sigma_X = np.diag(np.arange(1, n_features + 1))
mu_X = np.zeros(n_features) if X_centered else np.ones(n_features)

w_star_dist = "uniform"
<<<<<<< HEAD
noise_dist = "lognormal"

step_size = 0.05
T = 250
=======
noise_dist = "student"

step_size = 0.05
T = 150
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c

logging.info(
    "Lauching experiment with parameters : \n n_repeats = %d , n_samples = %d , n_features = %d , outliers = %r"
    % (n_repeats, n_samples, n_features, outliers)
)
if outliers:
    logging.info("n_outliers = %d" % n_outliers)
logging.info("block size for MOM_CGD is %f" % MOMreg_block_size)
logging.info(
<<<<<<< HEAD
    "random_seed = %d , mu_X = %r , Sigma_X = %r"
    % (random_seed, mu_X, Sigma_X)
=======
    "random_seed = %d , mu_X = %r , Sigma_X = %r" % (random_seed, mu_X, Sigma_X)
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c
)
logging.info(
    "w_star_dist = %s , noise_dist = %s , sigma = %f"
    % (w_star_dist, noise_dist, noise_sigma[noise_dist])
)
logging.info("step_size = %f , T = %d" % (step_size, T))

rng = np.random.RandomState(random_seed)  ## Global random generator

<<<<<<< HEAD
# def gmom(xs, tol=1e-7):
#     y = np.average(xs, axis=0)
#     eps = 1e-10
#     delta = 1
#     niter = 0
#     while delta > tol:
#         xsy = xs - y
#         dists = np.linalg.norm(xsy, axis=1)
#         inv_dists = 1 / dists
#         mask = dists < eps
#         inv_dists[mask] = 0
#         nb_too_close = (mask).sum()
#         ry = np.linalg.norm(np.dot(inv_dists, xsy))
#         cst = nb_too_close / ry
#         y_new = max(0, 1 - cst) * np.average(xs, axis=0, weights=inv_dists) + min(1, cst) * y
#         delta = np.linalg.norm(y - y_new)
#         y = y_new
#         niter += 1
#     # print(niter)
#     return y

=======
>>>>>>> db33ed04450be1678f384590a2526a9bda6b328c

def gen_w_star(d, dist="normal"):
    if dist == "normal":
        return rng.multivariate_normal(np.zeros(d), np.eye(d)).reshape(d)
    elif dist == "uniform":
        return 10 * rng.uniform(size=d) - 5
    else:
        raise Exception("Unknown w_star distribution")


def generate_gaussian_noise_sample(n_samples, sigma=20):
    noise = sigma * rng.normal(size=n_samples)
    expect_noise = 0
    noise_2nd_moment = sigma ** 2

    return noise, expect_noise, noise_2nd_moment


def generate_lognormal_noise_sample(n_samples, sigma=1.75):
    noise = rng.lognormal(0, sigma, n_samples)
    expect_noise = np.exp(0.5 * sigma ** 2)
    noise_2nd_moment = np.exp(2 * sigma ** 2)

    return noise, expect_noise, noise_2nd_moment


def generate_pareto_noise_sample(n_samples, sigma=10, pareto=2.05):
    noise = sigma * rng.pareto(pareto, n_samples)
    expect_noise = (sigma) / (pareto - 1)
    noise_2nd_moment = expect_noise ** 2 + (sigma ** 2) * pareto / (
        ((pareto - 1) ** 2) * (pareto - 2)
    )

    return noise, expect_noise, noise_2nd_moment


def generate_student_noise_sample(n_samples, sigma=10, df=2.2):
    noise = sigma * rng.standard_t(df, n_samples)
    expect_noise = 0
    noise_2nd_moment = expect_noise ** 2 + (sigma ** 2) * df / (df - 2)

    return noise, expect_noise, noise_2nd_moment


def generate_weibull_noise_sample(n_samples, sigma=10, a=0.65):
    from scipy.special import gamma

    noise = sigma * rng.weibull(a, n_samples)
    expect_noise = sigma * gamma(1 + 1 / a)
    noise_2nd_moment = (sigma ** 2) * gamma(1 + 2 / a)

    return noise, expect_noise, noise_2nd_moment


def generate_frechet_noise_sample(n_samples, sigma=10, alpha=2.2):
    from scipy.special import gamma

    noise = sigma * (1 / rng.weibull(alpha, n_samples))
    expect_noise = sigma * gamma(1 - 1 / alpha)
    noise_2nd_moment = (sigma ** 2) * gamma(1 - 2 / alpha)

    return noise, expect_noise, noise_2nd_moment


def generate_loglogistic_noise_sample(n_samples, sigma=10, c=2.2):
    from scipy.stats import fisk

    noise = sigma * fisk.rvs(c, size=n_samples)
    expect_noise = sigma * (np.pi / c) / np.sin(np.pi / c)
    noise_2nd_moment = (sigma ** 2) * (2 * np.pi / c) / np.sin(2 * np.pi / c)

    return noise, expect_noise, noise_2nd_moment


def gradient_descent(funs_to_track, x0, grad, step_size, T):
    """run gradient descent for given gradient grad and step size for T steps"""
    x = x0
    tracks = [np.zeros(T) for i in range(len(funs_to_track))]
    for t in range(T):
        for i, f in enumerate(funs_to_track):
            tracks[i][t] = f(x)
        grad_x = grad(x)
        # grad_error = np.linalg.norm(grad_x - true_gradient(x))
        x -= step_size * grad_x
        # tracks[len(funs_to_track)][t] = grad_error
    return tracks


col_try, col_time, col_algo, col_metric, col_val = [], [], [], [], []

if noise_dist == "gaussian":
    noise_fct = generate_gaussian_noise_sample
elif noise_dist == "lognormal":
    noise_fct = generate_lognormal_noise_sample
elif noise_dist == "pareto":
    noise_fct = generate_pareto_noise_sample
elif noise_dist == "student":
    noise_fct = generate_student_noise_sample
elif noise_dist == "weibull":
    noise_fct = generate_weibull_noise_sample
elif noise_dist == "frechet":
    noise_fct = generate_frechet_noise_sample
elif noise_dist == "loglogistic":
    np.random.seed(seed=random_seed)
    noise_fct = generate_loglogistic_noise_sample
else:
    raise Exception("unknown noise dist")

trackers = [(lambda w: w, (n_features + int(fit_intercept),))]

Algorithm = StateHollandCatoni = namedtuple(
    "Algorithm", ["name", "solver", "estimator", "max_iter"]
)

algorithms = [Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=T),
              Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=T),
              Algorithm(name="catoni_cgd", solver="cgd", estimator="holland_catoni", max_iter=T),
              Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean", max_iter=T),
              Algorithm(name="holland_gd", solver="gd", estimator="holland_catoni", max_iter=T),
              Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=T),
              Algorithm(name="implicit_gd", solver="gd", estimator="implicit", max_iter=T),
              #Algorithm(name="implicit_cgd", solver="cgd", estimator="implicit", max_iter=T),
              #Algorithm(name="tmean_gd", solver="gd", estimator="tmean", max_iter=T),
              #Algorithm(name="mom_gd", solver="gd", estimator="mom", max_iter=T),
              ]


for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d" % (rep + 1))
    logging.info(64 * "-")

    logging.info("generating data ...")
    X = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples)

    w_star = gen_w_star(n_features, dist=w_star_dist)
    noise, expect_noise, noise_2nd_moment = noise_fct(
        n_samples, noise_sigma[noise_dist]
    )
    y = X @ w_star + noise

    # outliers
    if outliers:
        X = np.concatenate(
            (X, np.max(Sigma_X) * np.ones((n_outliers, n_features))), axis=0
        )
        # y = np.concatenate((y, 10*np.ones(n_outliers)*np.max(np.abs(y))))
        y = np.concatenate(
            (
                y,
                2
                * (2 * np.random.randint(2, size=n_outliers) - 1)  # np.ones(n_outliers)
                * np.max(np.abs(y)),
            )
        )

    logging.info("generating risks and gradients ...")

    def empirical_risk(w):
        return ((X.dot(w) - y) ** 2).mean() / 2

    def true_risk(w):
        return 0.5 * (
            noise_2nd_moment
            + np.dot(mu_X, w - w_star) ** 2
            - 2 * expect_noise * np.dot(mu_X, w - w_star)
            + np.dot(w - w_star, Sigma_X @ (w - w_star))
        )

    def true_gradient(w):
        return (
            Sigma_X @ (w - w_star) + (-expect_noise + np.dot(mu_X, w - w_star)) * mu_X
        )

    XXT = X.T @ X
    Xy = X.T @ y

    Lip = np.linalg.eigh(XXT / X.shape[0])[0][-1]

    def empirical_gradient(w):
        return (XXT @ w - Xy) / n_samples

    # optimal_risk = true_risk(w_star)
    optimal_risk = minimize(true_risk, np.zeros(n_features), jac=true_gradient).fun
    optimal_empirical_risk = minimize(
        empirical_risk, np.zeros(n_features), jac=empirical_gradient
    ).fun

    def excess_empirical_risk(w):
        return empirical_risk(w.flatten()) - optimal_empirical_risk

    def excess_risk(w):
        return true_risk(w.flatten()) - optimal_risk

    outputs = {}

    logging.info("Running algorithms ...")

    metrics = [excess_empirical_risk, excess_risk]  # , "gradient_error"]

    def run_algorithm(algo, out):
        reg = Regressor(
            tol=0,
            max_iter=T,
            solver=algo.solver,
            estimator=algo.estimator,
            fit_intercept=fit_intercept,
            step_size=step_size,
            penalty="none",
        )
        reg.fit(X, y, trackers=trackers)
        out[algo.name] = reg.history_.records

    for algo in algorithms:
        run_algorithm(algo, outputs)

    oracle_output = gradient_descent(
        [excess_empirical_risk, excess_risk],
        np.zeros(n_features),
        true_gradient,
        step_size / Lip,
        T,
    )

    for tt in range(T):
        for ind_metric, metric in enumerate(metrics):
            col_try.append(rep)
            col_algo.append("oracle")
            col_metric.append(metric.__name__)
            col_val.append(oracle_output[ind_metric][tt])
            col_time.append(tt)

    logging.info("computing objective history")
    for alg in outputs.keys():
        for ind_metric, metric in enumerate(metrics):
            for i in range(T):
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric.__name__)
                col_val.append(metric(outputs[alg][0].record[i]))
                col_time.append(
                    i
                )  # outputs[alg][1].record[i] - outputs[alg][1].record[0])
    logging.info("repetition done")

logging.info("Creating pandas DataFrame")
data = pd.DataFrame(
    {
        "t": col_time,
        "repeat": col_try,
        "algo": col_algo,
        "metric": col_metric,
        "value": col_val,
    }
)

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    ensure_directory("exp_archives/linreg")
    import subprocess
    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "linreg_results_" + now + ".pickle"
    with open("exp_archives/linreg/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "commit": commit, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.0

g = sns.FacetGrid(data, col="metric", height=4, legend_out=True)
g.map(sns.lineplot, "t", "value", "algo", lw=line_width, ci=None,).set(
    xlabel="", ylabel=""
).set(yscale="log")

# g.set_titles(col_template="{col_name}")

axes = g.axes.flatten()
axes[0].set_title("Excess empirical risk")
axes[1].set_title("Excess risk")

color_palette = []
for line in axes[0].get_lines():
    color_palette.append(line.get_c())
color_palette = color_palette[:7]

# code_names = {
#     "ERM": "erm",
#     "empirical_gradient": "erm",
#     "Holland_gradient": "holland",
#     "true_gradient": "oracle",
#     "mom_cgd": "mom_cgd",
#     "Prasad_HeavyTail_gradient": "gmom_grad",
#     "Lecue_gradient": "implicit",
#     "catoni_cgd": "catoni_cgd",
#     "tmean_cgd": "tmean_cgd",
#     "Prasad_outliers_gradient": "Prasad_outliers_gradient",
# }

# plt.legend(
axes[0].legend(
    list(
        data["algo"].unique()
    ),  # [code_names[name] for name in data["algo"].unique()],
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="lower left",
    ncol=2,
    borderaxespad=0.2,
    columnspacing=1.0,
    fontsize=10,
)
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

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# axins = inset_axes(axes[-1], "40%", "30%", loc="lower left", borderpad=1)
#
# sns.lineplot(
#     x="t",
#     y="value",
#     hue="algo",
#     lw=line_width,
#     ci=None,
#     data=data.query(
#         "t >= %d and metric=='excess_risk' and algo!='true_gradient'" % ((T * 4) // 5)
#     ),
#     ax=axins,
#     legend=False,
#     palette=color_palette[:-1],
# ).set(
#     yscale="log"
# )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#
#
#
# axes[-1].indicate_inset_zoom(axins, edgecolor="black")
# axins.xaxis.set_visible(False)
# axins.yaxis.set_visible(False)
#
# axins0 = inset_axes(axes[0], "40%", "30%", loc="lower left", borderpad=1)
#
# sns.lineplot(
#     x="t",
#     y="value",
#     hue="algo",
#     lw=line_width,
#     ci=None,
#     data=data.query(
#         "t >= %d and metric=='excess_empirical_risk' and algo!='empirical_gradient'"
#         % ((T * 4) // 5)
#     ),
#     ax=axins0,
#     legend=False,
#     palette=color_palette[1:],
# ).set(
#     yscale="log"
# )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#
# axes[0].indicate_inset_zoom(axins0, edgecolor="black")
# axins0.xaxis.set_visible(False)
# axins0.yaxis.set_visible(False)

# for i, dataset in enumerate(df["dataset"].unique()):
#     axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
#     axes[i].set_title(dataset, fontsize=18)

plt.tight_layout()
plt.show()

if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    specs = "n%d_n_rep%d_%s%.2f_block_size=%.2f_w_dist=%s" % (
        n_samples,
        n_repeats,
        noise_dist,
        noise_sigma[noise_dist],
        MOMreg_block_size,
        w_star_dist,
    )
    ensure_directory("exp_archives/linreg/")

    fig_file_name = "exp_archives/linreg/" + specs + now + ".pdf"
    g.fig.savefig(fname=fig_file_name, bbox_inches="tight")
    logging.info("Saved figure into file : %s" % fig_file_name)
