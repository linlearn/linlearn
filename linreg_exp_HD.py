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
from collections import namedtuple
import os
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
parser.add_argument("--n_features", type=int, default=100000)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--n_repeats", type=int, default=5)
parser.add_argument("--outlier_types", nargs="+", type=int, default=[])
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--step_size", type=float, default=0.1)
parser.add_argument("--confidence", type=float, default=0.01)
parser.add_argument("--corruption_rate", type=float, default=0.0)
parser.add_argument(
    "--noise_dist",
    type=str,
    default="gaussian",
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
n_features = args.n_features
save_results = args.save_results
corruption_rate = args.corruption_rate

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

fit_intercept = False

confidence = args.confidence
random_seed = args.random_seed

percentage = np.log(4 / confidence) / n_samples + 2 * corruption_rate
block_size = 1 / (18 * np.log(1 / confidence))
llm_block_size = 1 / (4 * np.log(1 / confidence))
if corruption_rate > 0.0:
    block_size = min(block_size, 1 / (4 * (corruption_rate * n_samples)))
    llm_block_size = min(llm_block_size, 1 / (4 * (corruption_rate * n_samples)))
#print(1 / (4 * (corruption_rate * n_samples)))
logging.info("percentage is %.2f" % percentage)
logging.info("block size is :  %.2f" % block_size)

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
T = args.max_iter
outlier_types = args.outlier_types

Sigma_X = np.diag(np.arange(1, n_features + 1))
mu_X = np.zeros(n_features) if X_centered else np.ones(n_features)

w_star_dist = "uniform"

logging.info("Lauching experiment with parameters : \n %r" % args)

logging.info("mu_X = %r , Sigma_X = %r" % (mu_X, Sigma_X))
logging.info(
    "w_star_dist = %s , noise_dist = %s , sigma = %f"
    % (w_star_dist, noise_dist, noise_sigma[noise_dist])
)

rng = np.random.RandomState(random_seed)  ## Global random generator


def corrupt_data(X, y, types, corruption_rate):
    number = int((n_samples * corruption_rate) / len(types))
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
            X[i, :] = 10 * np.max(Sigma_X) * dir + rng.randn(n_features)
            y[i] = rng.randint(2)
        elif type == 4:
            vec = rng.randn(n_features)
            vec /= np.sqrt((vec * vec).sum())
            X[i, :] = 10 * np.max(Sigma_X) * vec

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


def gen_w_star(d, dist="normal"):
    if dist == "normal":
        return rng.multivariate_normal(np.zeros(d), np.eye(d)).reshape(d)
    elif dist == "uniform":
        return 10 * rng.uniform(size=d) - 5
    else:
        raise Exception("Unknown w_star distribution")


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


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator"])

algorithms = [
    Algorithm(name="erm_gd", solver="gd", estimator="erm"),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom"),
    Algorithm(name="catoni_cgd", solver="cgd", estimator="ch"),
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean"),
    Algorithm(name="holland_gd", solver="gd", estimator="ch"),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom"),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm"),
    Algorithm(name="hg_gd", solver="gd", estimator="hg"),
    # Algorithm(name="implicit_cgd", solver="cgd", estimator="implicit", max_iter=T),
    # Algorithm(name="tmean_gd", solver="gd", estimator="tmean", max_iter=T),
    # Algorithm(name="mom_gd", solver="gd", estimator="mom", max_iter=T),
]


for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d/%d" % (rep + 1, n_repeats))
    logging.info(64 * "-")

    logging.info("generating data ...")
    X = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples)

    w_star = gen_w_star(n_features, dist=w_star_dist)
    noise, expect_noise, noise_2nd_moment = noise_fct(
        rng, n_samples, noise_sigma[noise_dist]
    )

    noise -= expect_noise
    expect_noise = 0.0

    y = X @ w_star + noise

    # outliers
    if len(outlier_types) > 0:
        corrupted_indices = corrupt_data(X, y, outlier_types, corruption_rate)
    else:
        corrupted_indices = []

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

    # compute the Lipschitz constant for oracle GD without the outliers
    clean_index = list(set(range(n_samples)) - set(corrupted_indices))
    XXT_clean = X[clean_index, :].T @ X[clean_index, :]
    Lip = np.linalg.eigh(XXT_clean / n_samples)[0][-1]

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
            eps=confidence,
            solver=algo.solver,
            estimator=algo.estimator,
            fit_intercept=fit_intercept,
            step_size=step_size,  # *(5 if algo.estimator=="hg" else 1),
            penalty="none",
            block_size=llm_block_size if algo.estimator =="llm" else block_size,
            percentage=percentage,
            random_state=random_seed,
        )
        reg.fit(X, y)
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
    ensure_directory("exp_archives/" + experiment_name + "/")
    import subprocess

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = experiment_name + "_results_" + now + ".pickle"

    specs = {"n_samples": n_samples,
        "n_rep": n_repeats,
        "noise": noise_dist,
        "sigma": noise_sigma[noise_dist],
        "block_size": block_size,
        "w_star_dist": w_star_dist,
    }


    with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "commit": commit, "results": data, "specs": specs}, f)

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.2

# g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
# g.map(sns.lineplot, "t", "value", "algo", lw=line_width, ci=None,).set(
#     xlabel="", ylabel=""
# ).set(yscale="log")


# g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
ax = sns.lineplot(
    x="t",
    y="value",
    hue="algo",
    data=data.query("metric == '%s'" % metrics[1].__name__),
    legend=False,
    lw=line_width,
    ci=None,
)

ax.set(yscale="log")

plt.xlabel("")
plt.ylabel("")

# g.set_titles(col_template="{col_name}")

#axes = [ax]  # g.axes.flatten()
# axes[0].set_title("Excess empirical risk")
# axes[0].set_title("Excess risk")

color_palette = []
for line in ax.get_lines():
    color_palette.append(line.get_c())
color_palette = color_palette[: len(algorithms) + 1]


plt.legend(
    [
        "Oracle",
        "$\\mathtt{ERM}$ GD",
        "$\\mathtt{MOM}$ CGD",
        "$\\mathtt{CH}$ CGD",
        "$\\mathtt{TM}$ CGD",
        "$\\mathtt{CH}$ GD",
        "GMOM GD",
        "LLM GD",
        "HG GD",
    ],
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="lower left",#"upper right",
    ncol=2,
    borderaxespad=0.2,
    columnspacing=1.0,
    fontsize=10,
)
# axes[0].legend(
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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

zoom_on = False

if zoom_on:
    axins = inset_axes(ax, "40%", "30%", loc="lower left", borderpad=1)

    sns.lineplot(
        x="t",
        y="value",
        hue="algo",
        lw=line_width,
        ci=None,
        data=data.query(
            "t >= %d and metric=='excess_risk' and algo not in ['oracle', 'erm_gd']"
            % ((T * 4) // 5)
        ),
        ax=axins,
        legend=False,
        palette=color_palette[2:],
    ).set(
        yscale="log"
    )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#

    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)

# if zoom_on_excess_empirical_risk:
#     axins0 = inset_axes(axes[0], "40%", "30%", loc="lower left", borderpad=1)
#
#     sns.lineplot(
#         x="t",
#         y="value",
#         hue="algo",
#         lw=line_width,
#         ci=None,
#         data=data.query(
#             "t >= %d and metric=='excess_empirical_risk' and algo!='erm_gd'"
#             % ((T * 4) // 5)
#         ),
#         ax=axins0,
#         legend=False,
#         palette=color_palette[:1] + color_palette[2:],
#     ).set(
#         yscale="log"
#     )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#
#     axes[0].indicate_inset_zoom(axins0, edgecolor="black")
#     axins0.xaxis.set_visible(False)
#     axins0.yaxis.set_visible(False)

plt.tight_layout()
plt.show()

# save figure
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

ensure_directory("exp_archives/" + experiment_name + "/")

fig_file_name = "exp_archives/" + experiment_name + "/linreg_results_" + now + ".pdf"
fig = ax.get_figure()
fig.savefig(fname=fig_file_name, bbox_inches="tight")
logging.info("Saved figure into file : %s" % fig_file_name)
