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


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_directory("exp_archives/")


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

logging.info(48 * "=")
logging.info("Running new experiment session")
logging.info(48 * "=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 10

n_samples = 500
n_features = 5

fit_intercept = False

block_sizes = {"mom_cgd": 0.02, "implicit_gd": 0.02, "gmom_gd": 0.02}  # 555555}

random_seed = 43

noise_sigma = {
    "gaussian": 20,
    "lognormal": 1.75,
    "pareto": 30,
    "student": 20,
    "weibull": 20,
    "frechet": 10,
    "loglogistic": 10,
}

X_centered = False # True #
Sigma_X = np.diag(np.arange(1, n_features + 1))
mu_X = np.zeros(n_features) if X_centered else np.ones(n_features)

w_star_dist = "uniform"
noise_dist = "gaussian"#"student"

step_size = 0.05
T = 150

outlier_types = [5]
outliers = False

logging.info(
    "Lauching experiment with parameters : \n n_repeats = %d , n_samples = %d , n_features = %d , outliers = %r"
    % (n_repeats, n_samples, n_features, outliers)
)
if outliers:
    logging.info("outlier types :  %r" % outlier_types)
logging.info("block sizes are : %r" % block_sizes)
logging.info(
    "random_seed = %d , mu_X = %r , Sigma_X = %r" % (random_seed, mu_X, Sigma_X)
)
logging.info(
    "w_star_dist = %s , noise_dist = %s , sigma = %f"
    % (w_star_dist, noise_dist, noise_sigma[noise_dist])
)
logging.info("step_size = %f , T = %d" % (step_size, T))

rng = np.random.RandomState(random_seed)  ## Global random generator


def generate_outliers(y, number=10, type=1):
    dir = np.random.randn(n_features)
    dir /= np.sqrt((dir * dir).sum())  # random direction
    if type == 0:
        X_out = np.max(Sigma_X) * np.ones((number, n_features))
        y_out = 2 * np.ones(number) * np.max(np.abs(y))
    elif type == 1:
        X_out = 2 * np.max(Sigma_X) * dir + np.random.randn(number, n_features)
        y_out = 2 * np.ones(number) * np.max(np.abs(y))
    elif type == 2:
        X_out = np.max(Sigma_X) * np.ones((number, n_features))
        y_out = 2 * (2 * np.random.randint(2, size=number) - 1) * np.max(np.abs(y))
    elif type == 3:
        X_out = np.ones((number, n_features))
        y_out = np.ones(number)
    elif type == 4:
        X_out = 100 * np.max(Sigma_X) * dir + np.random.randn(number, n_features)
        y_out = np.random.randint(2, size=number)
    elif type == 5:
        X_out = np.random.randn(number, n_features)
        X_out = (
            100 * np.max(Sigma_X) * np.diag(1 / np.linalg.norm(X_out, axis=1)) @ X_out
        )
        y_out = np.max(np.abs(y)) * (
            (2 * np.random.randint(2, size=number) - 1)
            + (2 * np.random.rand(number) - 1) / 5
        )
    else:
        raise Exception("Unknown outliers type")

    return X_out, y_out


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


def generate_student_noise_sample(n_samples, sigma=10, df=2.1):
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

algorithms = [
    Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=T),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=T),
    Algorithm(name="catoni_cgd", solver="cgd", estimator="ch", max_iter=T),
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean", max_iter=T),
    Algorithm(name="holland_gd", solver="gd", estimator="ch", max_iter=T),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=T),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm", max_iter=T),
    # Algorithm(name="implicit_cgd", solver="cgd", estimator="implicit", max_iter=T),
    # Algorithm(name="tmean_gd", solver="gd", estimator="tmean", max_iter=T),
    # Algorithm(name="mom_gd", solver="gd", estimator="mom", max_iter=T),
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
        for typ in outlier_types:
            X_out, y_out = generate_outliers(y, type=typ)
        X = np.concatenate((X, X_out), axis=0)
        y = np.concatenate((y, y_out))

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
    XXT_clean = X[:n_samples, :].T @ X[:n_samples, :]
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
            solver=algo.solver,
            estimator=algo.estimator,
            fit_intercept=fit_intercept,
            step_size=step_size,
            penalty="none",
            block_size=block_sizes[algo.name]
            if algo.name in block_sizes.keys()
            else 0.07,
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

g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
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
color_palette = color_palette[: len(algorithms) + 1]


# plt.legend(
axes[1].legend(
    list(data["algo"].unique()),
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="lower left",  # "upper right",
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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

zoom_on_excess_empirical_risk = False
zoom_on_excess_risk = False

if zoom_on_excess_risk:
    axins = inset_axes(axes[-1], "40%", "30%", loc="lower left", borderpad=1)

    sns.lineplot(
        x="t",
        y="value",
        hue="algo",
        lw=line_width,
        ci=None,
        data=data.query(
            "t >= %d and metric=='excess_risk' and algo!='oracle' and algo!='erm_gd'"
            % ((T * 4) // 5)
        ),
        ax=axins,
        legend=False,
        palette=color_palette[2:],
    ).set(
        yscale="log"
    )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#

    axes[-1].indicate_inset_zoom(axins, edgecolor="black")
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)

if zoom_on_excess_empirical_risk:
    axins0 = inset_axes(axes[0], "40%", "30%", loc="lower left", borderpad=1)

    sns.lineplot(
        x="t",
        y="value",
        hue="algo",
        lw=line_width,
        ci=None,
        data=data.query(
            "t >= %d and metric=='excess_empirical_risk' and algo!='erm_gd'"
            % ((T * 4) // 5)
        ),
        ax=axins0,
        legend=False,
        palette=color_palette[:1] + color_palette[2:],
    ).set(
        yscale="log"
    )  # , xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)#
    axes[0].indicate_inset_zoom(axins0, edgecolor="black")
    axins0.xaxis.set_visible(False)
    axins0.yaxis.set_visible(False)

plt.tight_layout()
plt.show()

if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    specs = "n%d_n_rep%d_%s%.2f_block_size=%.2f_w_dist=%s" % (
        n_samples,
        n_repeats,
        noise_dist,
        noise_sigma[noise_dist],
        block_sizes["mom_cgd"],
        w_star_dist,
    )
    ensure_directory("exp_archives/linreg/")

    fig_file_name = "exp_archives/linreg/" + specs + now + ".pdf"
    g.fig.savefig(fname=fig_file_name, bbox_inches="tight")
    logging.info("Saved figure into file : %s" % fig_file_name)
