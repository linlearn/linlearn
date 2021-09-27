from linlearn import Regressor
import numpy as np
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys, os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time
from collections import namedtuple

file_handler = logging.FileHandler(filename="exp_archives/linreg_realdata.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


save_results = False
save_fig = True

logging.info(128 * "=")
logging.info("Running new experiment session")
logging.info(128 * "=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 1

fit_intercept = True

use_std_scaler = False

n_outliers = 10
outliers = False

block_sizes = {"mom_cgd": 0.07, "implicit_gd": 0.07, "gmom_gd": 0.07}  # 555555}

random_seed = 44
test_size = 0.3

dataset = "used_car"  # "CaliforniaHousing"#"Boston"#"Diabetes"  #


def load_used_car():

    csv = pd.read_csv("used_car/vw.csv")
    csv["year"] = 2020 - csv["year"]
    csv = csv.drop("model", axis=1)
    categoricals = ["transmission", "fuelType"]  # , "model"]
    label = "price"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv[cat], prefix=cat)
        csv = csv.drop(cat, axis=1)
        csv = csv.join(one_hot)
    csv = pd.DataFrame(StandardScaler().fit_transform(csv), columns=csv.columns)
    df_train, df_test = train_test_split(
        csv,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def load_data_boston():
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test


def load_california_housing():
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test


def fetch_diabetes():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test


if dataset == "Boston":
    X_train, X_test, y_train, y_test = load_data_boston()
elif dataset == "CaliforniaHousing":
    X_train, X_test, y_train, y_test = load_california_housing()
elif dataset == "Diabetes":
    X_train, X_test, y_train, y_test = fetch_diabetes()
elif dataset == "used_car":
    X_train, X_test, y_train, y_test = load_used_car()
else:
    raise ValueError("Unknown dataset")

if use_std_scaler:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.fit_transform(X_test)

for dat in [X_train, X_test, y_train, y_test]:
    dat = np.ascontiguousarray(dat)

n_samples, n_features = X_train.shape

step_size = 0.05
max_iter = 2 * 50


def risk(X, y, w, fit_intercept=fit_intercept, meantype="ordinary"):
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w

    objectives = 0.5 * ((X @ w1 + w0 - y) ** 2)

    if meantype == "ordinary":
        obj = objectives.mean()
    elif meantype == "catoni":
        obj = Holland_catoni_estimator(objectives)
    elif meantype == "mom":
        obj = median_of_means(objectives, int(MOMreg_block_size * len(objectives)))
    else:
        raise ValueError("unknown mean")

    if penalty:
        if fit_intercept:
            return obj + lamda * penalties[penalty](w[1:])
        else:
            return obj + lamda * penalties[penalty](w)
    else:
        return obj


def train_risk(w, algo_name=""):
    return risk(X_train, y_train, w, fit_intercept=fit_intercept)


def test_risk(w, algo_name=""):
    return risk(X_test, y_test, w, fit_intercept=fit_intercept, meantype="ordinary")


penalty = "l2"  # None  # "l1"#
lamda = 0.1  # 1/np.sqrt(X_train.shape[0])


def l1_penalty(x):
    return np.sum(np.abs(x))


def l2_penalty(x):
    return np.sum(x ** 2)


def l1_apply_single(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.0


def l1_apply(x, t):
    for j in range(len(x)):
        x[j] = l1_apply_single(x[j], t)


def l2_apply(x, t):
    x /= 1 + t


penalties = {"l1": l1_penalty, "l2": l2_penalty}
penalties_apply = {"l1": l1_apply, "l2": l2_apply}


logging.info(
    "Lauching experiment with parameters : \n dataset : %s, n_repeats = %d , n_samples = %d , n_features = %d, std scaling : %r , outliers = %r"
    % (dataset, n_repeats, n_samples, n_features, use_std_scaler, outliers)
)

logging.info("block sizes are : %r" % block_sizes)

logging.info("random_seed = %d " % random_seed)

logging.info("step_size = %f , max_iter = %d" % (step_size, max_iter))

# rng = np.random.RandomState(random_seed)  ## Global random generator

metrics = ["train_risk", "test_risk"]  # , "gradient_error"]


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator", "max_iter"])

algorithms = [
    Algorithm(name="holland_gd", solver="gd", estimator="ch", max_iter=max_iter),
    Algorithm(name="saga", solver="saga", estimator="erm", max_iter=3 * max_iter),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=2 * max_iter),
    Algorithm(name="erm_cgd", solver="cgd", estimator="erm", max_iter=3 * max_iter),
    Algorithm(name="catoni_cgd", solver="cgd", estimator="ch", max_iter=max_iter),
    # Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean", max_iter=max_iter),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=3 * max_iter),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm", max_iter=18 * max_iter),
    Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=5 * max_iter),
    Algorithm(name="svrg", solver="svrg", estimator="erm", max_iter=2 * max_iter),
    Algorithm(name="sgd", solver="sgd", estimator="erm", max_iter=4 * max_iter),
]


def run_repetition(rep):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d" % (rep + 1))
    logging.info(64 * "-")

    col_try, col_time, col_algo, col_metric, col_val = [], [], [], [], []

    outputs = {}

    def announce(x, status):
        logging.info(str(rep) + " : " + x + " " + status)

    def run_algorithm(algo, out):
        clf = Regressor(
            tol=0,
            max_iter=algo.max_iter,
            solver=algo.solver,
            estimator=algo.estimator,
            fit_intercept=fit_intercept,
            block_size=block_sizes[algo.name]
            if algo.name in block_sizes.keys()
            else 0.07,
            step_size=step_size,
            penalty=penalty or "none",
            C=1 / (n_samples * lamda) if penalty else 1.0,
        )
        clf.fit(X_train, y_train, dummy_first_step=True)
        announce(algo.name, "fitted")
        clf.compute_objective_history(X_train, y_train)
        clf.compute_objective_history(X_test, y_test)
        announce(algo.name, "computed history")
        out[algo.name] = clf.history_.records[1:]

    logging.info("Running algorithms ...")

    for algo in algorithms:
        run_algorithm(algo, outputs)

    for alg in outputs.keys():
        for ind_metric, metric in enumerate(metrics):
            for i in range(len(outputs[alg][0])):
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric)
                col_val.append(outputs[alg][ind_metric + 1].record[i])
                col_time.append(
                    outputs[alg][0].record[i] - outputs[alg][0].record[0]
                )  # i)#
    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time


results = [run_repetition(rep) for rep in range(n_repeats)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))

logging.info("Creating pandas DataFrame")
data = pd.DataFrame(
    {
        "time": col_time,
        "repeat": col_try,
        "algo": col_algo,
        "metric": col_metric,
        "value": col_val,
    }
)

indexNames = data[data["value"] > 1e8].index
# Delete these row indexes from dataFrame
logging.info("droping %d rows" % len(indexNames))
data.drop(indexNames, inplace=True)

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    import subprocess

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "linreg_realdata_results_%s" % dataset + now + ".pickle"
    ensure_directory("exp_archives/linreg_real_data/")
    with open("exp_archives/linreg_real_data/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "commit": commit, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.0

g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
g.map(sns.lineplot, "time", "value", "algo", lw=line_width).set(xlabel="", ylabel="")

g.set(
    ylim=(
        None,
        risk(
            X_test,
            y_test,
            np.zeros(n_features + int(fit_intercept)),
            fit_intercept=fit_intercept,
        ),
    )
)
# g.set_titles(col_template="{col_name}")

axes = g.axes.flatten()
axes[0].set_title("Train risk")
axes[1].set_title("Test risk")


# plt.legend(
axes[0].legend(
    list(data["algo"].unique()),
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    # loc="lower left",
    ncol=2,
    borderaxespad=0.2,
    columnspacing=1.0,
    fontsize=10,
)

plt.tight_layout()
plt.show()

if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    specs = "%s_block_size=%.2f" % (
        dataset,
        block_sizes["mom_cgd"],
    )
    fig_file_name = "exp_archives/linreg_realdata/" + specs + now + ".pdf"
    ensure_directory("exp_archives/linreg_realdata/")
    g.fig.savefig(fname=fig_file_name, bbox_inches="tight")
    logging.info("Saved figure into file : %s" % fig_file_name)
