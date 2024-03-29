from linlearn import Regressor
import numpy as np
import logging
import pickle
from datetime import datetime
import sys, os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import argparse
from linlearn.estimator.ch import holland_catoni_estimator
from linlearn.estimator.tmean import fast_trimmed_mean
from linlearn._loss import median_of_means
from collections import namedtuple
import joblib
from data_loaders import load_california_housing, load_used_car, load_data_boston, fetch_diabetes, load_simulated_regression

experiment_logfile = "exp_archives/linreg_realdata.log"
experiment_name = "linreg_realdata"


file_handler = logging.FileHandler(filename=experiment_logfile)
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

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Diabetes", choices=["used_car", "CaliforniaHousing", "Boston", "Diabetes", "simulated"])
parser.add_argument("--loss", type=str, default="logistic", choices=["leastsquares", "huber"])
parser.add_argument("--huber_delta", type=float, default=1.35)
parser.add_argument("--penalty", type=str, default="none", choices=["none", "l1", "l2", "elasticnet"])
parser.add_argument("--lamda", type=float, default=1.0)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--block_size", type=float, default=0.07)
parser.add_argument("--percentage", type=float, default=0.01)
parser.add_argument("--l1_ratio", type=float, default=0.5)
parser.add_argument("--meantype", type=str, default="mom", choices=["ordinary", "mom", "tmean", "ch"])
parser.add_argument("--random_seed", type=int, default=43)
parser.add_argument("--n_samples", type=int, default=10000)  # for simulated data
parser.add_argument("--n_features", type=int, default=20)  # for simulated data
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--max_iter", type=int, default=300)

args = parser.parse_args()


logging.info(128 * "=")
logging.info("Running new experiment session")
logging.info(128 * "=")

fit_intercept = True

percentage = args.percentage
block_size = args.block_size
n_repeats = args.n_repeats

loss = args.loss
huber_delta = args.huber_delta
random_state = args.random_seed
test_size = args.test_size
n_samples = args.n_samples
n_features = args.n_features

dataset = args.dataset
penalty = args.penalty
lamda = args.lamda  # 1/np.sqrt(X_train.shape[0])
l1_ratio = args.l1_ratio
tol = args.tol

logging.info("Received parameters : \n %r" % args)

def load_dataset(dataset):
    if dataset == "Boston":
        X_train, X_test, y_train, y_test = load_data_boston(test_size=test_size, random_state=random_state)
    elif dataset == "CaliforniaHousing":
        X_train, X_test, y_train, y_test = load_california_housing(test_size=test_size, random_state=random_state)
    elif dataset == "Diabetes":
        X_train, X_test, y_train, y_test = fetch_diabetes(test_size=test_size, random_state=random_state)
    elif dataset == "used_car":
        X_train, X_test, y_train, y_test = load_used_car(test_size=test_size, random_state=random_state)
    elif dataset == "simulated":
        X_train, X_test, y_train, y_test = load_simulated_regression(n_samples, n_features, test_size=test_size, random_state=random_state)

    else:
        raise ValueError("Unknown dataset")

    std_scaler = StandardScaler()

    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

meantype = args.meantype
step_size = args.step_size
max_iter = args.max_iter


def risk(X, y, clf, meantype=meantype, block_size=block_size, percentage=percentage):
    if loss == "leastsquares":
        objectives = 0.5 * ((clf.decision_function(X) - y) ** 2)
    else:
        ValueError("unimplemented loss %s"%loss)

    if meantype == "ordinary":
        obj = objectives.mean()
    elif meantype == "catoni":
        obj = holland_catoni_estimator(objectives)
    elif meantype == "mom":
        obj = median_of_means(objectives, int(block_size * len(objectives)))
    elif meantype == "tmean":
        obj = fast_trimmed_mean(objectives, len(objectives), percentage)
    else:
        raise ValueError("unknown mean")

    if penalty:
        obj += lamda * penalties[penalty](clf.coef_)

    return obj


def l1_penalty(x):
    return np.sum(np.abs(x))

def l2_penalty(x):
    return 0.5 * np.sum(x ** 2)

def elasticnet_penalty(x):
    return l1_ratio * l1_penalty(x) + (1.0 - l1_ratio) * l2_penalty(x)


penalties = {"l1": l1_penalty, "l2": l2_penalty, "elasticnet": elasticnet_penalty}



metrics = ["train_risk", "test_risk"]  # , "gradient_error"]


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator"])

algorithms = [
    Algorithm(name="holland_gd", solver="gd", estimator="ch"),
    # Algorithm(name="saga", solver="saga", estimator="erm"),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom"),
    Algorithm(name="erm_cgd", solver="cgd", estimator="erm"),
    Algorithm(name="catoni_cgd", solver="cgd", estimator="ch"),
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean"),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom"),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm"),
    Algorithm(name="erm_gd", solver="gd", estimator="erm"),
    # Algorithm(name="svrg", solver="svrg", estimator="erm"),
    Algorithm(name="sgd", solver="sgd", estimator="erm"),
]

def announce(rep, x, status):
    logging.info(str(rep) + " : " + x + " " + status)

def run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time):
    X_train, X_test, y_train, y_test = data
    n_samples = len(y_train)
    announce(rep, algo.name, "running")

    clf = Regressor(
        tol=tol,
        loss=loss,
        max_iter=max_iter,
        solver=algo.solver,
        estimator=algo.estimator,
        fit_intercept=fit_intercept,
        block_size=block_size,
        step_size=step_size,
        penalty=penalty,
        cgd_IS=algo.name[-2:] == "IS",
        l1_ratio=l1_ratio,
        C=1 / (n_samples * lamda),
    )
    clf.fit(X_train, y_train, dummy_first_step=True)
    announce(rep, algo.name, "fitted")
    clf.compute_objective_history(X_train, y_train)
    clf.compute_objective_history(X_test, y_test)
    announce(rep, algo.name, "computed history")

    records = clf.history_.records[1:]

    for j, metric in enumerate(["train_loss", "test_loss"]):
        # for i in range(len(records[0])):
        for i in range(records[0].cursor):
            col_try.append(rep)
            col_algo.append(algo.name)
            col_metric.append(metric)
            col_val.append(records[1+j].record[i])
            col_time.append(records[0].record[i] - records[0].record[0])#i)#


def run_repetition(rep):
    logging.info(64 * "-")
    logging.info("repeat : %d" % rep)
    logging.info(64 * "-")

    col_try, col_algo, col_metric, col_val, col_time = [], [], [], [], []
    data = load_dataset(dataset)
    for algo in algorithms:
        run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time)

    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time

if os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_repetition)(rep) for rep in range(1, n_repeats+1)
    )
else:
    results = [run_repetition(rep) for rep in range(1, n_repeats+1)]


col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))


data = pd.DataFrame(
    {
        "repeat": col_try,
        "algorithm": col_algo,
        "metric": col_metric,
        "value": col_val,
        "time": col_time,
    }
)


indexNames = data[data["value"] > 1e8].index
# Delete these row indexes from dataFrame
logging.info("dropping %d rows" % len(indexNames))
data.drop(indexNames, inplace=True)

# save results:
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

import subprocess

# Get the commit number as a string
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
commit = commit.decode("utf-8").strip()

filename = experiment_name + "_results_%s" % dataset + now + ".pickle"
ensure_directory("exp_archives/" + experiment_name + "/")
with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
    pickle.dump({"datetime": now, "commit": commit, "results": data}, f)

logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.5

g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
g.map(sns.lineplot, "time", "value", "algorithm", lw=line_width).set(xlabel="", ylabel="")

# g.set(
#     ylim=(
#         None,
#         risk(
#             X_test,
#             y_test,
#             np.zeros(n_features + int(fit_intercept)),
#             fit_intercept=fit_intercept,
#         ),
#     )
# )

# g.set_titles(col_template="{col_name}")

axes = g.axes.flatten()
axes[0].set_title("Train risk")
axes[1].set_title("Test risk")


# plt.legend(
axes[0].legend(
    list(data["algorithm"].unique()),
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    # loc="lower left",
    ncol=2,
    borderaxespad=0.2,
    columnspacing=1.0,
    fontsize=10,
)

g.fig.suptitle(
    "data : %s , loss=%s, meantype=%s"
    % (dataset, loss.upper(), meantype)
)

plt.tight_layout()
plt.show()

# save fig:
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
specs = "%s_block_size=%.2f" % (
    dataset,
    block_size,
)
fig_file_name = "exp_archives/" + experiment_name + "/" + specs + now + ".pdf"
ensure_directory("exp_archives/" + experiment_name + "/")
g.fig.savefig(fname=fig_file_name, bbox_inches="tight")
logging.info("Saved figure into file : %s" % fig_file_name)
