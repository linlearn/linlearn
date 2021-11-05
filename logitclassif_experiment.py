from linlearn import Classifier
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from scipy.special import logsumexp
import itertools
from collections import namedtuple
from linlearn.estimator.ch import holland_catoni_estimator
from linlearn.estimator.tmean import fast_trimmed_mean
from linlearn._loss import median_of_means
import joblib
import argparse
from pathlib import Path
# from data_loaders import (
#     load_aus,
#     load_stroke,
#     load_heart,
#     load_adult,
#     load_htru2,
#     load_bank,
#     load_mnist,
#     load__iris,
#     load_simulated,
# )
from linlearn.datasets import (  # noqa: E402
    load_adult,
    load_bank,
    load_boston,
    load_breastcancer,
    load_californiahousing,
    load_car,
    load_cardio,
    load_churn,
    load_default_cb,
    load_diabetes,
    load_iris,
    load_letter,
    load_mnist,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_amazon,
    load_covtype,
    load_kick,
    load_internet,
    load_higgs,
    load_kddcup99,
)

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


experiment_logfile = "exp_archives/logitclassif_exp.log"
experiment_name = "logitclassif"

ensure_directory("exp_archives/")

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

parser.add_argument(
    "--dataset",
    type=str.lower,
    default="car",
    choices=[
        "stroke",
        "bank",
        "car",
        "heart",
        "adult",
        "weatheraus",
        "htru2",
        "mnist",
        "iris",
        "simulated",
    ],
)
parser.add_argument(
    "--loss", type=str, default="logistic", choices=["logistic", "squaredhinge"]
)
parser.add_argument(
    "--penalty", type=str, default="none", choices=["none", "l1", "l2", "elasticnet"]
)
parser.add_argument("--lamda", type=float, default=1.0)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--block_size", type=float, default=0.07)
parser.add_argument("--percentage", type=float, default=0.01)
parser.add_argument("--l1_ratio", type=float, default=0.5)
parser.add_argument(
    "--meantype", type=str, default="mom", choices=["ordinary", "mom", "tmean", "ch"]
)
parser.add_argument("--random_seed", type=int, default=43)
parser.add_argument("--n_samples", type=int, default=10000)  # for simulated data
parser.add_argument("--n_features", type=int, default=20)  # for simulated data
parser.add_argument("--n_classes", type=int, default=5)  # for simulated data
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--max_iter", type=int, default=300)
parser.add_argument('--no_cgd_IS', dest='cgd_IS', action='store_false')


args = parser.parse_args()

logging.info(64 * "=")
logging.info("Running new experiment session")
logging.info(64 * "=")

loss = args.loss

n_repeats = args.n_repeats
random_state = args.random_seed
max_iter = args.max_iter
fit_intercept = True
cgd_IS = args.cgd_IS

block_size = args.block_size
percentage = args.percentage
test_size = args.test_size

n_samples = args.n_samples
n_features = args.n_features
n_classes = args.n_classes

meantype = args.meantype

dataset_name = args.dataset.lower()
penalty = args.penalty
lamda = args.lamda  # /np.sqrt(X_train.shape[0])
tol = args.tol
step_size = args.step_size
l1_ratio = args.l1_ratio


logging.info("Received parameters : \n %r" % args)

def set_dataloader(dataset_name):
    loaders_mapping = {
        "adult": load_adult,
        "bank": load_bank,
        "boston": load_boston,
        "breastcancer": load_breastcancer,
        "californiahousing": load_californiahousing,
        "car": load_car,
        "cardio": load_cardio,
        "churn": load_churn,
        "default-cb": load_default_cb,
        "diabetes": load_diabetes,
        "iris": load_iris,
        "letter": load_letter,
        "mnist": load_mnist,
        "satimage": load_satimage,
        "sensorless": load_sensorless,
        "spambase": load_spambase,
        # "amazon": load_amazon,
        # "covtype": load_covtype,
        # "internet": load_internet,
        # "kick": load_kick,
        # "kddcup": load_kddcup99,
        # "higgs": load_higgs,
    }
    return loaders_mapping[dataset_name]

# def load_dataset(dataset):
#
#     elif dataset == "heart":
#         X_train, X_test, y_train, y_test = load_heart(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "stroke":
#         X_train, X_test, y_train, y_test = load_stroke(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "weatheraus":
#         X_train, X_test, y_train, y_test = load_aus(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "htru2":
#         X_train, X_test, y_train, y_test = load_htru2(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "mnist":
#         X_train, X_test, y_train, y_test = load_mnist(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "iris":
#         X_train, X_test, y_train, y_test = load__iris(
#             test_size=test_size, random_state=random_state
#         )
#     elif dataset == "simulated":
#         X_train, X_test, y_train, y_test = load_simulated(
#             n_samples,
#             n_features,
#             n_classes,
#             test_size=test_size,
#             random_state=random_state,
#         )
#     else:
#         ValueError("unknown dataset")
#
#     std_scaler = StandardScaler()
#
#     X_train = std_scaler.fit_transform(X_train)
#     X_test = std_scaler.transform(X_test)
#
#     if binary:
#         y_train = 2 * y_train - 1
#         y_test = 2 * y_test - 1
#     for dat in [X_train, X_test, y_train, y_test]:
#         dat = np.ascontiguousarray(dat)
#     # print("n_features : %d" % X_train.shape[1])
#
#     return X_train, X_test, y_train, y_test


logging.info(
    "Parameters are : loss = %s , n_repeats = %d , max_iter = %d , fit_intercept=%r, meantype = %s"
    % (loss, n_repeats, max_iter, fit_intercept, meantype)
)


def l1_penalty(x):
    return np.sum(np.abs(x))


def l2_penalty(x):
    return 0.5 * np.sum(x ** 2)


def elasticnet_penalty(x):
    return l1_ratio * l1_penalty(x) + (1.0 - l1_ratio) * l2_penalty(x)


penalties = {"l1": l1_penalty, "l2": l2_penalty, "elasticnet": elasticnet_penalty}


def logit(x):
    if x > 0:
        return np.log(1 + np.exp(-x))
    else:
        return -x + np.log(1 + np.exp(x))


vec_logit = np.vectorize(logit)


def objective(X, y, clf, meantype=meantype, block_size=block_size, percentage=0.01):
    if binary:
        sample_objectives = vec_logit(clf.decision_function(X) * y)
    else:
        scores = clf.decision_function(X)
        sample_objectives = -scores[np.arange(X.shape[0]), y] + logsumexp(
            scores, axis=1
        )

    if meantype == "ordinary":
        obj = sample_objectives.mean()
    elif meantype == "ch":
        obj = holland_catoni_estimator(sample_objectives)
    elif meantype == "mom":
        obj = median_of_means(
            sample_objectives, int(block_size * len(sample_objectives))
        )
    elif meantype == "tmean":
        obj = fast_trimmed_mean(sample_objectives, len(sample_objectives), percentage)
    else:
        raise ValueError("unknown mean")
    if penalty != "none":
        obj += lamda * penalties[penalty](clf.coef_)
    return obj


def accuracy(X, y, clf, meantype=meantype, block_size=block_size, percentage=0.01):
    if binary:
        scores = clf.decision_function(X)  # clf.predict(X)
        decisions = ((y * scores) > 0).astype(int).astype(float)
    else:
        predictions = clf.predict(X)
        decisions = (y == predictions).astype(int).astype(float)

    if meantype == "ordinary":
        acc = decisions.mean()
    elif meantype == "ch":
        acc = holland_catoni_estimator(decisions)
    elif meantype == "mom":
        acc = median_of_means(decisions, int(block_size * len(decisions)))
    elif meantype == "tmean":
        acc = fast_trimmed_mean(decisions, len(decisions), percentage=percentage)
    else:
        raise ValueError("unknown mean")
    return acc


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator"])

algorithms = [
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean"),
    # Algorithm(
    #     name="batch_gd", solver="batch_gd", estimator="erm"
    # ),
    # Algorithm(name="saga", solver="saga", estimator="erm"),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom"),
    # Algorithm(name="mom_cgd_IS", solver="cgd", estimator="mom"),
    Algorithm(name="erm_cgd", solver="cgd", estimator="erm"),
    Algorithm(
       name="catoni_cgd", solver="cgd", estimator="ch"
    ),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom"),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm"),
    Algorithm(name="erm_gd", solver="gd", estimator="erm"),
    Algorithm(
       name="holland_gd", solver="gd", estimator="ch"
    ),
    # Algorithm(name="svrg", solver="svrg", estimator="erm"),
    # Algorithm(name="sgd", solver="sgd", estimator="erm"),
]

def get_finetuned_params(algo):
    if os.path.exists("results/" + dataset_name):

        paths = sorted(Path("results/" + dataset_name).iterdir(), key=os.path.getmtime, reverse=True)
        for path in paths:
            pth = str(path).lower()
            if algo.estimator + "_" + algo.solver in pth:
                with open(path, "rb") as f:
                    content = pickle.load(f)
                    params = content["best_param"]
                    params.pop("random_state")
                logging.info("Found finetuned parameters for algorithm %s in file : %s"%(algo.name, path))
                logging.info("parameters : %r" % params)
                return params
    logging.info("Could not find finetuned params for algorithm %s, will use defaults"%(algo.name))
    return None

logging.info("Collecting finetuned parameters for algorithms ... ")
finetuned_params = {algo.name: get_finetuned_params(algo) for algo in algorithms}


def announce(rep, x, status):
    logging.info(str(rep) + " : " + x + " " + status)

def run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time, col_sc_prods):

    X_train, X_test, y_train, y_test = data
    n_samples = len(y_train)
    announce(rep, algo.name, "running")
    params = finetuned_params[algo.name]
    if params is None:
        print("cgd_IS is %r"%cgd_IS)
        clf = Classifier(
            tol=tol,
            max_iter=max_iter,
            solver=algo.solver,
            loss=loss,
            estimator=algo.estimator,
            fit_intercept=fit_intercept,
            step_size=step_size,
            penalty=penalty,
            cgd_IS=cgd_IS,
            l1_ratio=l1_ratio,
            C=1 / (n_samples * lamda),
        )
    else:
        clf = Classifier(
            tol=tol,
            max_iter=max_iter,
            loss=loss,
            fit_intercept=fit_intercept,
            step_size=step_size,
            penalty=penalty,
            l1_ratio=l1_ratio,
            C=1 / (n_samples * lamda),
            **params
        )


    clf.fit(X_train, y_train, dummy_first_step=True)
    announce(rep, algo.name, "fitted")
    clf.compute_objective_history(X_train, y_train, metric="objective")
    clf.compute_objective_history(X_test, y_test, metric="objective")
    clf.compute_objective_history(X_train, y_train, metric="misclassif_rate")
    clf.compute_objective_history(X_test, y_test, metric="misclassif_rate")
    announce(rep, algo.name, "computed history")

    records = clf.history_.records[1:]

    for j, metric in enumerate(["train_loss", "test_loss", "misclassif_train", "misclassif_test"]):
        # for i in range(len(records[0])):
        for i in range(records[0].cursor):
            col_try.append(rep)
            col_algo.append(algo.name)
            col_metric.append(metric)
            col_val.append(records[1 + j].record[i])
            col_time.append(i)  # records[0].record[i] - records[0].record[0])#
            col_sc_prods.append(clf.history_.record_nm("sc_prods").record[i])

def run_repetition(rep):
    col_try, col_algo, col_metric, col_val, col_time, col_sc_prods = [], [], [], [], [], []
    loader = set_dataloader(dataset_name)
    logging.info("loading dataset %s"%dataset_name)
    dataset = loader()
    data = dataset.extract(random_state=random_state)
    for algo in algorithms:
        run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time, col_sc_prods)

    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time, col_sc_prods


if False:#os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_repetition)(rep) for rep in range(1, n_repeats + 1)
    )
else:
    results = [run_repetition(rep) for rep in range(1, n_repeats + 1)]


col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))
col_sc_prods = list(itertools.chain.from_iterable([x[5] for x in results]))


data = pd.DataFrame(
    {
        "repeat": col_try,
        "algorithm": col_algo,
        "metric": col_metric,
        "value": col_val,
        "time": col_time,
        "sc_prods": col_sc_prods
    }
)

# save_results:
logging.info("Saving results ...")
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

filename = experiment_name + "_" + dataset_name + "_results_" + now + ".pickle"
ensure_directory("exp_archives/" + experiment_name + "/")
with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
    pickle.dump({"datetime": now, "results": data}, f)

logging.info("Saved results in file %s" % filename)
logging.info("Plotting ...")

g = sns.FacetGrid(data, col="metric", height=4, col_wrap=2, legend_out=True, sharey=False)
g.map(
    sns.lineplot,
    "sc_prods",#"time",
    "value",
    "algorithm",
    # lw=4,
)  # .set(yscale="log")#, xlabel="", ylabel="")

# g.set_titles(col_template="{col_name}")

# g.set(ylim=(0, 1))
axes = g.axes.flatten()

# for ax in axes:
#     ax.set_title("")

# _, y_high = axes[2].get_ylim()
# axes[2].set_ylim([0.75, y_high])

# for i, dataset in enumerate(df["dataset"].unique()):
#     axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
#     axes[i].set_title(dataset, fontsize=18)


plt.legend(
    list(data["algorithm"].unique()),
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="upper right",
    # ncol=1,
    # borderaxespad=0.0,
    # fontsize=14,
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("data : %s , loss=%s, meantype=%s" % (dataset_name, loss.upper(), meantype))

plt.show()


# save figure :
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
ensure_directory("exp_archives/" + experiment_name + "/")
specs = "%s_nrep=%d_meantype=%s_" % (dataset_name, n_repeats, meantype)
fig_file_name = "exp_archives/" + experiment_name + "/" + specs + now + ".pdf"
g.fig.savefig(fname=fig_file_name)  # , bbox_inches='tight')
logging.info("Saved figure into file : %s" % fig_file_name)
