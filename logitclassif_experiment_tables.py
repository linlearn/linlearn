from linlearn import Classifier
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from collections import namedtuple
from scipy.special import logsumexp
from linlearn.estimator.ch import holland_catoni_estimator
from linlearn.estimator.tmean import fast_trimmed_mean
from linlearn._loss import median_of_means
import joblib
import itertools
import argparse
from data_loaders import load_aus, load_stroke, load_heart, load_adult, load_htru2, load_bank, load_mnist, load__iris, load_simulated

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

experiment_logfile = "exp_archives/logit_classif_tables_exp.log"
experiment_name = "logitclassif_tables"
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

parser.add_argument("--dataset", type=str, default="Stroke", choices=["Stroke", "Bank", "Heart", "Adult", "weatherAUS", "htru2", "MNIST", "iris", "simulated"])
parser.add_argument("--loss", type=str, default="logistic", choices=["logistic", "squaredhinge"])
parser.add_argument("--penalty", type=str, default="none", choices=["none", "l1", "l2", "elasticnet"])
parser.add_argument("--lamda", type=float, default=1.0)
parser.add_argument("--l1_ratio", type=float, default=0.5)
parser.add_argument("--block_size", type=float, default=0.07)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--percentage", type=float, default=0.01)
parser.add_argument("--meantype", type=str, default="mom", choices=["ordinary", "mom", "tmean", "ch"])
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--n_samples", type=int, default=10000)  # for simulated data
parser.add_argument("--n_features", type=int, default=20)  # for simulated data
parser.add_argument("--n_classes", type=int, default=5)  # for simulated data
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--n_repeats", type=int, default=10)
parser.add_argument("--max_iter", type=int, default=300)

args = parser.parse_args()

logging.info(48 * "=")
logging.info("Running new experiment session : %s"%experiment_name)
logging.info(48 * "=")


loss = args.loss

n_repeats = args.n_repeats
random_state = args.random_seed
max_iter = args.max_iter
fit_intercept = True

block_size = args.block_size
percentage = args.percentage

meantype = args.meantype

n_samples = args.n_samples
n_features = args.n_features
n_classes = args.n_classes

dataset = args.dataset
penalty = args.penalty
lamda = args.lamda  # /np.sqrt(X_train.shape[0])
l1_ratio = args.l1_ratio
tol = args.tol
step_size = args.step_size
test_size = args.test_size

logging.info("Received parameters : \n %r" % args)

logging.info("loading dataset %s" % dataset)


if dataset in ["MNIST", "iris"] or (dataset == "simulated" and n_classes > 2):
    binary = False
else:
    binary = True


def load_dataset(dataset):

    if dataset == "Bank":
        X_train, X_test, y_train, y_test = load_bank(test_size=test_size, random_state=random_state)
    elif dataset == "Adult":
        X_train, X_test, y_train, y_test = load_adult(test_size=test_size, random_state=random_state)
    elif dataset == "Heart":
        X_train, X_test, y_train, y_test = load_heart(test_size=test_size, random_state=random_state)
    elif dataset == "Stroke":
        X_train, X_test, y_train, y_test = load_stroke(test_size=test_size, random_state=random_state)
    elif dataset == "weatherAUS":
        X_train, X_test, y_train, y_test = load_aus(test_size=test_size, random_state=random_state)
    elif dataset == "htru2":
        X_train, X_test, y_train, y_test = load_htru2(test_size=test_size, random_state=random_state)
    elif dataset == "MNIST":
        X_train, X_test, y_train, y_test = load_mnist(test_size=test_size, random_state=random_state)
    elif dataset == "iris":
        X_train, X_test, y_train, y_test = load__iris(test_size=test_size, random_state=random_state)
    elif dataset == "simulated":
        X_train, X_test, y_train, y_test = load_simulated(n_samples, n_features, n_classes, test_size=test_size, random_state=random_state)
    else:
        ValueError("unknown dataset")

    std_scaler = StandardScaler()

    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    if binary:
        y_train = 2 * y_train - 1
        y_test = 2 * y_test - 1
    for dat in [X_train, X_test, y_train, y_test]:
        dat = np.ascontiguousarray(dat)
    print("n_features : %d"%X_train.shape[1])

    return X_train, X_test, y_train, y_test


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


def objective(X, y, clf, meantype=meantype, block_size=block_size, percentage=percentage):
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
        obj = median_of_means(sample_objectives, int(block_size * len(sample_objectives)))
    elif meantype == "tmean":
        obj = fast_trimmed_mean(sample_objectives, len(sample_objectives), percentage)
    else:
        raise ValueError("unknown mean")
    if penalty != "none":
        obj += lamda * penalties[penalty](clf.coef_)
    return obj


def accuracy(X, y, clf, meantype=meantype, block_size=block_size, percentage=percentage):
    if binary:
        scores = clf.decision_function(X)#clf.predict(X)
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


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator", "max_iter"])

algorithms = [
    Algorithm(
        name="batch_gd", solver="batch_gd", estimator="erm", max_iter=3 * max_iter
    ),
    # Algorithm(name="saga", solver="saga", estimator="erm", max_iter=3 * max_iter),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=2 * max_iter),
    Algorithm(name="mom_cgd_IS", solver="cgd", estimator="mom", max_iter=2 * max_iter),
    Algorithm(name="erm_cgd", solver="cgd", estimator="erm", max_iter=3 * max_iter),
    Algorithm(
       name="catoni_cgd", solver="cgd", estimator="ch", max_iter=max_iter
    ),
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean", max_iter=max_iter),
    Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=3 * max_iter),
    Algorithm(name="implicit_gd", solver="gd", estimator="llm", max_iter=9 * max_iter),
    Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=5 * max_iter),
    Algorithm(
       name="holland_gd", solver="gd", estimator="ch", max_iter=max_iter
    ),
    Algorithm(name="svrg", solver="svrg", estimator="erm", max_iter=2 * max_iter),
    # Algorithm(name="sgd", solver="sgd", estimator="erm", max_iter=4 * max_iter),
]


def announce(rep, x, status):
    logging.info(str(rep) + " : " + x + " " + status)

def run_algorithm(data, algo, rep, col_try, col_algo, col_train_loss, col_test_loss, col_train_acc, col_test_acc, col_fit_time):
    X_train, X_test, y_train, y_test = data
    n_samples = len(y_train)
    announce(rep, algo.name, "running")
    clf = Classifier(
        tol=tol,
        max_iter=max_iter,
        solver=algo.solver,
        loss=loss,
        estimator=algo.estimator,
        fit_intercept=fit_intercept,
        step_size=step_size,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=1/(n_samples * lamda),
    )
    clf.fit(X_train, y_train, dummy_first_step=True)
    announce(rep, algo.name, "fitted")
    col_try.append(rep)
    col_algo.append(algo.name)
    col_train_loss.append(objective(X_train, y_train, clf))
    col_test_loss.append(objective(X_test, y_test, clf))
    col_train_acc.append(accuracy(X_train, y_train, clf))
    col_test_acc.append(accuracy(X_test, y_test, clf))
    col_fit_time.append(clf.fit_time())


def run_repetition(rep):
    col_try, col_algo, col_train_loss, col_test_loss, col_train_acc, col_test_acc, col_fit_time = [], [], [], [], [], [], []
    data = load_dataset(dataset)
    for algo in algorithms:
        run_algorithm(data, algo, rep, col_try, col_algo, col_train_loss, col_test_loss, col_train_acc, col_test_acc, col_fit_time)

    logging.info("repetition done")
    return col_try, col_algo, col_train_loss, col_test_loss, col_train_acc, col_test_acc, col_fit_time

if os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_repetition)(rep) for rep in range(1, n_repeats+1)
    )
else:
    results = [run_repetition(rep) for rep in range(1, n_repeats+1)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_train_loss = list(itertools.chain.from_iterable([x[2] for x in results]))
col_test_loss = list(itertools.chain.from_iterable([x[3] for x in results]))
col_train_acc = list(itertools.chain.from_iterable([x[4] for x in results]))
col_test_acc = list(itertools.chain.from_iterable([x[5] for x in results]))
col_fit_time = list(itertools.chain.from_iterable([x[6] for x in results]))

data = pd.DataFrame(
    {
        "repeat": col_try,
        "algorithm": col_algo,
        "train_loss": col_train_loss,
        "test_loss": col_test_loss,
        "train_acc": col_train_acc,
        "test_acc": col_test_acc,
        "fit_time": col_fit_time,
    }
)

data_no_repeats = data.drop("repeat", axis=1)
means = (data_no_repeats.groupby(["algorithm"]).mean()).add_suffix("_mean")
means_stds = means.join((data_no_repeats.groupby(["algorithm"]).std()).add_suffix("_std")).sort_index(axis=1)


logging.info("Saving results ...")
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

filename = experiment_name + "_" + dataset + "_results_" + now + ".pickle"
ensure_directory("exp_archives/"+experiment_name+"/")
with open("exp_archives/"+experiment_name+"/" + filename, "wb") as f:
    pickle.dump({"datetime": now, "args" :args, "results": data, "means_stds":means_stds}, f)

logging.info("Saved results in file %s" % filename)

print(data)

print(means_stds)