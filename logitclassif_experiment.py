from linlearn import BinaryClassifier
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import itertools
from collections import namedtuple
import joblib
import time
from sklearn.metrics import accuracy_score


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory('exp_archives/')

file_handler = logging.FileHandler(filename='exp_archives/logitclassif_exp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
)

save_results = True
save_fig= True

logging.info(64*"=")
logging.info("Running new experiment session")
logging.info(64*"=")

step_size = 0.1

random_state = 43

max_iter = 2*50
fit_intercept = True

MOM_block_size = 0.05

test_loss_meantype = "ordinary"

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

dataset = "Adult"#"Stroke"#"Heart"#"weatherAUS"#"Bank"#

logging.info("loading dataset %s" % dataset)

def load_heart(test_size=0.3):
    csv_heart = pd.read_csv("heart/heart.csv")
    categoricals = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
    label = "output"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_heart[cat], prefix=cat)
        csv_heart = csv_heart.drop(cat, axis=1)
        csv_heart = csv_heart.join(one_hot)
    df_train, df_test = train_test_split(
        csv_heart,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_heart[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def load_stroke(test_size=0.3):
    csv_stroke = pd.read_csv("stroke/healthcare-dataset-stroke-data.csv").drop("id", axis=1).dropna()

    categoricals = ["gender", "hypertension", "heart_disease",
                    "ever_married", "work_type", "Residence_type", "smoking_status"]
    label = "stroke"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_stroke[cat], prefix=cat)
        csv_stroke = csv_stroke.drop(cat, axis=1)
        csv_stroke = csv_stroke.join(one_hot)
    df_train, df_test = train_test_split(
        csv_stroke,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_stroke[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def load_aus(test_size=0.3):
    #drop columns with too many NA values and Location because it has too many categories
    csv_aus = pd.read_csv("weatherAUS/weatherAUS.csv").drop(["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm", "Location"], axis=1)
    #convert date to yearday
    csv_aus = csv_aus.rename(columns={"Date": "yearday"})
    csv_aus["yearday"] = csv_aus.yearday.apply(lambda x: 30 * (int(x[5:7]) - 1) + int(x[8:])).astype(int)
    categoricals = ['WindGustDir','WindDir9am','WindDir3pm']
    label = "RainTomorrow"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_aus[cat], prefix=cat)
        csv_aus = csv_aus.drop(cat, axis=1)
        csv_aus = csv_aus.join(one_hot)

    csv_aus['RainToday'] = LabelBinarizer().fit_transform(csv_aus['RainToday'].astype('str'))
    csv_aus['RainTomorrow'] = LabelBinarizer().fit_transform(csv_aus['RainTomorrow'].astype('str'))
    csv_aus = csv_aus.dropna()
    df_train, df_test = train_test_split(
        csv_aus,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_aus[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

if dataset in ["Adult", "Bank"]:
    with open("pickled_%s.pickle"% dataset.lower(), "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    f.close()
elif dataset == "Heart":
    X_train, X_test, y_train, y_test = load_heart()
elif dataset == "Stroke":
    X_train, X_test, y_train, y_test = load_stroke()
elif dataset == "weatherAUS":
    X_train, X_test, y_train, y_test = load_aus()
else:
    ValueError("unknown dataset")

y_train = 2 * y_train - 1
y_test = 2 * y_test - 1
for dat in [X_train, X_test, y_train, y_test]:
    dat = np.ascontiguousarray(dat)

n_samples = X_train.shape[0]
n_repeats = 1

logging.info("Parameters are : n_repeats = %d , n_samples = %d , max_iter = %d , fit_intercept=%r, MOM_block_size = %.2f, test_loss_meantype = %s" % (n_repeats, n_samples or 0, max_iter, fit_intercept, MOM_block_size, test_loss_meantype))


#The below estimation is probably too sensitive to outliers
#Lip = np.linalg.eigh((X_train.T @ X_train)/X_train.shape[0])[0][-1] # highest eigenvalue
Lip = np.max([np.mean(X_train[:,j]**2) for j in range(X_train.shape[1])])

Lip_step = 1/(0.25*Lip) # 0.25 is Lipschitz smoothness of logistic loss

penalty = None#"l1"#
lamda = 1/np.sqrt(X_train.shape[0])

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
    x /= (1 + t)

penalties = {"l1" : l1_penalty, "l2" : l2_penalty}
penalties_apply = {"l1" : l1_apply, "l2" : l2_apply}


# record_type = [
#     ("record", float64[:]),
#     ("cursory_pred", int64),
# ]
#@jitclass(record_type)
# class Record(object):
#     def __init__(self, shape, capacity):
#         self.record = np.zeros(capacity) if shape == 1 else np.zeros(tuple([capacity] + list(shape)))
#         self.cursor = 0
#     def update(self, value):
#         self.record[self.cursor] = value
#         self.cursor += 1
#     def __len__(self):
#         return self.record.shape[0]

def gmom(xs, tol=1e-7, max_it=30):
    # from Vardi and Zhang 2000
    y = np.average(xs, axis=0)
    eps = 1e-10
    delta = 1
    niter = 0
    while delta > tol and niter < max_it:
        xsy = xs - y
        dists = np.linalg.norm(xsy, axis=1)
        inv_dists = 1 / dists
        mask = dists < eps
        inv_dists[mask] = 0
        nb_too_close = (mask).sum()
        ry = np.linalg.norm(np.dot(inv_dists, xsy))
        cst = nb_too_close / ry
        y_new = max(0, 1 - cst) * np.average(xs, axis=0, weights=inv_dists) + min(1, cst) * y
        delta = np.linalg.norm(y - y_new)
        y = y_new
        niter += 1
    # print(niter)
    return y

#@vectorize([float64(float64)])
def logit(x):
    if x > 0:
        return np.log(1 + np.exp(-x))
    else:
        return -x + np.log(1 + np.exp(x))
vec_logit = np.vectorize(logit)

#@vectorize([float64(float64)])
def sigmoid(z):
    if z > 0:
        return 1 / (1 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        return exp_z / (1 + exp_z)
vec_sigmoid = np.vectorize(sigmoid)


def sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0]
        w1 = w[1] if lnlearn else w[1:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    return vec_logit(y*scores)

def objective(X, y, w, fit_intercept=fit_intercept, lnlearn=False, meantype="ordinary"):
    objectives = sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=lnlearn)
    if meantype == "ordinary":
        obj = objectives.mean()
    elif meantype == "catoni":
        obj = Holland_catoni_estimator(objectives)
    elif meantype == "mom":
        obj = median_of_means(objectives, int(MOM_block_size*len(objectives)))
    else:
        raise ValueError("unknown mean")
    if penalty:
        if fit_intercept:
            if lnlearn:
                return obj + lamda * penalties[penalty](w[1])
            else:
                return obj + lamda * penalties[penalty](w[1:])
        else:
            return obj + lamda * penalties[penalty](w)
    else:
        return obj

def sample_gradients(X, y, w, fit_intercept=fit_intercept):
    scores = X @ w[1:] + w[0] if fit_intercept else X @ w

    derivatives = -y * vec_sigmoid(-y * scores)
    if fit_intercept:
        return np.hstack((derivatives[:,np.newaxis], np.einsum("i,ij->ij", derivatives, X)))
    else:
        return np.einsum("i,ij->ij", derivatives, X)

def gradient(X, y, w, fit_intercept=fit_intercept):
    return sample_gradients(X, y, w, fit_intercept=fit_intercept).mean(axis=0)

linlearn_algorithms = ["mom_cgd", "catoni_cgd", "tmean_cgd"]

def accuracy(X, y, w, fit_intercept=fit_intercept, lnlearn=False, meantype="ordinary"):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0]
        w1 = w[1] if lnlearn else w[1:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0

    decisions = ((y*scores) > 0).astype(int).astype(float)
    if meantype == "ordinary":
        acc = decisions.mean()
    elif meantype == "catoni":
        acc = Holland_catoni_estimator(decisions)
    elif meantype == "mom":
        acc = median_of_means(decisions, int(MOM_block_size*len(decisions)))
    else:
        raise ValueError("unknown mean")
    return acc


def train_loss(w, algo_name=""):
    return objective(X_train, y_train, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)
def test_loss(w, algo_name=""):
    return objective(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms, meantype=test_loss_meantype)

def test_accuracy(w, algo_name=""):
    return accuracy(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms, meantype=test_loss_meantype)



trackers = [(lambda w : w, (X_train.shape[1] + int(fit_intercept),)), (lambda _ : time.time(), 1)]

metrics = [train_loss, test_loss]#, test_accuracy]

Algorithm = StateHollandCatoni = namedtuple(
    "Algorithm", ["name", "solver", "estimator", "max_iter"]
)

algorithms = [Algorithm(name="saga", solver="saga", estimator="erm", max_iter=3*max_iter),
              Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=2*max_iter),
              Algorithm(name="erm_cgd", solver="cgd", estimator="erm", max_iter=3*max_iter),
              Algorithm(name="catoni_cgd", solver="cgd", estimator="holland_catoni", max_iter=max_iter),
              Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=3*max_iter),
              Algorithm(name="implicit_gd", solver="gd", estimator="implicit", max_iter=18*max_iter),
              Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=5*max_iter),
              Algorithm(name="holland_gd", solver="gd", estimator="holland_catoni", max_iter=max_iter),
              Algorithm(name="svrg", solver="svrg", estimator="erm", max_iter=2*max_iter),
              Algorithm(name="sgd", solver="sgd", estimator="erm", max_iter=4*max_iter),
              ]

def run_repetition(rep):
    col_try, col_algo, col_metric, col_val, col_time = [], [], [], [], []

    outputs = {}
    def announce(x):
        logging.info(str(rep)+" : "+x+" done")

    def run_algorithm(algo, out):
        clf = BinaryClassifier(tol=0, max_iter=algo.max_iter, solver=algo.solver, estimator=algo.estimator, fit_intercept=fit_intercept, lr_factor=step_size, penalty=penalty or "none")
        clf.fit(X_train, y_train, trackers=trackers, dummy_first_step=True)
        out[algo.name] = clf.history_.records
        announce(algo.name)

    for algo in algorithms:
        run_algorithm(algo, outputs)

    logging.info("computing objective history")
    for alg in outputs.keys():
        for ind_metric, metric in enumerate(metrics):
            for i in range(len(outputs[alg][0].record)):
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric.__name__)
                col_val.append(metric(outputs[alg][0].record[i]))
                col_time.append(outputs[alg][1].record[i] - outputs[alg][1].record[0])#i)#
    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time

if os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(run_repetition)(rep) for rep in range(n_repeats))
else:
    results = [run_repetition(rep) for rep in range(n_repeats)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))


data = pd.DataFrame({"repeat":col_try, "algorithm":col_algo, "metric":col_metric, "value":col_val, "time": col_time})

if save_results:
    logging.info("Saving results ...")
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "logitlasso_"+dataset+"_results_" + now + ".pickle"
    ensure_directory("exp_archives/logitlasso/")
    with open("exp_archives/logitlasso/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

g = sns.FacetGrid(
    data, col="metric", height=4, legend_out=True, sharey=False
)
g.map(
    sns.lineplot,
    "time",
    "value",
    "algorithm",
    #lw=4,
)#.set(yscale="log")#, xlabel="", ylabel="")

#g.set_titles(col_template="{col_name}")

#g.set(ylim=(0, 1))
axes = g.axes.flatten()

for ax in axes:
    ax.set_title("")

# _, y_high = axes[2].get_ylim()
# axes[2].set_ylim([0.75, y_high])

# for i, dataset in enumerate(df["dataset"].unique()):
#     axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
#     axes[i].set_title(dataset, fontsize=18)


plt.legend(
    list(data["algorithm"].unique()),
    #bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="upper right",
    #ncol=1,
    #borderaxespad=0.0,
    #fontsize=14,
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('data : %s , n=%d, block_size=%.2f' % (dataset, n_samples, MOM_block_size))

plt.show()


if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    ensure_directory("exp_archives/logitclassif/")
    specs = '%s_nrep=%d_meantype=%s_' % (dataset, n_repeats, test_loss_meantype)
    fig_file_name = "exp_archives/logitclassif/" + specs + now + ".pdf"
    g.fig.savefig(fname=fig_file_name)#, bbox_inches='tight')
    logging.info("Saved figure into file : %s" % fig_file_name)
