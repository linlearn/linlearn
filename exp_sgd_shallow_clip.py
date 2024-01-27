import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
import itertools
from math import floor, exp, log, fabs
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import joblib

import argparse
from numba import jit, objmode

from linlearn.datasets import (  # noqa: E402
    load_adult, # doesn't work
    load_bank, # doesn't work
    load_californiahousing,
    # load_car,
    # load_cardio,
    # load_churn,
    load_default_cb, # 30000, 24 doesn't work
    # load_diabetes,
    load_letter, # 20000, 17 # seems to work
    # load_satimage,
    load_sensorless, # 58509, 49 # seems to work
    # load_spambase,
    load_amazon, # 32769, 10
    load_covtype, # seems to work
    load_kick, # 72983, 33
    # load_internet,
    # load_higgs, ## too big
    load_codrna, # seems to work
    load_elec2, # doesn't work
    load_ijcnn1, # doesn't work
    load_phishing, # seems to work
    load_phishing_st, # seems to work
    load_kddcup99,
    # load_electrical,
    load_occupancy, # doesn't work
    load_avila, # 20867, 11 # doesn't work
    load_miniboone, # 130064, 51 # seems to work
    # load_gas,
    # load_eeg,
    # load_drybean,
    # load_cbm,
    load_metro, # doesn't work
    # load_ccpp,
    load_energy, # 19735, 27 reg # doesn't work
    load_gasturbine, # 36733,11 reg # doesn't work
    load_casp, # 45730, 10 reg # doesn't work
    load_superconduct, # 21263, 82 reg # doesn't work
    # load_bike,
    load_ovctt, # doesn't work
    load_sgemm, # doesn't work
    load_ypmsd,
    load_nupvotes, # 330045, 5 reg
    load_houseprices,
    load_fifa19,
    load_nyctaxi,
    # load_wine,
    # load_statlog,
)

MAX_PARAM_VALUE = 1e10

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


@jit(**jit_kwargs)
def update_buffer(x, buffer, ages, buff_size):

    ind = 0
    for i in range(buff_size):
        if ages[i] == buff_size-1:
            ind = i
            ages[i] = 0
            buffer[i] = x
        else:
            ages[i] += 1
    while ind > 0 and buffer[ind] < buffer[ind-1]:
        buffer[ind], buffer[ind-1] = buffer[ind-1], buffer[ind]
        ages[ind], ages[ind-1] = ages[ind-1], ages[ind]
        ind -= 1
    while ind < buff_size-1 and buffer[ind] > buffer[ind+1]:
        buffer[ind], buffer[ind+1] = buffer[ind+1], buffer[ind]
        ages[ind], ages[ind+1] = ages[ind+1], ages[ind]
        ind += 1


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def set_dataloader(dataset_name):
    loaders_mapping = {
        "adult": load_adult,
        "bank": load_bank,
        "californiahousing": load_californiahousing,
        "default-cb": load_default_cb,
        "elec2": load_elec2,
        "letter": load_letter,
        "sensorless": load_sensorless,
        "amazon": load_amazon,
        "covtype": load_covtype,
        "kick": load_kick,
        "kddcup": load_kddcup99,
        "occupancy": load_occupancy,
        "avila": load_avila,
        "miniboone": load_miniboone,
        "metro": load_metro,
        "energy": load_energy,
        "gasturbine": load_gasturbine,
        "casp": load_casp,
        "superconduct": load_superconduct,
        "sgemm": load_sgemm,
        "ovctt": load_ovctt,
        "ypmsd": load_ypmsd,
        "nupvotes": load_nupvotes,
        "houseprices": load_houseprices,
        "fifa19": load_fifa19,
        "nyctaxi": load_nyctaxi,
        "codrna": load_codrna,
        "ijcnn1": load_ijcnn1,
        "phishing": load_phishing,
        "phishing_st": load_phishing_st,
    }
    return loaders_mapping[dataset_name]

def binary_classif_metrics(model, Xtest, ytest):#, batch_size=100):

    output = model(Xtest)
    if learner == "shallow":
        yscores = F.softmax(output, dim=1).detach().numpy()[:, 1]
    else:
        yscores = torch.sigmoid(output).detach().numpy()  # [:, 1]
    y_pred = (yscores >= 0.5).astype(int)
    roc_auc = roc_auc_score(ytest, yscores)
    acc = accuracy_score(ytest, y_pred)
    avg_prec = average_precision_score(ytest, yscores)
    lss = loss(output, ytest if learner == "shallow" else ytest.reshape(output.shape)).detach().numpy()/test_data_size
    return acc, roc_auc, avg_prec, lss

def multiclassif_metrics(model, Xtest, ytest, ytestbinary):#, batch_size=100):

    output = model(Xtest)
    yscores = F.softmax(output, dim=1).detach().numpy()#[:, 1]
    y_pred = yscores.argmax(axis=1)

    acc = accuracy_score(ytest, y_pred)

    lss = loss(output, ytest).detach().numpy()/test_data_size

    roc_auc = roc_auc_score(ytest, yscores, multi_class="ovr", average="macro")
    # roc_auc_weighted = roc_auc_score(ytest, yscores, multi_class="ovr", average="weighted")

    avg_precision = average_precision_score(ytestbinary, yscores)
    # avg_precision_weighted = average_precision_score(ytestbinary, yscores, average="weighted")

    return acc, roc_auc, avg_precision, lss

def regression_metrics(model, Xtest, ytest):#, batch_size=100):

    output = model(Xtest)
    y_scores = output.detach().numpy()
    mse = mean_squared_error(ytest, y_scores)

    mae = mean_absolute_error(ytest, y_scores)

    return mse, mae

# def update_qc(model, optim, features_batch, targets_batch, buffer, ages):
#     optim.zero_grad()
#     pred = model(features_batch)
#     loss_val = loss(pred, targets_batch)
#     #train_loss.append(loss_val.item())
#     #loss_val = loss(pred, torch.unsqueeze(targets, 1))
#     loss_val.backward()
#     current_norm = nn.utils.clip_grad_norm_(model.parameters(), buffer[floor(len(buffer) * quant)])
#     update_buffer(current_norm.item(), buffer, ages, buffer_size)
#     optim.step()
#
# def update_noclip(model, optim, features_batch, targets_batch):
#     optim.zero_grad()
#     pred = model(features_batch)
#     loss_val = loss(pred, targets_batch)
#     #train_loss.append(loss_val.item())
#     #loss_val = loss(pred, torch.unsqueeze(targets, 1))
#     loss_val.backward()
#     optim.step()
#
# def update_cstclip(model, optim, features_batch, targets_batch, clip_level):
#     optim.zero_grad()
#     pred = model(features_batch)
#     loss_val = loss(pred, targets_batch)
#     #train_loss.append(loss_val.item())
#     #loss_val = loss(pred, torch.unsqueeze(targets, 1))
#     loss_val.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), clip_level)
#     optim.step()
#
# update_functions = [update_qc, update_noclip, update_cstclip]
#
# def get_update_params(i, buffer, ages, clip_level):
#     if i == 0:
#         return (buffer, ages)
#     elif i == 1:
#         return ()
#     else:
#         return (clip_level,)

ensure_directory("exp_archives/")
experiment_logfile = "exp_archives/exp_shallow_classif.log"
experiment_name = "shallow_classif"

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
    "--dataset_name",
    choices=[
        "adult",
        "bank",
        "boston",
        "breastcancer",
        "californiahousing",
        "car",
        "cardio",
        "churn",
        "default-cb",
        "elec2",
        "diabetes",
        "letter",
        "satimage",
        "sensorless",
        "spambase",
        "amazon",
        "covtype",
        "internet",
        "kick",
        "kddcup",
        "higgs",
        "electrical",
        "occupancy",
        "avila",
        "miniboone",
        "codrna",
        "ijcnn1",
        "phishing",
        "phishing_st",
        "gas",
        "eeg",
        "drybean",
        "cbm",
        "ccpp",
        "metro",
        "energy",
        "gasturbine",
        "bike",
        "casp",
        "superconduct",
        "sgemm",
        "ovctt",
        "ypmsd",
        "nupvotes",
        "houseprices",
        "fifa19",
        "nyctaxi",
    ],
)
parser.add_argument("--learner", choices=["shallow", "logistic"], default="shallow")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--n_repeats", type=int, default=10)
parser.add_argument("--buffer_size", type=int, default=100)
parser.add_argument("--evaluation_period", type=int, default=100)
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--test_data_size", type=int, default=5000)
parser.add_argument("--n_epochs", type=int, default=1)
#parser.add_argument("--outlier_types", nargs="+", type=int, default=[])
parser.add_argument("--step_size", type=float, default=0.01)
parser.add_argument("--cst_clip_quants", nargs="+", type=float, default=[0.25, 0.5, 0.75])
parser.add_argument("--tau_unif", type=float, default=10)
parser.add_argument("--quantile", type=float, default=0.9)
parser.add_argument("--corruption_rate", type=float, default=0.02)
parser.add_argument("--save_results", dest="save_results", action="store_true")
parser.set_defaults(save_results=True)
args = parser.parse_args()

logging.info(48 * "=")
logging.info("Running new experiment session")
logging.info(48 * "=")

learner = args.learner
n_repeats = args.n_repeats
hidden_size = args.hidden_size
test_data_size = args.test_data_size
n_epochs = args.n_epochs
evaluation_period = args.evaluation_period
dataset_name = args.dataset_name.lower()
random_seed = args.random_seed
lr = args.step_size
buffer_size = args.buffer_size
tau_unif = args.tau_unif
cst_clip_quants = args.cst_clip_quants
quantile = args.quantile

save_results = args.save_results
corruption_rate = args.corruption_rate
#pareto_param = args.pareto_param

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

logging.info("Lauching experiment with parameters : \n %r" % args)

loader = set_dataloader(dataset_name)
dataset = loader()

dataset.test_size = test_data_size
learning_task = dataset.task
classification_task = learning_task.endswith("classification")

if learning_task.endswith("regression"):
    loss = nn.MSELoss(reduction="sum")
else:
    if learner == "shallow" or learning_task == "multiclass-classification":
        loss = nn.CrossEntropyLoss(reduction="sum")
    else:
        loss = nn.BCEWithLogitsLoss(reduction="sum")

model_names = ["qc", "noclip"] + [f"cst_clip{quant}" for quant in cst_clip_quants]

def repeat(rep):
    logging.info("run #"+str(rep))

    col_try, col_iter, col_algo, col_qc_levels = [], [], [], []
    if learning_task.endswith("classification"):
        col_test_loss, col_acc, col_roc_auc, col_avg_prec = [], [], [], []
    else:
        col_mse, col_mae = [], []

    torch.manual_seed(random_seed + rep)

    X_train, X_test, y_train, y_test = dataset.extract_corrupt2(
        corruption_rate=corruption_rate,
        random_state=random_seed + rep,
    )
    if learner == "shallow":
        output_size = 1 if learning_task.endswith("regression") else len(dataset.classes_)
    else:
        output_size = 1 if learning_task in ["regression", "binary-classification"] else len(dataset.classes_)

    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    if output_size == 1:
        y_train = torch.Tensor(y_train)  # , dtype=torch.long)
        y_test = torch.Tensor(y_test)  # , dtype=torch.long)
    else:
        y_train = torch.Tensor(y_train).long()#, dtype=torch.long)
        y_test = torch.Tensor(y_test).long()#, dtype=torch.long)

    if learning_task == "multiclass-classification":
        lbin = LabelBinarizer()
        lbin.fit_transform(y_train)
        y_test_binary = lbin.transform(y_test)
    if learner == "shallow":
        models = [nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
        )]
    else:
        models = [nn.Sequential(
            nn.Linear(X_train.shape[1], output_size),
        )]

    models[0].train()
    optimizers = [optim.SGD(models[0].parameters(), lr=lr)]
    for _ in model_names[:-1]:
        othermodel = copy.deepcopy((models[0]))
        optimizer = optim.SGD(othermodel.parameters(), lr=lr)
        optimizers.append(optimizer)
        othermodel.train()
        models.append(othermodel)

    buffer = np.zeros(buffer_size)
    ages = np.arange(buffer_size)

    if output_size == 1:
        reshape_if_necessary = lambda x, pred_shape : x.reshape(pred_shape)
    else:
        reshape_if_necessary = lambda x, pred_shape : x

    # Figure out cst clipping levels

    for j in range(buffer_size):
        features, target = X_train[j % len(X_train):j % len(X_train) + 1], y_train[
                                                                           j % len(X_train):j % len(X_train) + 1]
        optimizers[-1].zero_grad()
        pred = models[-1](features)
        loss_val = loss(pred, reshape_if_necessary(target, pred.shape))
        loss_val.backward()
        norm = 0.0
        for p in models[-1].parameters():
            param_norm = p.grad.detach().data.norm(2)
            norm += param_norm.item() ** 2
        update_buffer(np.sqrt(norm), buffer, ages, buffer_size)
    optimizers[-1].zero_grad()
    cst_clip_levels = [buffer[floor(buffer_size * quant)] for quant in cst_clip_quants]

    for j in tqdm(range(n_epochs * len(X_train))):
        # if (j + 1) % len(X_train) == 0 or j == 0:
        #     print("epoch : " + str(1+(j+1)//len(X_train)))
        if (j + 1) % evaluation_period == 0 or j == 0:
            # print(f"evaluation {(j+1)//evaluation_period}")
            for i in range(len(models)):
                col_try.append(rep)
                col_iter.append(j)
                col_algo.append(model_names[i])
                col_qc_levels.append(buffer[floor(len(buffer) * quantile)])
                if classification_task:
                    if learning_task == "binary-classification":
                        acc, roc_auc, avg_prec, lss = binary_classif_metrics(models[i], X_test, y_test)
                    else:
                        acc, roc_auc, avg_prec, lss = multiclassif_metrics(models[i], X_test, y_test, y_test_binary)
                    col_acc.append(acc)
                    col_roc_auc.append(roc_auc)
                    col_avg_prec.append(avg_prec)
                    col_test_loss.append(lss)
                else:
                    mse, mae = regression_metrics(models[i], X_test, y_test)
                    col_mse.append(mse)
                    col_mae.append(mae)

        features, target = X_train[j%len(X_train):j%len(X_train)+1], y_train[j%len(X_train):j%len(X_train)+1]

        # Quantile CLIPPED SGD
        optimizers[0].zero_grad()
        pred = models[0](features)
        loss_val = loss(pred, reshape_if_necessary(target, pred.shape))
        loss_val.backward()
        current_norm = nn.utils.clip_grad_norm_(models[0].parameters(), buffer[floor(len(buffer) * quantile)])
        update_buffer(current_norm.item(), buffer, ages, buffer_size)
        optimizers[0].step()

        # Standard SGD
        for p in models[1].parameters():
            p.requires_grad = False
            torch.clamp(p, min=-MAX_PARAM_VALUE, max=MAX_PARAM_VALUE, out=p)
            p.requires_grad = True
        optimizers[1].zero_grad()
        pred = models[1](features)
        loss_val = loss(pred, reshape_if_necessary(target, pred.shape))
        loss_val.backward()
        # if not classification_task or dataset_name=="covtype":
        #     _ = nn.utils.clip_grad_norm_(models[1].parameters(), tau_unif)
        optimizers[1].step()

        # Constant CLIPPED SGD
        for k in range(len(cst_clip_quants)):
            optimizers[2+k].zero_grad()
            pred = models[2+k](features)
            loss_val = loss(pred, reshape_if_necessary(target, pred.shape))
            loss_val.backward()
            nn.utils.clip_grad_norm_(models[2+k].parameters(), cst_clip_levels[k])
            optimizers[2+k].step()
    if classification_task:
        return col_try, col_iter, col_algo, col_test_loss, col_acc, col_roc_auc, col_avg_prec, col_qc_levels, cst_clip_levels
    else:
        return col_try, col_iter, col_algo, col_mse, col_mae, col_qc_levels, cst_clip_levels

if False:#os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(repeat)(rep) for rep in range(1, n_repeats + 1)
    )
else:
    results = [repeat(rep) for rep in range(1, n_repeats + 1)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_iter = list(itertools.chain.from_iterable([x[1] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[2] for x in results]))
cst_clips_per_seed = {j+1: results[j][8] for j in range(n_repeats)}
if classification_task:
    col_test_loss = list(itertools.chain.from_iterable([x[3] for x in results]))
    col_acc = list(itertools.chain.from_iterable([x[4] for x in results]))
    col_roc_auc = list(itertools.chain.from_iterable([x[5] for x in results]))
    col_avg_prec = list(itertools.chain.from_iterable([x[6] for x in results]))
    col_qc_levels = list(itertools.chain.from_iterable([x[7] for x in results]))
    logging.info("Creating pandas DataFrame")
    data = pd.DataFrame(
        {
            "test_loss": col_test_loss,
            "acc": col_acc,
            "roc_auc": col_roc_auc,
            "avg_prec": col_avg_prec,
            "qc_levels": col_qc_levels,
            "t": col_iter,
        }
    )

else:
    col_mse = list(itertools.chain.from_iterable([x[3] for x in results]))
    col_mae = list(itertools.chain.from_iterable([x[4] for x in results]))
    col_qc_levels = list(itertools.chain.from_iterable([x[5] for x in results]))
    logging.info("Creating pandas DataFrame")
    data = pd.DataFrame(
        {
            "mse": col_mse,
            "mae": col_mae,
            "qc_levels": col_qc_levels,
            "t": col_iter,
        }
    )

#col_eta = list(itertools.chain.from_iterable([x[7] for x in results]))


data["repeat"] = pd.Series(pd.Categorical(col_try))
data["algo"] = pd.Series(pd.Categorical(col_algo, categories=model_names, ordered=False))

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    ensure_directory("exp_archives/" + experiment_name + "/")

    filename = experiment_name + "_" + dataset_name+str(corruption_rate) + "_"+ learner + "_results_" + now + ".pickle"

    with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
        pickle.dump(
            {"datetime": now, "results": data, "args": args, "cst_clips": cst_clips_per_seed}, f
        )

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 2.2

rlpl = sns.relplot(data=data,
                   x="t",
                   y="acc" if classification_task else "mae",
                   # estimator=np.median,
                   # col="eta",
                   hue="algo",
                   # palette=colors,
                   style="algo",
                   # col_wrap=3,
                   kind="line",
                   # legend=False,
                   lw=line_width,
                   aspect=1,  # 0.5 if reg else 1,
                   markevery=200,
                   markers=True,
                   errorbar="sd",
                   markersize=11,
                   facet_kws={'sharey': False, 'sharex': False}).set(yscale="log")

rlpl = sns.relplot(data=data,
                   x="t",
                   y="test_loss" if classification_task else "mse",
                   # estimator=np.median,
                   # col="eta",
                   hue="algo",
                   # palette=colors,
                   style="algo",
                   # col_wrap=3,
                   kind="line",
                   # legend=False,
                   lw=line_width,
                   aspect=1,  # 0.5 if reg else 1,
                   markevery=200,
                   markers=True,
                   errorbar="sd",
                   markersize=11,
                   facet_kws={'sharey': False, 'sharex': False}).set(yscale="log")

# plt.legend(loc='upper center', labels=labels, ncol=len(labels), columnspacing=0.6, bbox_to_anchor=(-0.68, 2.45),
#            title_fontsize=20, fontsize=20, frameon=False)  # title='Algorithm',
#
# for axes in rlpl.axes:
#     for ax in axes:
#         formatter = ticker.FuncFormatter(lambda y, _: '{:.2g}'.format(y))  # ticker.ScalarFormatter()
#         # formatter.set_scientific(False)
#         # ax = rlpl.axes[-1][1]
#         ax.yaxis.set_major_formatter(
#             formatter)  # ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
#         ax.yaxis.set_minor_formatter(
#             formatter)  # ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
#
# _, y_up = rlpl.axes[-1][0].get_ylim()
# rlpl.axes[-1][-1].set_ylim(top=y_up)
# rlpl.set_axis_labels(x_var="Iteration", y_var=metric.upper().replace('_', ' ').capitalize())  # , fontsize=20)
rlpl.fig.suptitle(' n_repeats = %d , step-size = %f' % (n_repeats, lr))

rlpl.fig.subplots_adjust(wspace=.1)

# rlpl.set_titles("{row_name}, CR = {col_name}")  # , size=20)  # use this argument literally

# plt.subplots_adjust(wspace=0.2, hspace=0.2)

plt.show()
plt.tight_layout()

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

ensure_directory("exp_archives/" + experiment_name + "/")

fig_file_name = "exp_archives/" + experiment_name + "/" + experiment_name + "_" + dataset_name + learner + str(corruption_rate) + "_results_" + now + ".pdf"
# fig = ax.get_figure()
# fig.savefig(fname=fig_file_name, bbox_inches="tight")
rlpl.savefig(fname=fig_file_name, bbox_inches="tight")
logging.info("Saved figure into file : %s" % fig_file_name)
