# License: BSD 3 clause


import sys
import os
import subprocess
from joblib import Parallel, delayed, parallel_backend
# from pqdm.threads import pqdm
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from scipy.special import expit, softmax

sys.path.extend([".", ".."])
# from wildwood.wildwood.datasets import (  # noqa: E402
from linlearn._loss import decision_function_factory
from linlearn._utils import np_float
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
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_amazon,
    load_covtype,
    load_kick,
    load_internet,
    load_higgs,
    load_kddcup99,
    load_electrical,
    load_occupancy,
    load_avila,
    load_miniboone,
    load_gas,
    load_eeg,
    load_drybean,
    load_cbm,
    load_metro,
    load_ccpp,
    load_energy,
    load_gasturbine,
    load_casp,
    load_superconduct,
    load_bike,
    load_ovctt,
    load_sgemm,
    load_ypmsd,
    load_nupvotes,
    load_houseprices,
    load_fifa19,
    load_nyctaxi,
    load_wine,
    load_airbnb,
    load_statlog,
    load_arcene,
    load_glaucoma,
    load_gisette,
    load_madelon,
    load_gene_expression,
    load_atp1d,
    load_atp7d
)

from experiment import (  # noqa: E402
    MOM_CGD_Experiment,
    CH_CGD_Experiment,
    CH_GD_Experiment,
    TMEAN_CGD_Experiment,
    LLM_GD_Experiment,
    GMOM_GD_Experiment,
    ERM_GD_Experiment,
    ERM_CGD_Experiment,
    HuberGrad_Experiment,
    Huber_Experiment,
    ModifiedHuber_CGD_Experiment,
    LAD_Experiment,
    RANSAC_Experiment,
    MD_TMEAN_Experiment,
    MD_DKK_Experiment,
    DA_TMEAN_Experiment,
    DA_DKK_Experiment,
    LLC19_TMEAN_Experiment,
    LLC19_MOM_Experiment,
    # TMEAN_HUBER_CGD_Experiment,
)


def set_experiment(
    clf_name,
    learning_task,
    max_hyperopt_eval,
    expe_random_states,
    output_folder_path,
):
    experiment_select = {
        "MOM_CGD": MOM_CGD_Experiment,
        "TMEAN_CGD": TMEAN_CGD_Experiment,
        "CH_CGD": CH_CGD_Experiment,
        "CH_GD": CH_GD_Experiment,
        "LLM_GD": LLM_GD_Experiment,
        "GMOM_GD": GMOM_GD_Experiment,
        "ERM_GD": ERM_GD_Experiment,
        "ERM_CGD": ERM_CGD_Experiment,
        "HG_GD": HuberGrad_Experiment,
        "RANSAC": RANSAC_Experiment,
        "LAD": LAD_Experiment,
        "HUBER": Huber_Experiment,
        "MODIFHUBER": ModifiedHuber_CGD_Experiment,
        "MD_TMEAN": MD_TMEAN_Experiment,
        "MD_DKK": MD_DKK_Experiment,
        "DA_TMEAN": DA_TMEAN_Experiment,
        "DA_DKK": DA_DKK_Experiment,
        "LLC_MOM": LLC19_MOM_Experiment,
        "LLC_TMEAN": LLC19_TMEAN_Experiment,
        # "TMEAN_HUBER_CGD": TMEAN_HUBER_CGD_Experiment,
    }
    return experiment_select[clf_name](
            learning_task,
            max_hyperopt_evals=max_hyperopt_eval,
            random_state=expe_random_states,
            output_folder_path=output_folder_path,
        )


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
        "letter": load_letter,
        "satimage": load_satimage,
        "sensorless": load_sensorless,
        "spambase": load_spambase,
        "amazon": load_amazon,
        "covtype": load_covtype,
        "internet": load_internet,
        "kick": load_kick,
        "kddcup": load_kddcup99,
        "higgs": load_higgs,
        "electrical": load_electrical,
        "occupancy": load_occupancy,
        "avila": load_avila,
        "miniboone": load_miniboone,
        "gas": load_gas,
        "eeg": load_eeg,
        "drybean": load_drybean,
        "cbm": load_cbm,
        "metro": load_metro,
        "ccpp": load_ccpp,
        "energy": load_energy,
        "gasturbine": load_gasturbine,
        "bike": load_bike,
        "casp": load_casp,
        "superconduct": load_superconduct,
        "sgemm": load_sgemm,
        "ovctt": load_ovctt,
        "ypmsd": load_ypmsd,
        "nupvotes": load_nupvotes,
        "houseprices": load_houseprices,
        "fifa19": load_fifa19,
        "nyctaxi": load_nyctaxi,
        "wine": load_wine,
        "airbnb": load_airbnb,
        "statlog": load_statlog,
        "arcene": load_arcene,
        "madelon": load_madelon,
        "gisette": load_gisette,
        "gene_expression": load_gene_expression,
        "glaucoma": load_glaucoma,
        "atp1d": load_atp1d,
        "atp7d": load_atp7d
    }
    return loaders_mapping[dataset_name]


def compute_binary_classif_history(model, X_train, y_train, X_test, y_test, seed):
    total_iter = model.history_.records[0].cursor
    train_decision_function = decision_function_factory(X_train, model.fit_intercept)
    test_decision_function = decision_function_factory(X_test, model.fit_intercept)
    train_inner_prods = np.empty((X_train.shape[0], model.n_classes), dtype=np_float)
    test_inner_prods = np.empty((X_test.shape[0], model.n_classes), dtype=np_float)

    (
        roc_auc_list,
        roc_auc_train_list,
        avg_precision_score_list,
        avg_precision_score_train_list,
        log_loss_list,
        log_loss_train_list,
        accuracy_list,
        accuracy_train_list,
        seed_list,
        time_list,
        sc_prods_list,
        iter_list,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])

    weights_record = model.history_.record_nm("weights").record
    time_record = model.history_.record_nm("time").record
    sc_prods_record = model.history_.record_nm("sc_prods").record

    for i in range(total_iter):
        train_decision_function(weights_record[i], train_inner_prods)
        test_decision_function(weights_record[i], test_inner_prods)

        y_scores = expit(test_inner_prods)
        y_scores_train = expit(train_inner_prods)

        y_pred = (y_scores >= 0.5).astype(int)
        y_pred_train = (y_scores_train >= 0.5).astype(int)

        roc_auc_list.append(roc_auc_score(y_test, y_scores))
        roc_auc_train_list.append(roc_auc_score(y_train, y_scores_train))
        avg_precision_score_list.append(average_precision_score(y_test, y_scores))
        avg_precision_score_train_list.append(
            average_precision_score(y_train, y_scores_train)
        )

        log_loss_list.append(log_loss(y_test, y_scores))
        log_loss_train_list.append(log_loss(y_train, y_scores_train))

        accuracy_list.append(accuracy_score(y_test, y_pred))
        accuracy_train_list.append(accuracy_score(y_train, y_pred_train))

        seed_list.append(seed)
        time_list.append(time_record[i] - time_record[0])
        sc_prods_list.append(sc_prods_record[i])
        iter_list.append(i)
    return (
        seed_list,
        time_list,
        sc_prods_list,
        iter_list,
        roc_auc_list,
        roc_auc_train_list,
        avg_precision_score_list,
        avg_precision_score_train_list,
        log_loss_list,
        log_loss_train_list,
        accuracy_list,
        accuracy_train_list,
    )


def compute_multi_classif_history(model, X_train, y_train, X_test, y_test, seed):
    total_iter = model.history_.records[0].cursor
    train_decision_function = decision_function_factory(X_train, model.fit_intercept)
    test_decision_function = decision_function_factory(X_test, model.fit_intercept)
    train_inner_prods = np.empty((X_train.shape[0], model.n_classes), dtype=np_float)
    test_inner_prods = np.empty((X_test.shape[0], model.n_classes), dtype=np_float)

    (
        roc_auc_list,
        roc_auc_train_list,
        roc_auc_weighted_list,
        roc_auc_weighted_train_list,
        avg_precision_score_list,
        avg_precision_score_train_list,
        avg_precision_score_weighted_list,
        avg_precision_score_weighted_train_list,
        log_loss_list,
        log_loss_train_list,
        accuracy_list,
        accuracy_train_list,
        seed_list,
        time_list,
        sc_prods_list,
        iter_list,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

    lbin = LabelBinarizer()
    y_train_binary = lbin.fit_transform(y_train)
    y_test_binary = lbin.transform(y_test)

    weights_record = model.history_.record_nm("weights").record
    time_record = model.history_.record_nm("time").record
    sc_prods_record = model.history_.record_nm("sc_prods").record

    for i in range(total_iter):
        train_decision_function(weights_record[i], train_inner_prods)
        test_decision_function(weights_record[i], test_inner_prods)

        y_scores = softmax(test_inner_prods, axis=1)
        y_scores_train = softmax(train_inner_prods, axis=1)

        y_pred = np.argmax(y_scores, axis=1)
        y_pred_train = np.argmax(y_scores_train, axis=1)

        roc_auc_list.append(
            roc_auc_score(y_test, y_scores, multi_class="ovr", average="macro")
        )
        roc_auc_train_list.append(
            roc_auc_score(y_train, y_scores_train, multi_class="ovr", average="macro")
        )
        roc_auc_weighted_list.append(
            roc_auc_score(y_test, y_scores, multi_class="ovr", average="weighted")
        )
        roc_auc_weighted_train_list.append(
            roc_auc_score(
                y_train, y_scores_train, multi_class="ovr", average="weighted"
            )
        )

        avg_precision_score_list.append(
            average_precision_score(y_test_binary, y_scores)
        )
        avg_precision_score_train_list.append(
            average_precision_score(y_train_binary, y_scores_train)
        )
        avg_precision_score_weighted_list.append(
            average_precision_score(y_test_binary, y_scores, average="weighted")
        )
        avg_precision_score_weighted_train_list.append(
            average_precision_score(y_train_binary, y_scores_train, average="weighted")
        )
        log_loss_list.append(log_loss(y_test, y_scores))
        log_loss_train_list.append(log_loss(y_train, y_scores_train))
        accuracy_list.append(accuracy_score(y_test, y_pred))
        accuracy_train_list.append(accuracy_score(y_train, y_pred_train))

        seed_list.append(seed)
        time_list.append(time_record[i] - time_record[0])
        sc_prods_list.append(sc_prods_record[i])
        iter_list.append(i)

    return (
        seed_list,
        time_list,
        sc_prods_list,
        iter_list,
        roc_auc_list,
        roc_auc_train_list,
        avg_precision_score_list,
        avg_precision_score_train_list,
        log_loss_list,
        log_loss_train_list,
        accuracy_list,
        accuracy_train_list,
        roc_auc_weighted_list,
        roc_auc_weighted_train_list,
        avg_precision_score_weighted_list,
        avg_precision_score_weighted_train_list,
    )


def compute_regression_history(model, X_train, y_train, X_test, y_test, seed):
    if not hasattr(model, "history_"):
        mse_list = [mean_squared_error(y_test, model.predict(X_test))]
        mse_train_list = [mean_squared_error(y_train, model.predict(X_train))]

        mae_list = [mean_absolute_error(y_test, model.predict(X_test))]
        mae_train_list = [mean_absolute_error(y_train, model.predict(X_train))]

        seed_list, time_list, sc_prods_list = [seed], [0], [0]
        if hasattr(model, "n_iter_"):
            iter_list = [model.n_iter_]
        else: # ransac
            iter_list = [model.n_trials_]
    else:

        total_iter = model.history_.records[0].cursor
        train_decision_function = decision_function_factory(X_train, model.fit_intercept)
        test_decision_function = decision_function_factory(X_test, model.fit_intercept)
        train_inner_prods = np.empty((X_train.shape[0], model.n_classes), dtype=np_float)
        test_inner_prods = np.empty((X_test.shape[0], model.n_classes), dtype=np_float)

        mse_list, mse_train_list, mae_list, mae_train_list = [], [], [], []

        seed_list, time_list, sc_prods_list, iter_list = [], [], [], []

        weights_record = model.history_.record_nm("weights").record
        time_record = model.history_.record_nm("time").record
        sc_prods_record = model.history_.record_nm("sc_prods").record

        for i in range(total_iter):
            train_decision_function(weights_record[i], train_inner_prods)
            test_decision_function(weights_record[i], test_inner_prods)

            y_scores = test_inner_prods
            y_scores_train = train_inner_prods

            mse_list.append(mean_squared_error(y_test, y_scores))
            mse_train_list.append(mean_squared_error(y_train, y_scores_train))

            mae_list.append(mean_absolute_error(y_test, y_scores))
            mae_train_list.append(mean_absolute_error(y_train, y_scores_train))

            seed_list.append(seed)
            time_list.append(time_record[i] - time_record[0])
            sc_prods_list.append(sc_prods_record[i])
            iter_list.append(i)

    return (
        seed_list,
        time_list,
        sc_prods_list,
        iter_list,
        mse_list,
        mse_train_list,
        mae_list,
        mae_train_list,
    )


def run_hyperopt(
    dataset,
    learner_name,
    learning_task,
    corruption_rate,
    max_hyperopt_eval,
    results_dataset_path,
):
    classification = learning_task.endswith("classification")

    col_it_time, col_it_iter, col_it_sc_prods, col_it_seed = [], [], [], []
    col_fin_fit_time, col_fin_seed = [], []
    if classification:

        col_it_roc_auc, col_it_avg_precision_score, col_it_log_loss, col_it_accuracy = (
            [],
            [],
            [],
            [],
        )
        (
            col_it_roc_auc_train,
            col_it_avg_precision_score_train,
            col_it_log_loss_train,
            col_it_accuracy_train,
        ) = ([], [], [], [])

        (
            col_fin_roc_auc,
            col_fin_avg_precision_score,
            col_fin_log_loss,
            col_fin_accuracy,
        ) = ([], [], [], [])
        (
            col_fin_roc_auc_train,
            col_fin_avg_precision_score_train,
            col_fin_log_loss_train,
            col_fin_accuracy_train,
        ) = ([], [], [], [])
        if learning_task == "multiclass-classification":
            (
                col_it_roc_auc_weighted,
                col_it_avg_precision_score_weighted,
                col_it_roc_auc_weighted_train,
                col_it_avg_precision_score_weighted_train,
            ) = ([], [], [], [])
            (
                col_fin_roc_auc_weighted,
                col_fin_avg_precision_score_weighted,
                col_fin_roc_auc_weighted_train,
                col_fin_avg_precision_score_weighted_train,
            ) = ([], [], [], [])

    else:
        col_it_mse, col_it_mse_train, col_it_mae, col_it_mae_train = [], [], [], []
        col_fin_mse, col_fin_mse_train, col_fin_mae, col_fin_mae_train = [], [], [], []

    train_perc = 0.7
    val_perc = 0.15
    test_perc = 0.15
    assert train_perc + val_perc + test_perc == 1.0

    # if dataset.name == "internet":
    #     counts_dict = dict(dataset.df_raw[dataset.label_column].value_counts())
    #     too_few = [k for k in counts_dict.keys() if counts_dict[k] < 20]
    #     dataset.df_raw = dataset.df_raw[~dataset.df_raw[dataset.label_column].isin(too_few)]

    dataset.test_size = val_perc + test_perc
    X_train, X_te, y_train, y_te = dataset.extract_corrupt(
        corruption_rate=corruption_rate, random_state=random_states["data_extract_random_state"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_te,
        y_te,
        test_size=test_perc / (test_perc + val_perc),
        random_state=random_states["train_val_split_random_state"],
        stratify=y_te if learning_task.endswith("classification") else None,
    )

    # if dataset.name == "kick":
    #     X_train = np.nan_to_num(X_train)
    #     X_val = np.nan_to_num(X_val)
    #     X_test = np.nan_to_num(X_test)

    # if dataset.name == "internet":
    #     std_scaler = StandardScaler()
    #     X_train = std_scaler.fit_transform(X_train)
    #     X_val = std_scaler.transform(X_val)
    #     X_test = std_scaler.transform(X_test)


    exp = set_experiment(
        learner_name,
        learning_task,
        max_hyperopt_eval,
        random_states["expe_random_state"],
        results_dataset_path,
    )

    if max_hyperopt_eval > 0:
        print("Run train-val hyperopt exp...")
        tuned_cv_result, best_param = exp.optimize_params(
            X_train,
            y_train,
            X_val,
            y_val,
            max_evals=max_hyperopt_eval,
            verbose=True,
        )
        print("\nThe best found params were : %r\n" % best_param)
    else:
        print("NO PARAMETER FINETUNING, using only default params")
        best_param = exp.default_params

    print("Run fitting with tuned params...")

    for fit_seed in fit_seeds:
        # tic = time()
        model, fit_time = exp.fit(
            best_param,
            X_train,
            y_train,
            seed=fit_seed,
        )
        # toc = time()
        logging.info("Fitted %s in %.2f seconds" % (learner_name, fit_time))

        if classification:
            if learning_task == "binary-classification":
                seed_run = compute_binary_classif_history(
                    model, X_train, y_train, X_test, y_test, fit_seed
                )
            else:
                seed_run = compute_multi_classif_history(
                    model, X_train, y_train, X_test, y_test, fit_seed
                )

            col_it_roc_auc += seed_run[4]
            col_fin_roc_auc.append(seed_run[4][-1])
            col_it_roc_auc_train += seed_run[5]
            col_fin_roc_auc_train.append(seed_run[5][-1])
            col_it_avg_precision_score += seed_run[6]
            col_fin_avg_precision_score.append(seed_run[6][-1])
            col_it_avg_precision_score_train += seed_run[7]
            col_fin_avg_precision_score_train.append(seed_run[7][-1])
            col_it_log_loss += seed_run[8]
            col_fin_log_loss.append(seed_run[8][-1])
            col_it_log_loss_train += seed_run[9]
            col_fin_log_loss_train.append(seed_run[9][-1])
            col_it_accuracy += seed_run[10]
            col_fin_accuracy.append(seed_run[10][-1])
            col_it_accuracy_train += seed_run[11]
            col_fin_accuracy_train.append(seed_run[11][-1])

            if learning_task == "multiclass-classification":
                col_it_roc_auc_weighted += seed_run[12]
                col_fin_roc_auc_weighted.append(seed_run[12][-1])
                col_it_roc_auc_weighted_train += seed_run[13]
                col_fin_roc_auc_weighted_train.append(seed_run[13][-1])
                col_it_avg_precision_score_weighted += seed_run[14]
                col_fin_avg_precision_score_weighted.append(seed_run[14][-1])
                col_it_avg_precision_score_weighted_train += seed_run[15]
                col_fin_avg_precision_score_weighted_train.append(seed_run[15][-1])
        else:  # regression
            seed_run = compute_regression_history(
                model, X_train, y_train, X_test, y_test, fit_seed
            )
            col_it_mse += seed_run[4]
            col_fin_mse.append(seed_run[4][-1])
            col_it_mse_train += seed_run[5]
            col_fin_mse_train.append(seed_run[5][-1])
            col_it_mae += seed_run[6]
            col_fin_mae.append(seed_run[6][-1])
            col_it_mae_train += seed_run[7]
            col_fin_mae_train.append(seed_run[7][-1])

        col_it_seed += seed_run[0]
        col_fin_seed.append(fit_seed)
        col_it_time += seed_run[1]
        col_fin_fit_time.append(fit_time)
        col_it_sc_prods += seed_run[2]
        col_it_iter += seed_run[3]

    if classification:

        iteration_df = pd.DataFrame(
            {
                "roc_auc": col_it_roc_auc,
                "avg_prec": col_it_avg_precision_score,
                "log_loss": col_it_log_loss,
                "accuracy": col_it_accuracy,
                "roc_auc_train": col_it_roc_auc_train,
                "avg_prec_train": col_it_avg_precision_score_train,
                "log_loss_train": col_it_log_loss_train,
                "accuracy_train": col_it_accuracy_train,
            }
        )

        finals_df = pd.DataFrame(
            {
                "roc_auc": col_fin_roc_auc,
                "avg_prec": col_fin_avg_precision_score,
                "log_loss": col_fin_log_loss,
                "accuracy": col_fin_accuracy,
                "roc_auc_train": col_fin_roc_auc_train,
                "avg_prec_train": col_fin_avg_precision_score_train,
                "log_loss_train": col_fin_log_loss_train,
                "accuracy_train": col_fin_accuracy_train,
            }
        )

        if learning_task == "multiclass-classification":
            iteration_df["avg_prec_w"] = col_it_avg_precision_score_weighted
            iteration_df["roc_auc_w"] = col_it_roc_auc_weighted
            iteration_df["roc_auc_w_train"] = col_it_roc_auc_weighted_train
            iteration_df["avg_prec_w_train"] = col_it_avg_precision_score_weighted_train

            finals_df["roc_auc_w"] = col_fin_roc_auc_weighted
            finals_df["avg_prec_w"] = col_fin_avg_precision_score_weighted
            finals_df["roc_auc_w_train"] = col_fin_roc_auc_weighted_train
            finals_df["avg_prec_w_train"] = col_fin_avg_precision_score_weighted_train

            logging.info(
                "AUC= %.2f, AUCW: %.2f, AVGP: %.2f, AVGPW: %.2f, LOGL: %.2f, ACC: %.2f"
                % (
                    float(np.mean(col_fin_roc_auc)),
                    float(np.mean(col_fin_roc_auc_weighted)),
                    float(np.mean(col_fin_avg_precision_score)),
                    float(np.mean(col_fin_avg_precision_score_weighted)),
                    float(np.mean(col_fin_log_loss)),
                    float(np.mean(col_fin_accuracy)),
                )
            )

        else:
            logging.info(
                "AUC= %.2f, AVGP: %.2f, LOGL: %.2f, ACC: %.2f"
                % (
                    float(np.mean(col_fin_roc_auc)),
                    float(np.mean(col_fin_avg_precision_score)),
                    float(np.mean(col_fin_log_loss)),
                    float(np.mean(col_fin_accuracy)),
                )
            )

    else:
        logging.info(
            "MSE= %.2f, MSE_TRAIN: %.2f, MAE= %.2f, MAE_TRAIN: %.2f"
            % (
                float(np.mean(col_fin_mse)),
                float(np.mean(col_fin_mse_train)),
                float(np.mean(col_fin_mae)),
                float(np.mean(col_fin_mae_train)),
            )
        )

        iteration_df = pd.DataFrame(
            {
                "mse": col_it_mse,
                "mse_train": col_it_mse_train,
                "mae": col_it_mae,
                "mae_train": col_it_mae_train,
            }
        )

        finals_df = pd.DataFrame(
            {
                "mse": col_fin_mse,
                "mse_train": col_fin_mse_train,
                "mae": col_fin_mae,
                "mae_train": col_fin_mae_train,
            }
        )

    iteration_df["seed"] = col_it_seed
    iteration_df["time"] = col_it_time
    iteration_df["sc_prods"] = col_it_sc_prods
    iteration_df["iter"] = col_it_iter

    finals_df["seed"] = col_fin_seed
    finals_df["fit_time"] = col_fin_fit_time

    results = {
        "dataset": dataset.name,
        "learner": learner_name,
        "corruption_rate": corruption_rate,
        "iteration_df": iteration_df,
        "finals_df": finals_df,
        "best_parameter": best_param,
    }

    return results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--learner_name",
    choices=[
        "MOM_CGD",
        "TMEAN_CGD",
        # "TMEAN_HUBER_CGD",
        "CH_CGD",
        "CH_GD",
        "LLM_GD",
        "GMOM_GD",
        "ERM_GD",
        "ERM_CGD",
        "HG_GD",
        "RANSAC",
        "LAD",
        "HUBER",
        "MODIFHUBER",
        "MD_TMEAN",
        "MD_DKK",
        "DA_TMEAN",
        "DA_DKK",
        "LLC_MOM",
        "LLC_TMEAN",
    ],
)
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
        "wine",
        "airbnb",
        "statlog",
        "arcene",
        "glaucoma",
        "gene_expression",
        "gisette",
        "madelon"
    ],
)
parser.add_argument("-n", "--hyperopt_evals", type=int, default=50)
parser.add_argument("--n_jobs", type=int, default=1)
parser.add_argument("--n_tuned_runs", type=int, default=10)
parser.add_argument("-o", "--output_folder_path", default=None)
parser.add_argument("--random_state_seed", type=int, default=42)
parser.add_argument("--corruption_rates", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])

args = parser.parse_args()

logging.info("Received parameters : \n %r" % args)

learner_name = args.learner_name
max_hyperopt_eval = args.hyperopt_evals
n_tuned_runs = args.n_tuned_runs
n_jobs = args.n_jobs
dataset_name = args.dataset_name.lower()
loader = set_dataloader(dataset_name)
random_state_seed = args.random_state_seed
corruption_rates = args.corruption_rates

if learner_name in ["ERM_GD", "ERM_CGD", "MODIFHUBER", "LAD"]:
    logging.info("UNTUNED algorithm : %s, no hyperoptimization and single run" % learner_name)
    max_hyperopt_eval = 0
    n_tuned_runs = 1

if args.output_folder_path is None:
    if not os.path.exists("results"):
        os.mkdir("results")
    results_home_path = "results/"
else:
    results_home_path = args.output_folder_path

random_states = {
    "data_extract_random_state": random_state_seed,
    "train_val_split_random_state": 1 + random_state_seed,
    "expe_random_state": 2 + random_state_seed,
}
fit_seeds = list(range(n_tuned_runs))#[0, 1, 2, 3, 4]

logging.info("=" * 128)
dataset = loader()
learning_task = dataset.task

logging.info("Launching experiments for %s" % dataset.name)

if not os.path.exists(results_home_path + dataset.name):
    os.mkdir(results_home_path + dataset.name)
results_dataset_path = results_home_path + dataset.name + "/"

def run_cr(corruption_rate):
    logging.info("Running hyperoptimisation for corruption rate %.2f"%corruption_rate)
    dataset = loader()
    results = run_hyperopt(
        dataset,
        learner_name,
        learning_task,
        corruption_rate,
        max_hyperopt_eval,
        results_dataset_path,
    )

    print(results)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = (
        "exp_hyperopt_"
        + str(max_hyperopt_eval)
        + "_"
        + dataset_name + str(corruption_rate)
        + "_"
        + learner_name
        + "_"
        + now
        + ".pickle"
    )

    with open(results_dataset_path + filename, "wb") as f:
        pkl.dump(
            {
                "datetime": now,
                "commit": commit,
                "max_hyperopt_eval": max_hyperopt_eval,
                "results": results,
            },
            f,
        )

    logging.info("Saved results in file %s" % results_dataset_path + filename)


with parallel_backend('threading', n_jobs=n_jobs):
    Parallel()(delayed(run_cr)(corruption_rate) for corruption_rate in corruption_rates)

# pqdm(corruption_rates, run_cr, n_jobs=n_jobs)