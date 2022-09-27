# License: BSD 3 clause


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time
from datetime import datetime
import pickle as pkl
import sys
from scipy.special import expit

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    mean_squared_error
)
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import HuberRegressor, RANSACRegressor, LinearRegression, SGDRegressor

from liuliu19 import liuliu19_solver

from linlearn import Classifier, Regressor

from numba import jit, objmode

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

sys.path.extend([".", ".."])


# TODO: to add regressors for every Experiment

min_mom_block_size = np.log(1e-5)
default_mom_block_size = 0.07
max_mom_block_size = np.log(0.2)

min_tmean_percentage = 1e-5
default_tmean_percentage = 0.01
max_tmean_percentage = 0.3

min_sparsity = 0.005
default_sparsity = 0.01
max_sparsity = 0.1

stage_length = 20
max_iter = 100

# min_C_reg = np.log(1e-4)
# max_C_reg = np.log(1e4)


class Experiment(object):
    """
    Base class, for hyper-parameters optimization experiments with hyperopt
    """

    def __init__(
        self,
        learning_task="classification",
        hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
        verbose=True,
    ):
        self.learning_task = learning_task
        self.hyperopt_evals = hyperopt_evals

        self.best_loss = np.inf

        self.output_folder_path = os.path.join(output_folder_path, "")
        self.default_params, self.best_params = None, None
        self.random_state = random_state

        self.verbose = verbose

        # to specify definitions in particular experiments
        self.title = None
        self.space = None
        self.trials = None

        if self.learning_task in ["binary-classification", "multiclass-classification"]:
            self.metric = "logistic"
        elif self.learning_task == "regression":
            self.metric = "leastsquares"
        else:
            raise ValueError('Task must be "classification" or "regression"')

    def optimize_params(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        max_evals=None,
        verbose=True,
    ):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(
            fn=lambda params: self.run(
                X_train,
                y_train,
                X_val,
                y_val,
                params,
                verbose=verbose,
            ),
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            rstate=np.random.RandomState(self.random_state)
        )

        self.best_params = self.trials.best_trial["result"]["params"]

        if self.verbose:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = (
                "best_params_results_"
                + str(self.hyperopt_evals)
                + "_"
                + now
                + ".pickle"
            )

            with open(self.output_folder_path + filename, "wb") as f:
                pkl.dump(
                    {
                        "datetime": now,
                        "max_hyperopt_eval": self.hyperopt_evals,
                        "result": self.trials.best_trial["result"],
                    },
                    f,
                )
        return self.trials.best_trial["result"], self.best_params

    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        params=None,
        verbose=False,
    ):
        params = params or self.default_params
        params = self.preprocess_params(params)
        # start_time = time.time()
        bst, fit_time = self.fit(
            params, X_train, y_train, seed=None
        )
        #time.time() - start_time
        y_scores = self.predict(bst, X_val)
        if self.learning_task.endswith("classification"):
            evals_result = log_loss(y_val, y_scores)
            y_pred = np.argmax(y_scores, axis=1) if y_scores.ndim > 1 else (y_scores > 0.5).astype(int)
        else:
            evals_result = mean_squared_error(y_val, y_scores)

        results = {
            "loss": evals_result,
            "fit_time": fit_time,
            "status": STATUS_FAIL if np.isnan(evals_result) else STATUS_OK,
            "params": params.copy(),
        }

        if self.learning_task == "binary-classification":
            roc_auc = roc_auc_score(y_val, y_scores[:, 1])
            roc_auc_weighted = roc_auc
            avg_precision_score = average_precision_score(y_val, y_scores[:, 1])
            avg_precision_score_weighted = avg_precision_score
            log_loss_ = log_loss(y_val, y_scores)
            accuracy = accuracy_score(y_val, y_pred)
            results.update(
                {
                    "roc_auc": roc_auc,
                    "roc_auc_weighted": roc_auc_weighted,
                    "avg_precision_score": avg_precision_score,
                    "avg_precision_score_weighted": avg_precision_score_weighted,
                    "log_loss": log_loss_,
                    "accuracy": accuracy,
                }
            )
        elif self.learning_task == "multiclass-classification":
            y_val_binary = LabelBinarizer().fit_transform(y_val)
            if not np.isfinite(y_scores).all():
                print("Found NaN /inf in scores")
                print(y_scores)
            roc_auc = roc_auc_score(y_val, y_scores, multi_class="ovr", average="macro")
            roc_auc_weighted = roc_auc_score(
                y_val, y_scores, multi_class="ovr", average="weighted"
            )
            avg_precision_score = average_precision_score(y_val_binary, y_scores)
            avg_precision_score_weighted = average_precision_score(
                y_val_binary, y_scores, average="weighted"
            )
            log_loss_ = log_loss(y_val, y_scores)
            accuracy = accuracy_score(y_val, y_pred)
            results.update(
                {
                    "roc_auc": roc_auc,
                    "roc_auc_weighted": roc_auc_weighted,
                    "avg_precision_score": avg_precision_score,
                    "avg_precision_score_weighted": avg_precision_score_weighted,
                    "log_loss": log_loss_,
                    "accuracy": accuracy,
                }
            )

        self.best_loss = min(self.best_loss, results["loss"])
        self.hyperopt_eval_num += 1
        self.random_state += 1  # change random state after a run
        results.update(
            {"hyperopt_eval_num": self.hyperopt_eval_num, "best_loss": self.best_loss}
        )

        if verbose:
            print(
                "[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}\nparams={6}".format(
                    self.hyperopt_eval_num,
                    self.hyperopt_evals,
                    fit_time,
                    self.metric,
                    results["loss"],
                    self.best_loss,
                    results["params"]
                )
            )
        return results


    def fit(
        self,
        params,
        X_train,
        y_train,
        seed=None,
    ):
        if seed is not None:
            params.update({"random_state": seed})

        if self.learning_task.endswith("classification"):
            learner = Classifier(**params, n_jobs=-1)
        else:
            learner = Regressor(**params, n_jobs=-1)

        learner.fit(X_train, y_train, dummy_first_step=True)
        return learner, learner.fit_time()

    def predict(self, bst, X_test):
        if self.learning_task.endswith("classification"):
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds

    # def preprocess_params(self, params):
    #     raise NotImplementedError("Method preprocess_params is not implemented.")

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "random_state": self.random_state,
            }
        )
        return params_


class MOM_CGD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "block_size": hp.loguniform("block_size", min_mom_block_size, max_mom_block_size),
            "cgd_IS": hp.choice("cgd_IS", [True, False]),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"block_size": default_mom_block_size, "cgd_IS": False}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "mom_cgd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "mom",
                "solver": "cgd",
                "random_state": self.random_state,
            }
        )
        return params_

class ERM_GD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"C": 1}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "erm_gd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "erm",
                "solver": "gd",
                "random_state": self.random_state,
            }
        )
        return params_

class ERM_CGD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"C": 1}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "erm_cgd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "erm",
                "solver": "cgd",
                "random_state": self.random_state,
            }
        )
        return params_

class ModifiedHuber_CGD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"C": 1}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "modifiedhuber_cgd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "erm",
                "solver": "cgd",
                "loss": "modifiedhuber",
                "random_state": self.random_state,
            }
        )
        return params_

class TMEAN_CGD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "cgd_IS": hp.choice("cgd_IS", [True, False]),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "cgd_IS": False}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "tmean_cgd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "tmean",
                "solver": "cgd",
                "random_state": self.random_state,
            }
        )
        return params_

# class TMEAN_HUBER_CGD_Experiment(Experiment):
#
#     def __init__(
#         self,
#         learning_task,
#         max_hyperopt_evals=50,
#         random_state=0,
#         output_folder_path="./",
#     ):
#         Experiment.__init__(
#             self,
#             learning_task,
#             max_hyperopt_evals,
#             random_state,
#             output_folder_path,
#         )
#         if learning_task != "regression":
#             raise ValueError("RANSAC is only used for regression")
#
#         # hard-coded params search space here
#         self.space = {
#             "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
#             "cgd_IS": hp.choice("cgd_IS", [True, False]),
#             # "C": hp.loguniform("C", min_C_reg, max_C_reg),
#         }
#         # hard-coded default params here
#         self.default_params = {"percentage": default_tmean_percentage, "cgd_IS": False}
#         self.default_params = self.preprocess_params(self.default_params)
#         self.title = "tmean_cgd"
#
#     def preprocess_params(self, params):
#         params_ = params.copy()
#         params_.update(
#             {
#                 "estimator": "tmean",
#                 "solver": "cgd",
#                 "loss": "huber",
#                 "random_state": self.random_state,
#             }
#         )
#         return params_

class CH_GD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "eps": hp.loguniform("eps", -10, 0),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"eps": 0.001}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "ch_gd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "ch",
                "solver": "gd",
                "random_state": self.random_state,
            }
        )
        return params_

class CH_CGD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "eps": hp.loguniform("eps", -10, 0),
            "cgd_IS": hp.choice("cgd_IS", [True, False]),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"eps": 0.001, "cgd_IS" : False}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "ch_cgd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "ch",
                "solver": "cgd",
                "random_state": self.random_state,
            }
        )
        return params_

class LLM_GD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "block_size": hp.loguniform("block_size", min_mom_block_size, max_mom_block_size),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"block_size": default_mom_block_size}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "llm_gd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "llm",
                "solver": "gd",
                "random_state": self.random_state,
            }
        )
        return params_


class GMOM_GD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "block_size" : hp.loguniform("block_size", min_mom_block_size, max_mom_block_size),
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"block_size": default_mom_block_size}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "gmom_gd"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "gmom",
                "solver": "gd",
                "random_state": self.random_state,
            }
        )
        return params_

class HuberGrad_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "hubergrad"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "hg",
                "solver": "gd",
                "random_state": self.random_state,
            }
        )
        return params_


class RANSAC_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )
        if learning_task != "regression":
            raise ValueError("RANSAC is only used for regression")

        # hard-coded params search space here
        self.space = {
            "min_samples": 5 + hp.randint("min_samples", 100)
        }
        # hard-coded default params here
        self.default_params = {}#"min_samples": 5}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "ransac"

    def fit(
            self,
            params,
            X_train,
            y_train,
            seed=None,
    ):
        #  X_val, y_val not used since no early stopping
        if seed is not None:
            params.update({"random_state": seed})

        reg = RANSACRegressor(**params)
        t0 = time.time()
        reg.fit(X_train, y_train)
        fit_time = time.time() - t0

        return reg, fit_time

class LAD_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )
        if learning_task != "regression":
            raise ValueError("LAD is only used for regression")

        # hard-coded params search space here
        self.space = {
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"loss": "epsilon_insensitive", "epsilon": 0.0, "alpha": 0.0}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "lad"

    def fit(
            self,
            params,
            X_train,
            y_train,
            seed=None,
    ):
        if seed is not None:
            params.update({"random_state": seed})

        reg = SGDRegressor(**params)
        t0 = time.time()
        reg.fit(X_train, y_train)
        fit_time = time.time() - t0

        return reg, fit_time


class Huber_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )
        if learning_task != "regression":
            raise ValueError("Huber is only used for regression")

        # hard-coded params search space here
        self.space = {
            "epsilon": hp.uniform("epsilon", 1.0, 2.5)
            # "C": hp.loguniform("C", min_C_reg, max_C_reg),
        }
        # hard-coded default params here
        self.default_params = {"epsilon": 1.35, "alpha": 0.0}
        # self.default_params = self.preprocess_params(self.default_params)
        self.title = "huber"

    def fit(
            self,
            params,
            X_train,
            y_train,
            seed=None,
    ):
        # if seed is not None:
        #     params.update({"random_state": seed})

        reg = HuberRegressor(**params)
        t0 = time.time()
        reg.fit(X_train, y_train)
        fit_time = time.time() - t0

        return reg, fit_time

    def preprocess_params(self, params):
        params_ = params.copy()
        return params_

class MD_TMEAN_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "sparsity_ub": default_sparsity}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "md_tmean"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "tmean",
                "stage_length": stage_length,
                "solver": "md",
                "random_state": self.random_state,
            }
        )
        return params_

class MD_DKK_Experiment(Experiment):

    def __init__(
            self,
            learning_task,
            max_hyperopt_evals=50,
            random_state=0,
            output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "sparsity_ub": default_sparsity}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "md_dkk"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "dkk",
                "solver": "md",
                "stage_length": stage_length,
                "random_state": self.random_state,
            }
        )
        return params_

class DA_TMEAN_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "sparsity_ub": default_sparsity}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "da_tmean"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "tmean",
                "stage_length": stage_length,
                "solver": "da",
                "random_state": self.random_state,
            }
        )
        return params_

class DA_DKK_Experiment(Experiment):

    def __init__(
            self,
            learning_task,
            max_hyperopt_evals=50,
            random_state=0,
            output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "sparsity_ub": default_sparsity}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "da_dkk"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "dkk",
                "solver": "da",
                "stage_length": stage_length,
                "random_state": self.random_state,
            }
        )
        return params_

# def get_llc_deriv_func(learning_task):
#     if learning_task == "regression":
#         @jit(**jit_kwargs)
#         def drv(y1, y2):
#             return y1 - y2
#
#         return drv
#     elif learning_task == "binary-classification":
#         @jit(inline="always", **jit_kwargs)
#         def drv(z, y):
#             return -y * expit(-y * z)  # sigmoid(-y * z[0])#
#
#         return drv
#     else:  # multiclass classification
#
#         @jit(**jit_kwargs)
#         def drv(y, z):
#             n_classes = len(z)
#             out = np.empty_like(z)
#             max_z = z[0]
#             for k in range(1, n_classes):
#                 if z[k] > max_z:
#                     max_z = z[k]
#
#             norm = 0.0
#             for k in range(n_classes):
#                 out[k] = np.exp(z[k] - max_z)  # not much speed difference between exp and np.exp
#                 norm += out[k]
#             for k in range(n_classes):
#                 out[k] /= norm
#             # out /= out.sum()
#
#             out[y] -= 1
#             return out
#
#         return drv


class LLC19_TMEAN_Experiment(Experiment):
    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "percentage": hp.uniform("percentage", min_tmean_percentage, max_tmean_percentage),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"percentage": default_tmean_percentage, "sparsity_ub": default_sparsity}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "llc19_tmean"


    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "tmean",
                "solver": "llc19",
                "random_state": self.random_state,
            }
        )
        return params_

class LLC19_MOM_Experiment(Experiment):

    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            max_hyperopt_evals,
            random_state,
            output_folder_path,
        )


        # hard-coded params search space here
        self.space = {
            "block_size": hp.loguniform("block_size", min_mom_block_size, max_mom_block_size),
            "sparsity_ub": hp.uniform("sparsity_ub", min_sparsity, max_sparsity),
        }
        # hard-coded default params here
        self.default_params = {"block_size": default_mom_block_size, "sparsity_ub": default_sparsity}
        # self.default_params = self.preprocess_params(self.default_params)
        self.title = "llc19_mom"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {
                "estimator": "mom",
                "solver": "llc19",
                "random_state": self.random_state,
            }
        )
        return params_

