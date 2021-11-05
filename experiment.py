# License: BSD 3 clause


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time
from datetime import datetime
import pickle as pkl
import sys

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    mean_squared_error
)
from sklearn.preprocessing import LabelBinarizer

from linlearn import Classifier, Regressor
sys.path.extend([".", ".."])


# TODO: to add regressors for every Experiment

min_mom_block_size = 0.001
default_mom_block_size = 0.07
max_mom_block_size = 0.2

min_tmean_percentage = 0.001
default_tmean_percentage = 0.01
max_tmean_percentage = 0.3

min_C_reg = np.log(1e-4)
max_C_reg = np.log(1e4)


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
        bst = self.fit(
            params, X_train, y_train, seed=None
        )
        fit_time = bst.fit_time()#time.time() - start_time
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
        #  X_val, y_val not used since no early stopping
        if seed is not None:
            params.update({"random_state": seed})

        if self.learning_task.endswith("classification"):
            learner = Classifier(**params, n_jobs=-1)
        else:
            learner = Regressor(**params, n_jobs=-1)

        learner.fit(X_train, y_train, dummy_first_step=True)
        return learner

    def predict(self, bst, X_test):
        if self.learning_task.endswith("classification"):
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds

    def preprocess_params(self, params):
        raise NotImplementedError("Method preprocess_params is not implemented.")


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
            "block_size": hp.uniform("block_size", min_mom_block_size, max_mom_block_size),
            "cgd_IS": hp.choice("cgd_IS", [True, False]),
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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
            "eps": hp.loguniform("eps", -10, -1),
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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
            "eps": hp.loguniform("eps", -10, -1),
            "cgd_IS": hp.choice("cgd_IS", [True, False]),
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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
            "block_size": hp.uniform("block_size", min_mom_block_size, max_mom_block_size),
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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
            "block_size" : hp.uniform("block_size", min_mom_block_size, max_mom_block_size),
            "C": hp.loguniform("C", min_C_reg, max_C_reg),
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

