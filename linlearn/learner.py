# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
#          Ibrahim Merad <imerad7@gmail.com>
# License: BSD 3 clause

# Parts of the code below are directly from scikit-learn, in particular from
# sklearn/linear_model/_logistic.py

from warnings import warn

import numbers
import numpy as np
from scipy.special import expit, softmax
from numba import jit

from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

from ._loss import (
    Logistic,
    MultiLogistic,
    LeastSquares,
    Huber,
    ModifiedHuber,
    MultiModifiedHuber,
    SquaredHinge,
    MultiSquaredHinge,
    compute_steps,
    compute_steps_cgd,
    decision_function_factory,
)
from ._penalty import NoPen, L2Sq, L1, ElasticNet
from .solver import CGD, GD, SGD, SVRG, SAGA, batch_GD, History
from .estimator import ERM, MOM, TMean, LLM, GMOM, CH, HG
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, np_float, numba_seed_numpy

jit_kwargs = {
    "nopython": NOPYTHON,
    "nogil": NOGIL,
    "boundscheck": BOUNDSCHECK,
    "fastmath": FASTMATH,
}


# TODO: serialization


class BaseLearner(ClassifierMixin, BaseEstimator):
    _losses = [
        "logistic",
        "leastsquares",
        "huber",
        "modifiedhuber",
        "multimodifiedhuber",
        "multilogistic",
        "squaredhinge",
        "multisquaredhinge",
    ]
    _penalties = ["none", "l2", "l1", "elasticnet"]
    _estimators = ["erm", "mom", "tmean", "llm", "gmom", "ch", "hg"]
    _solvers = ["cgd", "gd", "sgd", "svrg", "saga", "batch_gd"]

    def __init__(
        self,
        *,
        penalty="l2",
        C=1.0,
        step_size=1.0,
        loss="logistic",
        fit_intercept=True,
        estimator="erm",
        block_size=0.07,
        percentage=0.01,
        eps=0.001,
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.5,
        sgd_exponent=0.5,
        cgd_IS=False,
    ):
        self.penalty = penalty
        self.C = C
        self.step_size = step_size
        self.loss = loss
        self.estimator = estimator
        self.block_size = block_size
        self.percentage = percentage
        self.eps = eps
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.sgd_exponent = sgd_exponent
        self.cgd_IS = cgd_IS

        self.history_ = None
        self.intercept_ = None
        self.coef_ = None
        self.optimization_result_ = None
        self.n_iter_ = None
        self.classes_ = None

        self.check_estimator_solver_combination(estimator, solver)

        if random_state is not None:
            # seed numba's random generator
            numba_seed_numpy(random_state)
            # randomness is not confined to numba jitted code, need to seed numpy's rng as well
            np.random.seed(random_state)

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, val):
        if val not in BaseLearner._penalties:
            raise ValueError(
                "penalty must be one of %r; got (penalty=%r)" % (self._penalties, val)
            )
        else:
            self._penalty = val

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        if not isinstance(val, numbers.Real) or val < 0:
            raise ValueError("C must be a positive number; got (C=%r)" % val)
        else:
            self._C = float(val)

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, val):
        if not isinstance(val, numbers.Real) or val <= 0:
            raise ValueError(
                "step_size must be a positive number; got (step_size=%r)" % val
            )
        else:
            self._step_size = float(val)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        if val not in BaseLearner._losses:
            raise ValueError(
                "loss must be one of %r; got (loss=%r)" % (self._losses, val)
            )
        else:
            self._loss = val

    @property
    def fit_intercept(self):
        return self._fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, val):
        if not isinstance(val, bool):
            raise ValueError("fit_intercept must be True or False; got (C=%r)" % val)
        else:
            self._fit_intercept = val

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, val):
        if val not in BaseLearner._estimators:
            raise ValueError(
                "estimator must be one of %r; got (estimator=%r)"
                % (self._estimators, val)
            )
        else:
            self._estimator = val

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, val):
        if not isinstance(val, numbers.Real) or val <= 0.0 or val > 1:
            raise ValueError("block_size must be in (0, 1]; got (block_size=%r)" % val)
        else:
            self._block_size = val

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, val):
        if not isinstance(val, numbers.Real) or val < 0.0 or val >= 0.5:
            raise ValueError("percentage must be in [0, 0.5); got (percentage=%r)" % val)
        else:
            self._percentage = val

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, val):
        if not isinstance(val, numbers.Real) or val <= 0.0 or val > 1:
            raise ValueError("eps must be in (0, 1]; got (eps=%r)" % val)
        else:
            self._eps = val

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        if val not in BaseLearner._solvers:
            raise ValueError(
                "solver must be one of %r; got (solver=%r)" % (self._solvers, val)
            )
        else:
            self._solver = val

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, val):
        if not isinstance(val, numbers.Real) or val < 0.0:
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "non negative; got (tol=%r)" % val
            )
        else:
            self._tol = val

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, val):
        if not isinstance(val, numbers.Real) or val <= 0:
            raise ValueError(
                "Maximum number of iteration must be positive;"
                " got (max_iter=%r)" % val
            )
        else:
            self._max_iter = int(val)

    @property
    def l1_ratio(self):
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, val):
        if not isinstance(val, numbers.Real) or val < 0.0 or val > 1.0:
            raise ValueError("l1_ratio must be in (0, 1]; got (l1_ratio=%r)" % val)
        else:
            self._l1_ratio = val

    @property
    def sgd_exponent(self):
        return self._sgd_exponent

    @sgd_exponent.setter
    def sgd_exponent(self, val):
        if not isinstance(val, numbers.Real) or val < 0.5 or val > 1:
            raise ValueError(
                "sgd_exponent must be between 0.5 and 1; got (sgd_exponent=%r)" % val
            )
        else:
            self._sgd_exponent = float(val)

    @property
    def cgd_IS(self):
        return self._cgd_IS

    @cgd_IS.setter
    def cgd_IS(self, val):
        if not isinstance(val, bool):
            raise ValueError("cgd_IS must be a boolean; got (cgd_IS=%r)" % val)
        else:
            self._cgd_IS = val

    # TODO: properties for class_weight=None, random_state=None, verbose=0, warm_start=False, n_jobs=None

    def check_estimator_solver_combination(self, estimator, solver):
        if solver in ["sgd", "svrg", "saga", "batch_gd"] and estimator != "erm":
            warn(
                "Your choice of robust estimator will be ignored because it is not supported by SGD type solvers (SGD, SVRG and SAGA)"
            )
        elif solver == "gd" and estimator == "mom":
            warn(
                "The Median-of-Means estimator computes only single gradient coordinates, a full gradient can be constituted for Gradient Descent but we recommend to either use mom estimator with cgd solver or gmom estimator with gd solver instead"
            )
        # elif solver == "gd" and estimator == "catoni":
        #     warn(
        #         "The Catoni estimator computes only single gradient coordinates, doing Gradient Descent with this estimator is equivalent to using holland estimator with GD. Switching estimator to 'holland'"
        #     )
        #     self.estimator = "holland"
        # elif solver == "cgd" and estimator == "holland":
        #     warn(
        #         "The Holland estimator computes full gradients, doing CGD with this estimator is equivalent to using catoni estimator with CGD. Switching estimator to 'catoni'"
        #     )
        #     self.estimator = "catoni"
        elif solver == "gd" and estimator == "tmean":
            warn(
                "The Trimmed Mean estimator computes only single gradient coordinates, full gradients for GD will be constituted from coordinates."
            )
        elif solver == "cgd" and estimator in ["gmom", "hg"]:
            raise ValueError(
                "The GMOM and HG estimators compute whole gradients and cannot be used with CGD."
            )

    def _get_loss(self):
        if self.loss == "logistic":
            return Logistic()
        elif self.loss == "leastsquares":
            return LeastSquares()
        elif self.loss == "huber":
            return Huber()
        elif self.loss == "modifiedhuber":
            return ModifiedHuber()
        elif self.loss == "multimodifiedhuber":
            return MultiModifiedHuber(self.n_classes)
        elif self.loss == "multilogistic":
            return MultiLogistic(self.n_classes)
        elif self.loss == "squaredhinge":
            return SquaredHinge()
        elif self.loss == "multisquaredhinge":
            return MultiSquaredHinge(self.n_classes)
        else:
            raise ValueError("Loss unknown")

    def _get_estimator(self, X, y, loss):
        if self.estimator == "erm":
            return ERM(X, y, loss, self.n_classes, self.fit_intercept)
        elif self.estimator == "mom":
            n_samples = y.shape[0]
            n_samples_in_block = max(int(self.block_size * n_samples), 1)
            return MOM(
                X, y, loss, self.n_classes, self.fit_intercept, n_samples_in_block
            )
        elif self.estimator == "tmean":
            return TMean(
                X, y, loss, self.n_classes, self.fit_intercept, self.percentage
            )
        elif self.estimator == "ch":
            return CH(X, y, loss, self.n_classes, self.fit_intercept, self.eps)
        elif self.estimator == "llm":
            return LLM(
                X, y, loss, self.n_classes, self.fit_intercept, max(int(1 / self.block_size), 1)
            )
        elif self.estimator == "gmom":
            n_samples = y.shape[0]
            n_samples_in_block = max(int(self.block_size * n_samples), 1)
            return GMOM(
                X, y, loss, self.n_classes, self.fit_intercept, n_samples_in_block
            )
        elif self.estimator == "hg":
            return HG(
                X, y, loss, self.n_classes, self.fit_intercept, eps=self.percentage
            )
        else:
            raise ValueError("Unknown estimator")

    # TODO: get penalty

    def _get_penalty(self, n_samples):
        strength = 1 / (self.C * n_samples)
        if self.penalty == "l2":
            return L2Sq(strength)
        elif self.penalty == "l1":
            return L1(strength)
        elif self.penalty == "none":
            return NoPen(strength)
        elif self.penalty == "elasticnet":
            return ElasticNet(strength, self.l1_ratio)
        else:
            raise ValueError("Unknown penalty")

    def _get_solver(self, X, y):
        n_samples, n_features = X.shape

        # # Get the loss object
        # loss_factory = losses_factory[self.loss]
        # loss = loss_factory()

        # Get the loss object
        loss = self._get_loss()
        # Get the estimator object
        estimator = self._get_estimator(X, y, loss)
        # Get the penalty object
        penalty = self._get_penalty(n_samples)
        # penalty_factory = penalties_factory[self.penalty]
        # The strength is scaled using following scikit-learn's scaling
        # strength = 1 / (self.C * n_samples)
        # penalty = penalty_factory(strength=strength, l1_ratio=self.l1_ratio)
        n_samples_in_block = max(int(n_samples * self.block_size), 1)

        if self.solver == "cgd":
            step = compute_steps_cgd(X, self.estimator, self.fit_intercept, loss.lip, self.percentage,
                                     n_samples_in_block, self.eps)
        else:
            step = compute_steps(X, self.solver, self.estimator, self.fit_intercept, loss.lip, self.percentage,
                                 max(1, int(1 / self.block_size)), self.eps)

        step *= self.step_size

        if self.solver == "cgd":
            # Create an history object for the solver
            history = History("CGD", self.max_iter, self.verbose)
            self.history_ = history

            return CGD(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
                importance_sampling=self.cgd_IS,
            )

        elif self.solver == "gd":
            # Create an history object for the solver
            history = History("GD", self.max_iter, self.verbose)
            self.history_ = history

            return GD(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
            )
        elif self.solver == "sgd":
            # Create an history object for the solver
            history = History("SGD", self.max_iter, self.verbose)
            self.history_ = history

            return SGD(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
                exponent=self.sgd_exponent,
            )

        elif self.solver == "svrg":
            # Create an history object for the solver
            history = History("SVRG", self.max_iter, self.verbose)
            self.history_ = history

            return SVRG(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
            )
        elif self.solver == "saga":
            # Create an history object for the solver
            history = History("SAGA", self.max_iter, self.verbose)
            self.history_ = history

            return SAGA(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
            )
        elif self.solver == "batch_gd":
            # Create an history object for the solver
            history = History("batch_GD", self.max_iter, self.verbose)
            self.history_ = history

            return batch_GD(
                X,
                y,
                loss,
                self.n_classes,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                step,
                history,
                batch_size=self.block_size,
            )

        else:
            raise NotImplementedError("%s is not implemented yet" % self.solver)

    def _get_initial_iterate(self, X, y):
        # Deal with warm-starting here
        n_samples, n_features = X.shape
        if self.fit_intercept:
            w = np.zeros(
                (n_features + int(self.fit_intercept), self.n_classes),
                dtype=X.dtype,
                order="F",
            )
        else:
            w = np.zeros((n_features, self.n_classes), dtype=X.dtype, order="F")
        return w

    def _check_multiclass_loss(self):
        if self.loss == "logistic":
            # if we are in the multiclass case switch to multiclass loss
            self.loss = "multilogistic"
        elif self.loss == "multilogistic":
            pass
        elif self.loss == "squaredhinge":
            # if we are in the multiclass case switch to multiclass loss
            self.loss = "multisquaredhinge"
        elif self.loss == "multisquaredhinge":
            pass
        elif self.loss == "modifiedhuber":
            # if we are in the multiclass case switch to multiclass loss
            self.loss = "multimodifiedhuber"
        elif self.loss == "multimodifiedhuber":
            pass
        else:
            raise ValueError(
                "You should specify a classification loss, got loss=%s" % self.loss
            )

    def _check_binary_loss(self):
        if self.loss == "multilogistic":
            self.loss = "logistic"
        elif self.loss == "logistic":
            pass
        elif self.loss == "multisquaredhinge":
            self.loss = "squaredhinge"
        elif self.loss == "squaredhinge":
            pass
        elif self.loss == "multimodifiedhuber":
            self.loss = "modifiedhuber"
        elif self.loss == "modifiedhuber":
            pass
        else:
            raise ValueError(
                "You should specify a classification loss, got loss=%s" % self.loss
            )

    def _check_regression_loss(self):
        if self.loss not in ["leastsquares", "huber"]:
            raise ValueError(
                "You should specify a regression loss, got loss=%s" % self.loss
            )

    def objective_factory(self, y):

        value_loss = self._get_loss().value_batch_factory()
        value_penalty = (self._get_penalty(len(y))).value_factory()
        if self.fit_intercept:

            @jit(**jit_kwargs)
            def objective(weights, inner_products):
                obj = value_loss(y, inner_products)
                obj += value_penalty(weights[1:])
                return obj

            return objective
        else:

            @jit(**jit_kwargs)
            def objective(weights, inner_products):
                obj = value_loss(y, inner_products)
                obj += value_penalty(weights)
                return obj

            return objective

    def fit_time(self):
        # TODO : check_is_fitted is not throwing an error when it should
        check_is_fitted(self)
        time_record = self.history_.records[1]
        return time_record.record[time_record.cursor-1] - time_record.record[0]


    def compute_objective_history(self, X, y, metric="objective"):
        self.history_.allocate_record(1, metric)
        new_record = self.history_.records[-1]
        fit_intercept = self.fit_intercept
        inner_products = np.empty((X.shape[0], self.n_classes), dtype=np_float)

        # decision function
        decision_function = decision_function_factory(X, fit_intercept)

        if metric == "objective":
            # Get the objective function
            obj = self.objective_factory(y)
            metric_fct = lambda w, i_p, y: obj(w, i_p)
        elif metric == "misclassif_rate":
            if self.__class__.__name__ != "Classifier":
                raise ValueError("Cannot compute misclassification rates for Regressor")

            from sklearn.metrics import accuracy_score
            classes = self.classes_
            def metric_fct(w, i_p, y):
                if i_p.ndim == 1:
                    indices = (i_p > 0).astype(int)
                else:
                    indices = i_p.argmax(axis=1)
                predictions = classes[indices]
                return 1 - accuracy_score(y, predictions)
        else:
            raise ValueError("Unknown metric %r"%metric)

        parameter_record = self.history_.records[0]
        total_iter = parameter_record.cursor

        for i in range(total_iter):  # totalitaire
            weights = parameter_record.record[i]
            decision_function(weights, inner_products)
            new_record.update(metric_fct(weights, inner_products, y))

    def fit(self, X, y, sample_weight=None, dummy_first_step=False):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        sample_weight is not supported yet
        """
        # TODO: sample_weight support

        # Ideal data ordering depends on the solver
        # TODO: raise a warning if a copy is made ?
        if self.solver == "cgd":
            accept_sparse = "csc"
            order = "F"
            accept_large_sparse = False
        else:
            accept_sparse = "csr"
            order = "C"
            accept_large_sparse = False

        estimator_name = self.__class__.__name__
        is_classifier = estimator_name == "Classifier"
        X = check_array(
            X,
            order=order,
            accept_sparse=accept_sparse,
            dtype="numeric",
            accept_large_sparse=accept_large_sparse,
            estimator=estimator_name,
        )

        check_consistent_length(X, y)

        if is_classifier:
            y = check_array(y, ensure_2d=False, dtype=None, estimator=estimator_name)
            check_consistent_length(X, y)
            # Ensure that the label type is binary
            y_type = type_of_target(y)
            if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
                raise ValueError("Unknown label type: %r" % y_type)
            # TODO: random_state = check_random_state(random_state)
            # This replaces the target modalities by elements in {0, 1}
            le = LabelEncoder()
            if y_type == "multilabel-indicator":
                y_encoded = le.fit_transform(np.argmax(y, axis=1))
            else:
                y_encoded = le.fit_transform(y)
            # Keep track of the classes
            self.classes_ = le.classes_
            self.n_classes = len(self.classes_)
            if y_type == "binary":
                # We need to put the targets in {-1, 1}
                y_encoded = 2 * y_encoded - 1.0
                self.n_classes = 1
                self._check_binary_loss()
            else:
                self._check_multiclass_loss()

        else:
            y = check_array(
                y, ensure_2d=False, dtype="numeric", estimator=estimator_name
            )
            y_encoded = y
            self.n_classes = 1
            self._check_regression_loss()

        # TODO: sample weights stuff, later...
        # # If sample weights exist, convert them to array (support for lists)
        # # and check length
        # # Otherwise set them to 1 for all examples
        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        #
        # # If class_weights is a dict (provided by the user), the weights
        # # are assigned to the original labels. If it is "balanced", then
        # # the class_weights are assigned after masking the labels with a OvR.
        # if isinstance(class_weight, dict) or multi_class == "multinomial":
        #     class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        #     sample_weight *= class_weight_[le.fit_transform(y)]
        #
        # # For doing a ovr, we need to mask the labels first. for the
        # # multinomial case this is not necessary.
        # if multi_class == "ovr":
        #     w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        # mask_classes = np.array([-1, 1])
        #     # for compute_class_weight
        #
        #     if class_weight == "balanced":
        #         class_weight_ = compute_class_weight(
        #             class_weight, classes=mask_classes, y=y_bin
        #         )
        #         sample_weight *= class_weight_[le.fit_transform(y_bin)]
        #
        # else:
        #     if solver not in ["sag", "saga"]:
        #         lbin = LabelBinarizer()
        #         Y_multi = lbin.fit_transform(y)
        #         if Y_multi.shape[1] == 1:
        #             Y_multi = np.hstack([1 - Y_multi, Y_multi])
        #     else:
        #         # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
        #         le = LabelEncoder()
        #         Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
        #
        #     w0 = np.zeros(
        #         (classes.size, n_features + int(fit_intercept)),
        #         order="F",
        #         dtype=X.dtype,
        #     )

        #######
        solver = self._get_solver(X, y_encoded)
        w = self._get_initial_iterate(X, y_encoded)
        optimization_result = solver.solve(w, dummy_first_step=dummy_first_step)
        self.history_.record_nm("sc_prods").record = np.cumsum(self.history_.record_nm("sc_prods").record)

        self.optimization_result_ = optimization_result
        self.n_iter_ = np.asarray([optimization_result.n_iter], dtype=np.int32)

        w = optimization_result.w

        if self.fit_intercept:
            self.intercept_ = np.array([w[0]]).reshape(self.n_classes)
            self.coef_ = w[1:].T.copy()
        else:
            self.intercept_ = np.zeros(self.n_classes)
            self.coef_ = w[:].T.copy()

        return self

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        # TODO: this is from scikit-learn, cite and put authors
        check_is_fitted(self)

        # For now, no sparse arrays
        # X = check_array(X, accept_sparse="csr")
        X = check_array(X, accept_sparse=False, estimator=self.__class__.__name__)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        if self.n_classes == 1:
            return scores.ravel()
        else:
            return scores


class Classifier(BaseLearner):
    """
    Binary classifier. The default is standard penalized logistic regression,
    but it includes other learning algorithms, including very robust ones.
    It allows to combine several ``loss``, ``estimator`` and ``penalty`` in order to
    define a learning strategy.

    - ``estimator`` can be 'erm' (empirical risk minimization, which is the
      standard approach), 'mom' (median of means, which is a variant leading to
      classifiers robust to both outliers and heavy-tails), among several others (see
      below).

    - ``solver`` can be 'cgd' (coordinate gradient descent), 'gd' (gradient descent),
      'sgd' (stochastic gradient descent), 'svrg' (stochastic variance reduced
      gradient descent) and 'saga' (?)

    - ``penalty`` can be 'none' (no penalization), 'l2' (ridge penalization),
      'l1' (L1 penalization) and 'elasticnet'.

    Parameters
    ----------
    loss : {'logistic'}, default='logistic'
        The loss used in the goodness-of-fit criterion. Defaults to logistic
        regression loss.

    estimator : {'erm', 'mom', 'tmean', 'implicit', 'gmom', 'holland_catoni'}, \
              default='erm'
        The estimator used to compute gradients or partial derivatives. Some of these
        lead to a classifier that is very robust to outliers and/or heavy tails. A
        choice leading to a nice computations/performance balance is ``estimator='cgd'``
        together with  ``solver='cgd'``. Read more in the :ref:`User Guide
        <robust_estimation>`.

    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Used to specify the norm used in the penalization. If 'none', no regularization
         is applied.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Smaller values
        specify stronger regularization.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    step_size : float, default=1.0
        What the hell is this ? TODO

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, default=None
        Used when the solver or the estimator involves random shuffling.

    solver : {'cgd', 'gd', 'sgd', 'svrg', 'saga'}, default='cgd'
        Algorithm to use in the optimization problem.

        TODO: more blabla here

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    """

    def __init__(
        self,
        *,
        penalty="l2",
        C=1.0,
        step_size=1.0,
        loss="logistic",
        fit_intercept=True,
        estimator="erm",
        block_size=0.07,
        percentage=0.01,
        eps=0.001,
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.5,
        cgd_IS=False,
    ):
        super(Classifier, self).__init__(
            penalty=penalty,
            C=C,
            step_size=step_size,
            loss=loss,
            fit_intercept=fit_intercept,
            estimator=estimator,
            block_size=block_size,
            percentage=percentage,
            eps=eps,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
            cgd_IS=cgd_IS,
        )

        self.class_weight = class_weight
        self.classes_ = None

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        if self.loss not in ["logistic", "multilogistic"]:
            raise ValueError("Classification probabilities can only be computed for logistic classification but loss is : %s" % self.loss)

        prob = self.decision_function(X)
        if self.n_classes == 1:
            expit(prob, out=prob)
            return np.vstack([1 - prob, prob]).T
        else:
            return softmax(prob, axis=1)



    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        # TODO: deal with threshold for predictions
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class Regressor(BaseLearner, RegressorMixin):
    def __init__(
        self,
        *,
        penalty="l2",
        C=1.0,
        step_size=1.0,
        loss="leastsquares",
        fit_intercept=True,
        estimator="erm",
        block_size=0.07,
        percentage=0.05,
        eps=0.001,
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.5,
        cgd_IS=False,
    ):
        super(Regressor, self).__init__(
            penalty=penalty,
            C=C,
            step_size=step_size,
            loss=loss,
            fit_intercept=fit_intercept,
            estimator=estimator,
            block_size=block_size,
            percentage=percentage,
            eps=eps,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
            cgd_IS=cgd_IS,
        )

    def predict(self, X):

        # TODO: deal with threshold for predictions
        return self.decision_function(X)

    def mse(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        Returns
        -------
        score : float
            MSE of ``self.predict(X)`` wrt. `y`.
        """
        return 0.5 * ((y - self.predict(X)) ** 2).mean()
