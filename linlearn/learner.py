"""
Binary Classifier
"""

# Author: Stephane Gaiffas <stephane.gaiffas@gmail.com>

# Parts of the code below are directly from scikit-learn, in particular from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_logistic.py

import numbers
import numpy as np
from scipy.special import expit

from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

from ._loss import steps_coordinate_descent, Logistic, LeastSquares
from ._penalty import NoPen, L2Sq, L1, ElasticNet
from ._solver import CGD, GD, SGD, SVRG, SAGA, History
from ._estimator import ERM, MOM, TMean, Catoni, Implicit, GMOM, Holland


# TODO: serialization

class BaseLearner(ClassifierMixin, BaseEstimator):

    _losses = ["logistic", "leastsquares"]
    _penalties = ["none", "l2", "l1", "elasticnet"]
    _estimators = ["erm", "mom", "tmean", "catoni", "implicit", "gmom", "holland"]
    _solvers = ["cgd", "gd", "sgd", "svrg", "saga"]

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
        percentage=0.05,
        eps=0.001,
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.5
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

        self.history_ = None
        self.intercept_ = None
        self.coef_ = None
        self.optimization_result_ = None
        self.n_iter_ = None
        self.classes_ = None

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
            raise ValueError("step_size must be a positive number; got (step_size=%r)" % val)
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
        if not isinstance(val, numbers.Real) or val <= 0.0 or val > 1:
            raise ValueError("percentage must be in (0, 1]; got (percentage=%r)" % val)
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

    # TODO: properties for class_weight=None, random_state=None, verbose=0, warm_start=False, n_jobs=None

    def _get_loss(self):
        if self.loss == "logistic":
            return Logistic()
        elif self.loss == "leastsquares":
                return LeastSquares()
        else:
            raise ValueError("Loss unknown")

    def _get_estimator(self, X, y, loss):
        if self.estimator == "erm":
            return ERM(X, y, loss, self.fit_intercept)
        elif self.estimator == "mom":
            n_samples = y.shape[0]
            n_samples_in_block = int(self.block_size * n_samples)
            return MOM(X, y, loss, self.fit_intercept, n_samples_in_block)
        elif self.estimator == "tmean":
            return TMean(X, y, loss, self.fit_intercept, self.percentage)
        elif self.estimator == "catoni":
            return Catoni(X, y, loss, self.fit_intercept, self.eps)
        elif self.estimator == "holland":
            return Holland(X, y, loss, self.fit_intercept, self.eps)
        elif self.estimator == "implicit":
            return Implicit(X, y, loss, self.fit_intercept, int(1/self.block_size))
        elif self.estimator == "gmom":
            n_samples = y.shape[0]
            n_samples_in_block = int(self.block_size * n_samples)
            return GMOM(X, y, loss, self.fit_intercept, n_samples_in_block)
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

    def _get_solver(self, X, y, trackers=None):
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
        n_samples_in_block = int(n_samples * self.block_size)

        if self.solver == "cgd":
            # Get the gradient descent steps for each coordinate
            steps = self.step_size * steps_coordinate_descent(loss.lip, X, n_samples_in_block, self.fit_intercept)
            # Create an history object for the solver
            history = History("CGD", self.max_iter, self.verbose, trackers=trackers)
            self.history_ = history

            return CGD(
                X,
                y,
                loss,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                self.random_state,
                steps,
                history
            )

        elif self.solver == "gd":
            # Get the gradient descent steps for each coordinate
            step = self.step_size * np.min(steps_coordinate_descent(loss.lip, X, n_samples_in_block, self.fit_intercept))
            # Create an history object for the solver
            history = History("GD", self.max_iter, self.verbose, trackers=trackers)
            self.history_ = history

            return GD(
                X,
                y,
                loss,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                self.random_state,
                step,
                history
            )
        elif self.solver == "sgd":
            # Get the gradient descent steps for each coordinate
            step = self.step_size * np.min(steps_coordinate_descent(loss.lip, X, n_samples_in_block, self.fit_intercept))
            # Create an history object for the solver
            history = History("SGD", self.max_iter, self.verbose, trackers=trackers)
            self.history_ = history

            return SGD(
                X,
                y,
                loss,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                self.random_state,
                step,
                history
            )

        elif self.solver == "svrg":
            # Get the gradient descent steps for each coordinate
            step = self.step_size * np.min(steps_coordinate_descent(loss.lip, X, n_samples_in_block, self.fit_intercept))
            # Create an history object for the solver
            history = History("SVRG", self.max_iter, self.verbose, trackers=trackers)
            self.history_ = history

            return SVRG(
                X,
                y,
                loss,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                self.random_state,
                step,
                history
            )
        elif self.solver == "saga":
            # Get the gradient descent steps for each coordinate
            step = self.step_size * np.min(steps_coordinate_descent(loss.lip, X, n_samples_in_block, self.fit_intercept))
            # Create an history object for the solver
            history = History("SAGA", self.max_iter, self.verbose, trackers=trackers)
            self.history_ = history

            return SAGA(
                X,
                y,
                loss,
                self.fit_intercept,
                estimator,
                penalty,
                self.max_iter,
                self.tol,
                self.random_state,
                step,
                history
            )

        else:
            raise NotImplementedError("%s is not implemented yet" % self.solver)

    def _get_initial_iterate(self, X, y):
        # Deal with warm-starting here
        n_samples, n_features = X.shape
        if self.fit_intercept:
            w = np.zeros(n_features + 1)
        else:
            w = np.zeros(n_features)
        return w


    def fit(self, X, y, sample_weight=None, trackers=None, dummy_first_step=True):
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
        is_classifier = estimator_name == "BinaryClassifier"
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
            # Ensure that the label type is binary
            y_type = type_of_target(y)
            # if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
            #     raise ValueError("Unknown label type: %r" % y_type)
            if y_type != "binary":
                raise ValueError("Unknown label type: %r" % y_type)
            # TODO: random_state = check_random_state(random_state)
            # This replaces the target modalities by elements in {0, 1}
            le = LabelEncoder()
            # if y_type =="multilabel-indicator":
            #     y_encoded = le.fit_transform(np.argmax(y, axis=1))
            # else:
            #     y_encoded = le.fit_transform(y)
            y_encoded = le.fit_transform(y)
            # Keep track of the classes
            self.classes_ = le.classes_
            # We need to put the targets in {-1, 1}
            y_encoded[y_encoded == 0] = -1.0

        else:
            y = check_array(y, ensure_2d=False, dtype="numeric", estimator=estimator_name)
            y_encoded = y


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
        solver = self._get_solver(X, y_encoded, trackers=trackers)
        w = self._get_initial_iterate(X, y_encoded)
        optimization_result = solver.solve(w, dummy_first_step=dummy_first_step)

        self.optimization_result_ = optimization_result
        self.n_iter_ = np.asarray([optimization_result.n_iter], dtype=np.int32)

        w = optimization_result.w

        if self.fit_intercept:
            self.intercept_ = np.array([w[0]])
            self.coef_ = w[np.newaxis, 1:].copy()
        else:
            self.intercept_ = np.zeros(1)
            self.coef_ = w[np.newaxis, :].copy()

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
        X = check_array(X, accept_sparse=False, estimator="BinaryClassifier")

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores.ravel()



class BinaryClassifier(BaseLearner):

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
        percentage=0.05,
        eps=0.001,
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.5
    ):
        super(BinaryClassifier, self).__init__(
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
            l1_ratio=l1_ratio
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
        prob = self.decision_function(X)
        expit(prob, out=prob)
        return np.vstack([1 - prob, prob]).T

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
        l1_ratio=0.5
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
            l1_ratio=l1_ratio
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
        return ((y - self.predict(X))**2).mean()
