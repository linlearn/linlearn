import numpy as np
import numbers
from scipy.special import expit, logsumexp

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import log_logistic, safe_sparse_dot, softmax, squared_norm

from .loss import losses_factory, steps_coordinate_descent
from .penalty import penalties_factory
from .solver import coordinate_gradient_descent, History, solvers_factory
from .strategy import (
    decision_function,
    decision_function_coef_intercept,
    strategies_factory,
)

# __losses = [
#     "hinge",
#     "smoothed hinge",
#     "logistic",
#     "quadratic hinge",
#     "modified huber",
# ]

# TODO: support for penalty = "none" and "elasticnet"
# class LogisticRegression(LinearClassifierMixin,
#                          SparseCoefMixin,
#                          BaseEstimator):

# TODO: serialization


class BinaryClassifier(ClassifierMixin, BaseEstimator):

    _losses = list(losses_factory.keys())
    _penalties = list(penalties_factory.keys())
    _strategies = list(strategies_factory.keys())
    _solvers = list(solvers_factory.keys())

    # __losses = [
    #     "hinge",
    #     "smoothed hinge",
    #     "logistic",
    #     "quadratic hinge",
    #     "modified huber",
    # ]
    #
    # __penalties = ["l2", "l1", "none", "elasticnet"]

    # w = coordinate_gradient_descent(
    #     # loss_value_batch,
    #     logistic_value_batch,
    #     # loss_derivative,
    #     logistic_derivative,
    #     # penalty_apply_single,
    #     # l2sq_apply_single,
    #     l1_apply_single,
    #     # penalty_value,
    #     # l2sq_value,
    #     l1_value,
    #     penalty_strength / n_samples,
    #     w,
    #     X,
    #     y,
    #     fit_intercept,
    #     steps,
    #     max_iter,
    #     history,
    # )

    def __init__(
        self,
        *,
        penalty="l2",
        C=1.0,
        loss="logistic",
        fit_intercept=True,
        strategy="erm",
        solver="cgd",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
        # The order is important here: this calls the properties defined below
        self.penalty = penalty
        self.C = C
        self.loss = loss
        self.strategy = strategy
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
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
        if val not in BinaryClassifier._penalties:
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
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        if val not in BinaryClassifier._losses:
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
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, val):
        if val not in BinaryClassifier._strategies:
            raise ValueError(
                "strategy must be one of %r; got (strategy=%r)"
                % (self._strategies, val)
            )
        else:
            self._strategy = val

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        if val not in BinaryClassifier._solvers:
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
        if not isinstance(val, numbers.Real) or val <= 0.0:
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol=%r)" % val
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

    # TODO: properties for class_weight=None, random_state=None, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None
    # if self.penalty == "elasticnet":
    #     if (
    #         not isinstance(self.l1_ratio, numbers.Number)
    #         or self.l1_ratio < 0
    #         or self.l1_ratio > 1
    #     ):
    #         raise ValueError(
    #             "l1_ratio must be between 0 and 1;"
    #             " got (l1_ratio=%r)" % self.l1_ratio
    #         )
    # elif self.l1_ratio is not None:
    #     warnings.warn(
    #         "l1_ratio parameter is only used when penalty is "
    #         "'elasticnet'. Got "
    #         "(penalty={})".format(self.penalty)
    #     )
    # if self.penalty == "none":
    #     if self.C != 1.0:  # default values
    #         warnings.warn(
    #             "Setting penalty='none' will ignore the C and l1_ratio "
    #             "parameters"
    #         )
    #         # Note that check for l1_ratio is done right above
    #     C_ = np.inf
    #     penalty = "l2"
    # else:
    #     C_ = self.C
    #     penalty = self.penalty
    # if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
    #     raise ValueError(
    #         "Maximum number of iteration must be positive;"
    #         " got (max_iter=%r)" % self.max_iter
    #     )

    def _get_solver(self, X, y):
        # TODO: X, y must have been checked at this point
        n_samples, n_features = X.shape

        # Get the loss object
        loss_factory = losses_factory[self.loss]
        loss = loss_factory()

        # Get the penalty object
        penalty_factory = penalties_factory[self.penalty]
        # The strength is scaled using following scikit-learn's scaling
        strength = 1 / (self.C * n_samples)
        penalty = penalty_factory(strength)

        # TODO: clean this
        self.block_size = 42
        # Get the strategy
        strategy_factory = strategies_factory[self.strategy]
        strategy = strategy_factory(
            loss, X, y, self.fit_intercept, block_size=self.block_size
        )

        if self.solver == "cgd":
            # Get the gradient descent steps for each coordinate
            steps = steps_coordinate_descent(loss.lip, X, self.fit_intercept)
            self.history_ = History("CGD", self.max_iter, self.verbose)
            # TODO: X should be F-major or CSC. Raise a warning if not. Do the same as scikit
            # TODO: verbose LogReg in scikit to see how it behaves
            def solve(w):
                return coordinate_gradient_descent(
                    loss,
                    penalty,
                    strategy,
                    w,
                    X,
                    y,
                    self.fit_intercept,
                    steps,
                    self.max_iter,
                    self.tol,
                    self.history_,
                )

            return solve

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

    def fit(self, X, y, sample_weight=None):
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
        sample_weight is not suported yet
        """
        # TODO: sample_weight support
        if self.solver == "cgd":
            accept_sparse = "csc"
            order = "F"
            accept_large_sparse = True
        else:
            accept_sparse = "csr"
            order = "C"
            accept_large_sparse = True

        # TODO: y'a une verif dans le fit et puis aussi dans le compute path
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=accept_sparse,
            order=order,
            accept_large_sparse=accept_large_sparse,
            estimator="BinaryClassifier",
        )

        # TODO: random_state = check_random_state(random_state)

        # TODO: non c'est pas bon du tout, faut que tout soit dans -1, 1
        check_classification_targets(y)

        le = LabelEncoder()
        # This replaces the modalities by elements in {0, 1}
        y_encoded = le.fit_transform(y)
        if le.classes_.size != 2:
            # TODO: try out with sklearn to get correct error message
            raise ValueError("We need exactly two classes to fit BinaryClassifier")

        self.classes_ = le.classes_

        mask = y_encoded == 1
        # But we want to ensure the same dtype as X and values in {-1, 1}
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.0

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
        solver = self._get_solver(X, y_bin)
        w = self._get_initial_iterate(X, y_bin)
        optimization_result = solver(w)

        self.optimization_result_ = optimization_result
        self.n_iter_ = np.asarray([optimization_result.n_iter], dtype=np.int32)

        # print(optimization_result)

        w = optimization_result.w

        # TODO: Not correct wih respect to what scikit returns
        if self.fit_intercept:
            self.intercept_ = np.array([w[0]])
            self.coef_ = w[1:]
        else:
            self.intercept_ = np.zeros(1)
            self.coef_ = w

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
        check_is_fitted(self)

        # Which sparse array do we accept ?
        X = check_array(X, accept_sparse="csr")

        n_features = self.coef_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores
        # return scores.ravel() if scores.shape[1] == 1 else scores

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

        # ovr = self.multi_class in ["ovr", "warn"] or (
        #     self.multi_class == "auto"
        #     and (self.classes_.size <= 2 or self.solver == "liblinear")
        # )
        # if ovr:
        #     return super()._predict_proba_lr(X)
        # else:
        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)

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
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

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

    # def decision_function(self, X):
    #     # TODO: test that classifier is fitted
    #     # TODO: test X
    #     out = np.empty(X.shape[0], dtype=X.dtype)
    #     decision_function_coef_intercept(
    #         X, self.fit_intercept, self.coef, self.intercept, out
    #     )
    #     return out

    # if not self._fitted:
    #     raise ValueError("You must call ``fit`` before")
    # else:
    #     X = self._safe_array(X, dtype=X.dtype)
    #     z = X.dot(self.weights)
    #     if self.intercept:
    #         z += self.intercept
    #     return z
    #
    # # TODO: add threshold in predict
    # def predict(self, X):
    #     """Predict class for given samples
    #
    #     Parameters
    #     ----------
    #     X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
    #         Samples.
    #
    #     Returns
    #     -------
    #     output : `np.array`, shape=(n_samples,)
    #         Returns predicted values.
    #     """
    #     logits = self.decision_function(X)
    #     indices = (logits > 0).astype(np.int)
    #     return indices
    #     # return self.classes[indices]

    # def predict_proba(self, X):
    #     """
    #     Probability estimates.
    #
    #     The returned estimates for all classes are ordered by the
    #     label of classes.
    #
    #     Parameters
    #     ----------
    #     X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
    #         Input features matrix
    #
    #     Returns
    #     -------
    #     output : `np.ndarray`, shape=(n_samples, 2)
    #         Returns the probability of the sample for each class
    #         in the model in the same order as in `self.classes`
    #     """
    #     # if not self._fitted:
    #     #     raise ValueError("You must call ``fit`` before")
    #     # else:
    #
    #     logits = self.decision_function(X)
    #     n_samples = logits.shape[0]
    #
    #     # probs_class_1 = sigmoid(logits)
    #     # probs = np.empty((n_samples, 2))
    #     # probs[:, 1] = probs_class_1
    #     # probs[:, 0] = 1.0 - probs_class_1
    #     # return probs

    # def __repr__(self):
    #     r = self.__class__.__name__
    #     r += "("
    #     r += "C={C}".format(C=self.C)
    #     r += ', penalty="{penalty}"'.format(penalty=self.penalty)
    #     r += ', loss="{loss}"'.format(loss=self.loss)
    #     r += ', solver="{solver}"'.format(solver=self.solver)
    #     r += ', strategy="{strategy}"'.format(strategy=self.strategy)
    #     r += ")"
    #     return r
