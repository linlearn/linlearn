# License: BSD 3 clause

import numpy as np
from numba import njit

from ..model import Features, Logistic, sigmoid
from ..solver import SVRG


# TODO: a base class for a learner
# TODO: barzilai-borwein
# TODO:


@njit
def predict_proba_smp(X_extended, y_extended, X_predict, w_start,
                      fit_intercept, max_iter, step):
    # X_extended is one more than n_samples
    n_samples_train, n_features = X_extended.shape
    n_samples_train -= 1

    n_samples_predict = X_predict.shape[0]

    logistic = Logistic()
    w = np.empty(w_start.shape)

    # If using warm-starting we should use less iterations
    # max_iter_extended = max_iter

    max_iter_extended = 200

    probs = np.empty((n_samples_predict, 2))

    for i in range(n_samples_predict):
        # First, a logistic regression augmented by a sample (x, 1)
        X_extended[-1] = X_predict[i]

        y_extended[-1] = 1.
        # TODO: We can created features_extended outside and just update X_extended
        features_extended = Features(fit_intercept).set(X_extended)
        w[:] = w_start
        w = svrg(features_extended, y_extended, logistic, w, max_iter_extended, step)
        if fit_intercept:
            z = X_predict[i].dot(w[1:]) + w[0]
        else:
            z = X_predict[i].dot(w)
        sigmoid_1 = sigmoid(z)

        y_extended[-1] = -1.
        # TODO: We can created features_extended outside and just update X_extended
        features_extended = Features(fit_intercept).set(X_extended)
        w[:] = w_start
        w = svrg(features_extended, y_extended, logistic, w, max_iter_extended, step)
        if fit_intercept:
            z = X_predict[i].dot(w[1:]) + w[0]
        else:
            z = X_predict[i].dot(w)
        sigmoid_0 = 1 - sigmoid(z)

        ss = sigmoid_0 + sigmoid_1
        probs[i, 0] = sigmoid_0 / ss
        probs[i, 1] = sigmoid_1 / ss

    return probs


class LogisticRegression(object):
    """
    Logistic regression learner, with many choices of penalization and
    solvers.

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'tv', 'none', 'binarsity'}, default='l2'
        The penalization to use. Default is ridge penalization

    solver : {'gd', 'agd', 'bfgs', 'svrg', 'sdca', 'sgd'}, default='svrg'
        The name of the solver to use

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used in gd, agd, sgd and svrg
        solvers

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Other Parameters
    ----------------
    sdca_ridge_strength : `float`, default=1e-3
        It controls the strength of the additional ridge penalization. Used in
        'sdca' solver

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : `int` seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers

    blocks_start : `numpy.array`, shape=(n_features,), default=None
        The indices of the first column of each binarized feature blocks. It
        corresponds to the ``feature_indices`` property of the
        ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    blocks_length : `numpy.array`, shape=(n_features,), default=None
        The length of each binarized feature blocks. It corresponds to the
        ``n_values`` property of the ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    Attributes
    ----------
    weights : `numpy.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    intercept : `float` or None
        The intercept, if ``fit_intercept=True``, otherwise `None`

    classes : `numpy.array`, shape=(n_classes,)
        The class labels of our problem
    """

    def __init__(self, penalty=None, tol=1e-4, C=1.0,
                 fit_intercept=True, class_weight=None, smp=False,
                 random_state=None, solver='svrg', max_iter=100, step=None,
                 verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):

        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

        self.intercept_ = None
        self.coef_ = None

        self.step = step

        self.smp = smp

        # LearnerGLM.__init__(
        #     self, fit_intercept=fit_intercept, penalty=penalty, C=C,
        #     solver=solver, step=step, tol=tol, max_iter=max_iter,
        #     verbose=verbose, warm_start=warm_start, print_every=print_every,
        #     record_every=record_every, sdca_ridge_strength=sdca_ridge_strength,
        #     elastic_net_ratio=elastic_net_ratio, random_state=random_state,
        #     blocks_start=blocks_start, blocks_length=blocks_length)
        # self.classes = None

    # def _construct_model_obj(self, fit_intercept=True):
    #     return ModelLogReg(fit_intercept)

    # def _encode_labels_vector(self, labels):
    #     """Encodes labels values to canonical labels -1 and 1
    #
    #     Parameters
    #     ----------
    #     labels : `np.array`, shape=(n_samples,)
    #         Labels vector
    #
    #     Returns
    #     -------
    #     output : `np.array`, shape=(n_samples,)
    #         Encoded labels vector which takes values -1 and 1
    #     """
    #     # If it is already in the canonical shape return it
    #     # Additional check as if self.classes.dtype is not a number it raises
    #     # a warning
    #     if np.issubdtype(self.classes.dtype, np.number) and \
    #             np.array_equal(self.classes, [-1, 1]):
    #         return labels
    #
    #     mapper = {self.classes[0]: -1., self.classes[1]: 1.}
    #     y = np.vectorize(mapper.get)(labels)
    #     return y

    def fit(self, X, y, sample_weight=None):
        # self.classes = np.unique(y)
        # if len(self.classes) != 2:
        #     raise ValueError('You wan only fit binary problems with '
        #                      'LogisticRegression')
        #
        # # For [0, 1] and [-1, 1] specific cases we force this mapping
        # if set(self.classes) == {-1, 1}:
        #     self.classes[0] = -1.
        #     self.classes[1] = 1.
        # elif set(self.classes) == {0, 1}:
        #     self.classes[0] = 0.
        #     self.classes[1] = 1.

        # If classes are not in the canonical shape we must transform them
        # y = self._encode_labels_vector(y)

        features = Features(self.fit_intercept).set(X)
        logistic = Logistic()

        if self.fit_intercept:
            w = np.zeros(X.shape[1] + 1)
        else:
            w = np.zeros(X.shape[1])

        svrg = SVRG(step=self.step).set(features=features, labels=y, loss=logistic, prox=None)

        w = svrg.solve(w)

        # w = svrg(features, y, logistic, w, self.max_iter, self.step, self.verbose)

        self._set_coef(w)
        # if self.fit_intercept:
        #     self.intercept_ = w[0]
        #     self.coef_ = w[1:]
        # else:
        #     self.coef_ = w

        if self.smp:
            n_samples, n_features = X.shape
            X_extended = np.empty((n_samples + 1, n_features))
            y_extended = np.empty(n_samples + 1)
            X_extended[:-1] = X
            y_extended[:-1] = y
            self.X_extended = X_extended
            self.y_extended = y_extended

    def _get_w(self):
        if self.fit_intercept:
            w = np.empty(self.coef_.shape[0] + 1)
            w[0] = self.intercept_
            w[1:] = self.coef_
        else:
            w = np.empty(self.coef_.shape[0])
            w[:] = self.coef_
        return w

    def _set_coef(self, w):
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.coef_ = w

    def decision_function(self, X):
        """
        Predict scores for given samples

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Samples.

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Confidence scores.
        """
        features = Features(self.fit_intercept).set(X)
        out = np.empty(X.shape[0])
        features.inner_prods(self._get_w(), out)
        return out
        # if not self._fitted:
        #     raise ValueError("You must call ``fit`` before")
        # else:
        #     X = self._safe_array(X, dtype=X.dtype)
        #     z = X.dot(self.weights)
        #     if self.intercept:
        #         z += self.intercept
        #     return z

    def predict(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Samples.

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Returns predicted values.
        """
        logits = self.decision_function(X)
        indices = (logits > 0).astype(np.int)
        return indices
        # return self.classes[indices]

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Input features matrix

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, 2)
            Returns the probability of the sample for each class
            in the model in the same order as in `self.classes`
        """
        # if not self._fitted:
        #     raise ValueError("You must call ``fit`` before")
        # else:

        if self.smp:
            w_start = self._get_w().copy()
            return predict_proba_smp(self.X_extended, self.y_extended, X, w_start,
                                     self.fit_intercept, self.max_iter, self.step)

        else:
            logits = self.decision_function(X)
            n_samples = logits.shape[0]
            probs_class_1 = sigmoid(logits)
            probs = np.empty((n_samples, 2))
            probs[:, 1] = probs_class_1
            probs[:, 0] = 1. - probs_class_1
            return probs
