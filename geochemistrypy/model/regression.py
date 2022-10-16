# -*- coding: utf-8 -*-
# import sys
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from utils.base import save_fig
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from ._base import WorkflowBase
from .func.algo_regression._polynomial import show_formula
# sys.path.append("..")


class RegressionWorkflowBase(WorkflowBase):
    """The base workflow class of regression algorithms."""

    common_function = ['Model Score', 'Cross Validation']

    def __init__(self) -> None:
        super().__init__()

    def data_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def fit(self, X, y=None):
        self.model.fit(X, y)

    def predict(self, X):
        y_predict = self.model.predict(X)
        return y_predict

    def plot_predict(self,y_test,y_test_predict):
        y_test = np.array(y_test).reshape(1, len(y_test)).flatten()
        y_test_predict = np.array(y_test_predict).reshape(1, len(y_test_predict)).flatten()
              # the lien between a and b.
        line_a = y_test.min()
        line_b = y_test.max()
        #plot figure
        print("-----* Plot prediction *-----")
        plt.figure(figsize=(4, 4))
        plt.plot([line_a, line_b], [line_a, line_b], '-r', linewidth=1)
        plt.plot(y_test, y_test_predict, 'o', color='gold', alpha=0.3)
        plt.title('Predicted image')
        plt.xlabel('y_test')
        plt.ylabel('y_test_predict')
        save_fig('Plot Prediction', MODEL_OUTPUT_IMAGE_PATH)

    @staticmethod
    def score(y_test, y_test_predict):
        mse = mean_squared_error(y_test, y_test_predict)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_predict)
        r2 = r2_score(y_test, y_test_predict)
        evs = explained_variance_score(y_test, y_test_predict)
        print("-----* Model Score *-----")
        print("RMSE score:", rmse)
        print("MAE score:", mae)
        print("R2 score:", r2)
        print("Explained Variance Score:", evs)

    @staticmethod
    def _display_cross_validation_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    def cross_validation(self, X_train, y_train, cv_num=10):
        print("-----* Cross Validation *-----")
        # self.model comes from the subclass of every regression algorithm
        scores = cross_validate(self.model, X_train, y_train,
                                scoring=('neg_root_mean_squared_error',
                                         'neg_mean_absolute_error',
                                         'r2',
                                         'explained_variance'),
                                cv=cv_num)
        for key, values in scores.items():
            print("*", key.upper(), "*")
            self._display_cross_validation_scores(values)
            print('-------------')
        return scores

    @staticmethod
    def np2pd(array, columns_name):
        """The type of the data set is transformed from numpy.ndarray to pandas.DataFrame."""
        return pd.DataFrame(array, columns=columns_name)

    # TODO: How to prevent overfitting
    def is_overfitting():
        pass

    # TODO: Do Hyperparameter Searching
    def search_best_hyper_parameter():
        pass


class PolynomialRegression(RegressionWorkflowBase):

    name = "Polynomial Regression"
    special_function = ["Polynomial Regression Formula"]

    def __init__(self,
                 degree: int = 2,
                 interaction_only: bool = False,
                 is_include_bias: bool = False,
                 order: str = 'C',
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 copy_X: bool = True,
                 n_jobs: Optional[int] = None) -> None:

        super().__init__()
        self.degree = degree
        self.is_include_bias = is_include_bias
        self.interaction_only = interaction_only
        self.order = order
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.model = LinearRegression(fit_intercept=self.fit_intercept,
                                      copy_X=self.copy_X,
                                      n_jobs=self.n_jobs)

        # special attributes
        self._features_name = None

    def poly(self, X_train, X_test):
        poly_features = PolynomialFeatures(degree=self.degree,
                                           include_bias=self.is_include_bias,
                                           interaction_only=self.interaction_only,
                                           order=self.order)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.fit_transform(X_test)
        try:
            # scikit-learn >= 1.0
            self._features_name = poly_features.get_feature_names_out()
        except AttributeError:
            self._features_name = poly_features.get_feature_names()
        return X_train_poly, X_test_poly

    @staticmethod
    def _show_formula(coef, intercept, features_name):
        print("-----* Polynomial Regression Formula *-----")
        show_formula(coef, intercept, features_name)

    def special_components(self, **kwargs):
        self._show_formula(coef=self.model.coef_, intercept=self.model.intercept_, features_name=self._features_name)


class XgboostRegression(RegressionWorkflowBase, BaseEstimator):
    # https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

    _SklObjective = Optional[
        Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ]
    ]

    name = "Xgboost"
    special_function = ['Feature Importance']

    def __init__(
        self,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        n_estimators: int = 100,
        verbosity: Optional[int] = None,
        objective: _SklObjective = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        super().__init__(random_state=42)
        self.n_estimators = n_estimators
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        if kwargs:
            self.kwargs = kwargs

        self.model = xgboost.XGBRegressor(
            n_estimators=self.n_estimators,
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters,
            predictor=self.predictor,
            enable_categorical=self.enable_categorical,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds)

    def _feature_importance(self):
        print("-----* Feature Importance *-----")
        columns_name = RegressionWorkflowBase.X.columns
        # print the feature importance value orderly
        for feature_name, score in zip(list(columns_name), self.model.feature_importances_):
            print(feature_name, ":", score)

        # histograms present feature weights for XGBoost predictions
        plt.figure(figsize=(40, 6))
        plt.bar(range(len(columns_name)), self.model.feature_importances_, tick_label=columns_name)
        save_fig("xgb_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

        # feature importance map ranked by importance
        plt.rcParams["figure.figsize"] = (14, 8)
        xgboost.plot_importance(self.model)
        save_fig("xgb_feature_importance_score", MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self._feature_importance()


class SVM(RegressionWorkflowBase, BaseEstimator):
    pass


class DecisionTreeRegression(RegressionWorkflowBase, BaseEstimator):
    name = "Decision Tree"
    special_function = ["Decision Tree Plot"]

    def __init__(self,
                 criteria='gini',
                 splitter='best',
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0
    )->None:
        """
        Parameters
        ----------
        criterion : {"squared_error", "friedman_mse", "absolute_error", \
                "poisson"}, default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, "friedman_mse", which uses
            mean squared error with Friedman's improvement score for potential
            splits, "absolute_error" for the mean absolute error, which minimizes
            the L1 loss using the median of each terminal node, and "poisson" which
            uses reduction in Poisson deviance to find splits.
            .. versionadded:: 0.18
               Mean Absolute Error (MAE) criterion.
            .. versionadded:: 0.24
                Poisson deviance criterion.
            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.
            .. deprecated:: 1.0
                Criterion "mae" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="absolute_error"` which is equivalent.

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.
            .. versionchanged:: 0.18
               Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.
            .. versionchanged:: 0.18
               Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at each
              split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator. The features are always
            randomly permuted at each split, even if ``splitter`` is set to
            ``"best"``. When ``max_features < n_features``, the algorithm will
            select ``max_features`` at random at each split before finding the best
            split among them. But the best found split may vary across different
            runs, even if ``max_features=n_features``. That is the case, if the
            improvement of the criterion is identical for several splits and one
            split has to be selected at random. To obtain a deterministic behaviour
            during fitting, ``random_state`` has to be fixed to an integer.
            See :term:`Glossary <random_state>` for details.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.
            .. versionadded:: 0.19

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.
            .. versionadded:: 0.22

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Decision_tree_learning
        [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
               and Regression Trees", Wadsworth, Belmont, CA, 1984.
        [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
               Learning", Springer, 2009.
        [4] L. Breiman, and A. Cutler, "Random Forests",
               https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
        """
        super().__init__(random_state=42)
        self.criteria = criteria,
        self.splitter = splitter,
        self.max_depth = max_depth,
        self.min_samples_split = min_samples_split,
        self.min_samples_leaf = min_samples_leaf,
        self.min_weight_fraction_leaf = min_weight_fraction_leaf,
        self.max_features = max_features,
        self.random_state = random_state,
        self.max_leaf_nodes = max_leaf_nodes,
        self.min_impurity_decrease = min_impurity_decrease,
        self.ccp_alpha = ccp_alpha

        self.model = DecisionTreeRegressor(criteria=self.criteria,
                                           splitter=self.splitter,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                           max_features=self.max_features,
                                           random_state=self.random_state,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           min_impurity_decrease=self.min_impurity_decrease,
                                           ccp_alpha=self.ccp_alpha)

    def plot_tree_function(self):
        ###################################################
        # Drawing decision tree diagrams
        ###################################################
        print("-----* Decision Tree Plot *-----")
        y = RegressionWorkflowBase().y
        X = RegressionWorkflowBase().X
        clf = self.model.fit(X, y)
        plt.figure()
        plot_tree(clf, filled=True)
        save_fig('plot_decision_tree_regression', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.plot_tree_function()


class ExtraTreeRegression(RegressionWorkflowBase, BaseEstimator):
    name = "Extra-Trees"
    special_function = ["Feature Importance"]

    def __init__(self,
                 n_estimator: int = 500,
                 bootstrap: bool = False,
                 oob_score: bool = False,
                 max_leaf_nodes: int = 20,
                 random_state: int = 42,
                 n_jobs: int = -1):
        super().__init__(random_state=42)
        self.n_estimators = n_estimator
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                         bootstrap=self.bootstrap,
                                         oob_score=self.oob_score,
                                         max_leaf_nodes=self.max_leaf_nodes,
                                         random_state=self.random_state,
                                         n_jobs=self.n_jobs)

    def feature_importances(self):
        importances_values = self.model.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.X_train.columns, columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)

        importance = importance.sort_values(["importance"], ascending=True)
        importance["importance"] = (importance["importance"]).astype(float)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
        save_fig("ExtraTreeRegression_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

    def extratree(self):
        pass

    def special_components(self):
        self.feature_importances()
        pass


class RandomForestRegression(RegressionWorkflowBase, BaseEstimator):
    name = "Random Forest"
    special_function = ["Feature Importance"]

    def __init__(self,
                 n_estimators: int = 500,
                 oob_score: bool = True,
                 max_leaf_nodes: int = 15,
                 n_jobs: int = -1,
                 random_state: int = 42):
        super().__init__(random_state=42)
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                           oob_score=self.oob_score,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state)

    def feature_importances(self):
        print("-----* Feature Importance *-----")
        importances_values = self.model.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.X_train.columns, columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)

        importance = importance.sort_values(["importance"], ascending=True)
        importance["importance"] = (importance["importance"]).astype(float)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
        save_fig("RandomForestRegression_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

    def plot(self):
        pass

    def special_components(self):
        self.feature_importances()
        self.plot()
        pass

class SupportVectorRegression(RegressionWorkflowBase, BaseEstimator):
    name = "Support Vector Machine"
    special_function = []

    def __init__(
        self,
        kernel='rbf',
        degree: int = 3,
        gamma='scale',
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
            default='rbf'
             Specifies the kernel type to be used in the algorithm.
             If none is given, 'rbf' will be used. If a callable is given it is
             used to precompute the kernel matrix.

        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.

        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - if ``gamma='scale'`` (default) is passed then it uses
              1 / (n_features * X.var()) as value of gamma,
            - if 'auto', uses 1 / n_features.
            .. versionchanged:: 0.22
               The default value of ``gamma`` changed from 'auto' to 'scale'.

        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.

        tol : float, default=1e-3
            Tolerance for stopping criterion.

        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive.
            The penalty is a squared l2 penalty.

        epsilon : float, default=0.1
             Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
             within which no penalty is associated in the training loss function
             with points predicted within a distance epsilon from the actual
             value.
        shrinking : bool, default=True
            Whether to use the shrinking heuristic.
            See the :ref:`User Guide <shrinking_svm>`.

        cache_size : float, default=200
            Specify the size of the kernel cache (in MB).

        verbose : bool, default=False
            Enable verbose output. Note that this setting takes advantage of a
            per-process runtime setting in libsvm that, if enabled, may not work
            properly in a multithreaded context.

        max_iter : int, default=-1
            Hard limit on iterations within solver, or -1 for no limit.

        References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_
    .. [2] `Platt, John (1999). "Probabilistic outputs for support vector
        machines and comparison to regularizedlikelihood methods."
        <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
        """
        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

        self.model = SVR(kernel=self.kernel,
                        degree=self.degree,
                        gamma=self.gamma,
                        coef0=self.coef0,
                        tol=self.tol,
                        C=self.C,
                        epsilon=self.epsilon,
                        shrinking=self.shrinking,
                        cache_size=self.cache_size,
                        verbose=self.verbose,
                        max_iter=self.max_iter)

        self.naming = SupportVectorRegression.name

    # special attributes
    def special_components(self):
        pass