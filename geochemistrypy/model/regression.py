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
from sklearn.neural_network import MLPRegressor
from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from ._base import WorkflowBase
from .func.algo_regression._polynomial import show_formula
from .func.algo_regression._dnn import plot_pred
from .func.algo_regression._linear import show_formula, plot_2d_graph, plot_3d_graph
# sys.path.append("..")


class RegressionWorkflowBase(WorkflowBase):
    """The base workflow class of regression algorithms."""

    common_function = ['Model Score', 'Cross Validation']

    def __init__(self) -> None:
        super().__init__()

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
    def score(y_true, y_predict):
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_predict)
        r2 = r2_score(y_true, y_predict)
        evs = explained_variance_score(y_true, y_predict)
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

    # TODO(Sany sanyhew1097618435@163.com): How to prevent overfitting
    def is_overfitting():
        pass

    # TODO(Sany sanyhew1097618435@163.com): Do Hyperparameter Searching
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


class XgboostRegression(RegressionWorkflowBase):

    name = "Xgboost"
    special_function = ['Feature Importance']

    # In fact, it's used for type hint in the original xgboost package.
    # Hence, we have to copy it here again. Just ignore it
    _SklObjective = Optional[
        Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ]
    ]

    # TODO: find out the attributes importance_type effect
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
        importance_type: Optional[str] = 'gain',
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        """
        References
        ----------
        Xgboost API for the scikit-learn wrapper:
        https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
        """

        super().__init__()
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


class DecisionTreeRegression(RegressionWorkflowBase):
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
    ) -> None:
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

        super().__init__()
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


class ExtraTreeRegression(RegressionWorkflowBase):
    """An extra-trees regressor"""
    
    name = "Extra-Trees"
    special_function = ["Feature Importance"]

    def __init__(self,
                 n_estimator: int = 500,
                 bootstrap: bool = False,
                 oob_score: bool = False,
                 max_leaf_nodes: int = 20,
                 random_state: int = 42,
                 n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.

            .. versionchanged:: 0.22
               The default value of ``n_estimators`` changed from 10 to 100
               in 0.22.

        criterion : {"squared_error", "absolute_error"}, default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion, and "absolute_error"
            for the mean absolute error.

            .. versionadded:: 0.18
               Mean Absolute Error (MAE) criterion.

            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.

            .. deprecated:: 1.0
                Criterion "mae" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="absolute_error"` which is equivalent.

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

        max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `round(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
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

        bootstrap : bool, default=False
            Whether bootstrap samples are used when building trees. If False, the
            whole dataset is used to build each tree.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score.
            Only available if bootstrap=True.

        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls 3 sources of randomness:

            - the bootstrapping of the samples used when building trees
              (if ``bootstrap=True``)
            - the sampling of the features to consider when looking for the best
              split at each node (if ``max_features < n_features``)
            - the draw of the splits for each of the `max_features`

            See :term:`Glossary <random_state>` for details.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`the Glossary <warm_start>`.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.

            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0.0, 1.0]`.

            .. versionadded:: 0.22

        References
        ----------
        scikit API: sklearn.ensemble.ExtraTreesRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extratreesregressor#sklearn.ensemble.ExtraTreesRegressor
        """
        
        super().__init__()
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
    """A random forest regressor"""

    name = "Random Forest"
    special_function = ["Feature Importance"]

    def __init__(self,
                 n_estimators: int = 500,
                 oob_score: bool = True,
                 max_leaf_nodes: int = 15,
                 n_jobs: int = -1,
                 random_state: int = 42
    ) ->None:
        """
        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.

            .. versionchanged:: 0.22
               The default value of ``n_estimators`` changed from 10 to 100
               in 0.22.

        criterion : {"squared_error", "absolute_error", "poisson"}, \
                default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion, "absolute_error"
            for the mean absolute error, and "poisson" which uses reduction in
            Poisson deviance to find splits.
            Training using "absolute_error" is significantly slower
            than when using "squared_error".

            .. versionadded:: 0.18
               Mean Absolute Error (MAE) criterion.

            .. versionadded:: 1.0
               Poisson criterion.

            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.

            .. deprecated:: 1.0
                Criterion "mae" was deprecated in v1.0 and will be removed in
                version 1.2. Use `criterion="absolute_error"` which is equivalent.

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

        max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `round(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
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

        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees. If False, the
            whole dataset is used to build each tree.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score.
            Only available if bootstrap=True.

        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls both the randomness of the bootstrapping of the samples used
            when building trees (if ``bootstrap=True``) and the sampling of the
            features to consider when looking for the best split at each node
            (if ``max_features < n_features``).
            See :term:`Glossary <random_state>` for details.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`the Glossary <warm_start>`.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.

            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0.0, 1.0]`.

            .. versionadded:: 0.22

        References
        ----------
        scikit API: sklearn.ensemble.RandomForestRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor
        """
        
        super().__init__()
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


class SupportVectorRegression(RegressionWorkflowBase):
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


class DNNRegression(RegressionWorkflowBase, BaseEstimator):

    name = "Deep Neural Networks"
    special_function = ["Loss Record"]

    def __init__(
            self,
            hidden_layer_sizes: tuple = (50, 25, 5),
            activation: List[str] = 'relu',
            solver: List[str] ='adam',
            alpha: float = 0.0001,
            batch_size: Union[int, str] ='auto',
            learning_rate: List[str] = 'constant',
            learning_rate_init: float = 0.001,
            max_iter: int = 200,
            shuffle: bool = True,
            random_state: int = None,
            tol: float = 1e-4,
            verbose: bool = False,
            warm_start: bool = False,
            early_stopping: bool = False,
            validation_fraction: float = 0.1,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-8,
            n_iter_no_change: int = 10,
    ):


        """
        Parameters
        ----------
        hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
            The ith element represents the number of neurons in the ith
            hidden layer.

        activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            Activation function for the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck,
              returns f(x) = x
            - 'logistic', the logistic sigmoid function,
              returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function,
              returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function,
              returns f(x) = max(0, x)

        solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
            The solver for weight optimization.
            - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
            - 'sgd' refers to stochastic gradient descent.
            - 'adam' refers to a stochastic gradient-based optimizer proposed by
              Kingma, Diederik, and Jimmy Ba
            Note: The default solver 'adam' works pretty well on relatively
            large datasets (with thousands of training samples or more) in terms of
            both training time and validation score.
            For small datasets, however, 'lbfgs' can converge faster and perform
            better.

        alpha : float, default=0.0001
            Strength of the L2 regularization term. The L2 regularization term
            is divided by the sample size when added to the loss.

        batch_size : int, default='auto'
            Size of minibatches for stochastic optimizers.
            If the solver is 'lbfgs', the classifier will not use minibatch.
            When set to "auto", `batch_size=min(200, n_samples)`.

        learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
            Learning rate schedule for weight updates.
            - 'constant' is a constant learning rate given by
              'learning_rate_init'.
            - 'invscaling' gradually decreases the learning rate ``learning_rate_``
              at each time step 't' using an inverse scaling exponent of 'power_t'.
              effective_learning_rate = learning_rate_init / pow(t, power_t)
            - 'adaptive' keeps the learning rate constant to
              'learning_rate_init' as long as training loss keeps decreasing.
              Each time two consecutive epochs fail to decrease training loss by at
              least tol, or fail to increase validation score by at least tol if
              'early_stopping' is on, the current learning rate is divided by 5.
            Only used when solver='sgd'.

        learning_rate_init : float, default=0.001
            The initial learning rate used. It controls the step-size
            in updating the weights. Only used when solver='sgd' or 'adam'.

        power_t : float, default=0.5
            The exponent for inverse scaling learning rate.
            It is used in updating effective learning rate when the learning_rate
            is set to 'invscaling'. Only used when solver='sgd'.

        max_iter : int, default=200
            Maximum number of iterations. The solver iterates until convergence
            (determined by 'tol') or this number of iterations. For stochastic
            solvers ('sgd', 'adam'), note that this determines the number of epochs
            (how many times each data point will be used), not the number of
            gradient steps.

        shuffle : bool, default=True
            Whether to shuffle samples in each iteration. Only used when
            solver='sgd' or 'adam'.

        random_state : int, RandomState instance, default=None
            Determines random number generation for weights and bias
            initialization, train-test split if early stopping is used, and batch
            sampling when solver='sgd' or 'adam'.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        tol : float, default=1e-4
            Tolerance for the optimization. When the loss or score is not improving
            by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
            unless ``learning_rate`` is set to 'adaptive', convergence is
            considered to be reached and training stops.

        verbose : bool, default=False
            Whether to print progress messages to stdout.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous
            call to fit as initialization, otherwise, just erase the
            previous solution. See :term:`the Glossary <warm_start>`.

        momentum : float, default=0.9
            Momentum for gradient descent update.  Should be between 0 and 1. Only
            used when solver='sgd'.

        nesterovs_momentum : bool, default=True
            Whether to use Nesterov's momentum. Only used when solver='sgd' and
            momentum > 0.

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation
            score is not improving. If set to true, it will automatically set
            aside 10% of training data as validation and terminate training when
            validation score is not improving by at least ``tol`` for
            ``n_iter_no_change`` consecutive epochs.
            Only effective when solver='sgd' or 'adam'.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
            Only used if early_stopping is True.

        beta_1 : float, default=0.9
            Exponential decay rate for estimates of first moment vector in adam,
            should be in [0, 1). Only used when solver='adam'.

        beta_2 : float, default=0.999
            Exponential decay rate for estimates of second moment vector in adam,
            should be in [0, 1). Only used when solver='adam'.

        epsilon : float, default=1e-8
            Value for numerical stability in adam. Only used when solver='adam'.

        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet ``tol`` improvement.
            Only effective when solver='sgd' or 'adam'.

            .. versionadded:: 0.20

        max_fun : int, default=15000
            Only used when solver='lbfgs'. Maximum number of function calls.
            The solver iterates until convergence (determined by 'tol'), number
            of iterations reaches max_iter, or this number of function calls.
            Note that number of function calls will be greater than or equal to
            the number of iterations for the MLPRegressor.

            .. versionadded:: 0.22

        References
        ----------------------------------------
        Hinton, Geoffrey E. "Connectionist learning procedures."
        Artificial intelligence 40.1 (1989): 185-234.
        Glorot, Xavier, and Yoshua Bengio.
        "Understanding the difficulty of training deep feedforward neural networks."
        International Conference on Artificial Intelligence and Statistics. 2010.
        :arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
        Surpassing human-level performance on imagenet classification." <1502.01852>`
        :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
        "Adam: A method for stochastic optimization." <1412.6980>`
        """
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                   activation=self.activation,
                                   solver=self.solver,
                                   alpha=self.alpha,
                                   batch_size=self.batch_size,
                                   learning_rate=self.learning_rate,
                                   learning_rate_init=self.learning_rate_init,
                                   max_iter=self.max_iter,
                                   shuffle=self.shuffle,
                                   random_state=self.random_state,
                                   tol=self.tol,
                                   verbose=self.verbose,
                                   warm_start=self.warm_start,
                                   early_stopping=self.early_stopping,
                                   validation_fraction=self.validation_fraction,
                                   beta_1=self.beta_1,
                                   beta_2=self.beta_2,
                                   epsilon=self.epsilon,
                                   n_iter_no_change=self.n_iter_no_change)

        self.naming = DNNRegression.name

    def plot_learning_curve(self, algorithm_name: str, store_path):
        print("-----* Loss Record *-----")
        pd.DataFrame(self.model.loss_curve_).plot(title="Loss")
        save_fig(f'Loss Record - {algorithm_name}', store_path)

    @staticmethod
    def _plot_pred(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str, store_path: str):
        print("-----* Truth v.s. Prediction *-----")
        plot_pred(y_test_predict, y_test, algorithm_name)
        save_fig(f'Ground Truth v.s. Prediction - {algorithm_name}', store_path)

    def special_components(self, **kwargs) -> None:
        self.plot_learning_curve(self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._plot_pred(y_test_predict=DNNRegression.y_test_predict,
                        y_test=DNNRegression.y_test, algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)


class LinearRegression2(RegressionWorkflowBase):

    name = "Linear Regression"
    special_function = ["Linear Regression Formula", "Two/Three-dimensional Linear Regression Image"]

    def __init__(
            self,
            fit_intercept: bool = True,
            normalize: bool = False,
            copy_X: bool = True,
            n_jobs: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
        normalize : bool, default=False
            This parameter is ignored when ``fit_intercept`` is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use
                            :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
            on an estimator with ``normalize=False``.
            .. deprecated:: 1.0
               `normalize` was deprecated in version 1.0 and will be
               removed in 1.2.
        copy_X : bool, default=True
                        If True, X will be copied; else, it may be overwritten.
        n_jobs : int, default=None
            The number of jobs to use for the computation. This will only provide
            speedup in case of sufficiently large problems, that is if firstly
                            `n_targets > 1` and secondly `X` is sparse or if `positive` is set
            to `True`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. See :term:`Glossary <n_jobs>` for more details.
        positive : bool, default=False
                        When set to ``True``, forces the coefficients to be positive. This
            option is only supported for dense arrays.
            .. versionadded:: 0.24

        References
        ----------
        scikit API: sklearn.linear_model.LinearRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression
        """
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.model = LinearRegression(fit_intercept=self.fit_intercept,
                                      copy_X=self.copy_X,
                                      normalize=self.normalize,
                                      n_jobs=self.n_jobs)
        self.naming = LinearRegression2.name

    @staticmethod
    def _show_formula(coef, intercept, columns_name):
        print("-----* Linear Regression Formula *-----")
        show_formula(coef, intercept, columns_name)

    @staticmethod
    def _plot_2d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str,
                       store_path: str):
        print("-----* Plot 2D Graph *-----")
        plot_2d_graph(feature_data, target_data)
        save_fig(f"2D Scatter Graph - {algorithm_name}", store_path)

    @staticmethod
    def _plot_3d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str,
                       store_path: str):
        print("-----* Plot 3D Graph *-----")
        plot_3d_graph(feature_data, target_data)
        save_fig(f"3D Scatter Graph - {algorithm_name}", store_path)

    def special_components(self, **kwargs):
        self._show_formula(coef=self.model.coef_, intercept=self.model.intercept_,
                           columns_name=LinearRegression2.X.columns)

        columns_num = LinearRegression2.X.shape[1]
        if kwargs['n_dimen'] == 2:
            if columns_num > 1:
                # choose one of dimensions to draw
                two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LinearRegression2.X, 1)
                self._plot_2d_graph(feature_data=two_dimen_data, target_data=LinearRegression2.y,
                                    algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
            elif columns_num == 1:
                # no need to choose
                self._plot_2d_graph(feature_data=LinearRegression2.X, target_data=LinearRegression2.y,
                                    algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        elif kwargs['n_dimen'] == 3:
            if columns_num > 2:
                # choose two of dimensions to draw
                three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(LinearRegression2.X, 2)
                self._plot_3d_graph(feature_data=three_dimen_data, target_data=LinearRegression2.y,
                                    algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
            elif columns_num == 2:
                # no need to choose
                self._plot_3d_graph(feature_data=LinearRegression2.X, target_data=LinearRegression2.y,
                                    algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        else:
            pass
