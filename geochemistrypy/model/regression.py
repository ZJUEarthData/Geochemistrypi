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
from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
# sys.path.append("..")


class RegressionWorkflowBase(object):

    # Default for child class
    X = None
    y = None
    name = None
    common_function = ['Model Score', 'Cross Validation']
    special_function = None

    @classmethod
    def show_info(cls):
        print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
        print("Expected Functionality:")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print("+ ", function[i])

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = None

    @staticmethod
    def data_split(X_data, y_data, test_size=0.2, random_state=42):
        RegressionWorkflowBase.X = X_data  # child class is able to access to the data
        RegressionWorkflowBase.y = y_data
        X_train, X_test, y_train, y_test = train_test_split(RegressionWorkflowBase.X,
                                                            RegressionWorkflowBase.y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_test_prediction = self.model.predict(X_test)
        return y_test_prediction

    @staticmethod
    def score(y_test, y_test_prediction):
        mse = mean_squared_error(y_test, y_test_prediction)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_prediction)
        r2 = r2_score(y_test, y_test_prediction)
        evs = explained_variance_score(y_test, y_test_prediction)
        print("-----* Model Score *-----")
        print("RMSE score:", rmse)
        print("MAE score:", mae)
        print("R2 score:", r2)
        print("Explained Variance Score:", evs)
        # return rmse, mae

    @staticmethod
    def _display_cross_validation_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    def cross_validation(self, X_train, y_train, cv_num=10):
        # param_grid = {}
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

    # TODO: How to prevent overfitting
    def is_overfitting():
        pass

    # TODO: Do Hyperparameter Searching
    def search_best_hyper_parameter():
        pass


class PolynomialRegression(RegressionWorkflowBase, BaseEstimator):

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

        super().__init__(random_state=42)
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

        self.__features_name = None
        self.__coefficient = None
        self.__intercept = None

    def poly(self, X_train, X_test):
        poly_features = PolynomialFeatures(degree=self.degree,
                                           include_bias=self.is_include_bias,
                                           interaction_only=self.interaction_only,
                                           order=self.order)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.fit_transform(X_test)
        try:
            # scikit-learn >= 1.0
            self.__features_name = poly_features.get_feature_names_out()
        except AttributeError:
            self.__features_name = poly_features.get_feature_names()
        return X_train_poly, X_test_poly

    def _show_formula(self):
        print("-----* Polynomial Regression Formula *-----")
        self.__coefficient = self.model.coef_
        self.__intercept = self.model.intercept_
        term = []
        coef = np.around(self.__coefficient, decimals=3).tolist()[0]
        for i in range(len(coef)):
            # the first value stay the same
            if i == 0:
                # not append if zero
                if coef[i] != 0:
                    temp = str(coef[i]) + self.__features_name[i]
                    term.append(temp)
            else:
                # add plus symbol if positive, maintain if negative, not append if zero
                if coef[i] > 0:
                    temp = '+' + str(coef[i]) + self.__features_name[i]
                    term.append(temp)
                elif coef[i] < 0:
                    temp = str(coef[i]) + self.__features_name[i]
                    term.append(temp)
        if self.__intercept[0] >= 0:
            # formula of polynomial regression
            formula = ''.join(term) + '+' + str(self.__intercept[0])
        else:
            formula = ''.join(term) + str(self.__intercept[0])
        print('y =', formula)

    def special_components(self):
        self._show_formula()


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
                 ):
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

        self.model = DecisionTreeRegressor()

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

