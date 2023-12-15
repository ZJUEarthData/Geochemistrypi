# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import xgboost
from flaml import AutoML
from multipledispatch import dispatch
from rich import print
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..constants import MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH, RAY_FLAML
from ..utils.base import clear_output, save_data, save_fig, save_text
from ._base import LinearWorkflowMixin, TreeWorkflowMixin, WorkflowBase
from .func.algo_regression._common import cross_validation, plot_predicted_vs_actual, plot_residuals, score
from .func.algo_regression._decision_tree import decision_tree_manual_hyper_parameters
from .func.algo_regression._elastic_net import elastic_net_manual_hyper_parameters
from .func.algo_regression._extra_tree import extra_trees_manual_hyper_parameters
from .func.algo_regression._gradient_boosting import gradient_boosting_manual_hyper_parameters
from .func.algo_regression._knn import knn_manual_hyper_parameters
from .func.algo_regression._lasso_regression import lasso_regression_manual_hyper_parameters
from .func.algo_regression._linear_regression import linear_regression_manual_hyper_parameters
from .func.algo_regression._multi_layer_perceptron import multi_layer_perceptron_manual_hyper_parameters
from .func.algo_regression._polynomial_regression import polynomial_regression_manual_hyper_parameters
from .func.algo_regression._rf import random_forest_manual_hyper_parameters
from .func.algo_regression._sgd_regression import sgd_regression_manual_hyper_parameters
from .func.algo_regression._svr import svr_manual_hyper_parameters
from .func.algo_regression._xgboost import xgboost_manual_hyper_parameters


class RegressionWorkflowBase(WorkflowBase):
    """The base workflow class of regression algorithms."""

    common_function = ["Model Score", "Cross Validation", "Model Prediction", "Model Persistence", "Predicted vs. Actual Diagram", "Residuals Diagram", "Permutation Importance Diagram"]

    def __init__(self) -> None:
        super().__init__()
        # These two attributes are used for the customized models of FLAML framework
        self.customized = False
        self.customized_name = None
        self.mode = "Regression"

    @dispatch(object, object)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model by Scikit-learn framework."""
        self.model.fit(X, y)

    @dispatch(object, object, bool)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, is_automl: bool = False) -> None:
        """Fit the model by FLAML framework and RAY framework."""
        # print(f"-*-*- {self.naming} - AutoML -*-*-.")
        if self.naming not in RAY_FLAML:
            self.automl = AutoML()
            if self.customized:  # When the model is not built-in in FLAML framwork, use FLAML customization.
                self.automl.add_learner(learner_name=self.customized_name, learner_class=self.customization)
            if y.shape[1] == 1:  # FLAML's data format validation mechanism
                y = y.squeeze()  # Convert a single dataFrame column into a series
            self.automl.fit(X_train=X, y_train=y, **self.settings)
        else:
            # When the model is not built-in in FLAML framework, use RAY + FLAML customization.
            self.ray_tune(
                RegressionWorkflowBase.X_train,
                RegressionWorkflowBase.X_test,
                RegressionWorkflowBase.y_train,
                RegressionWorkflowBase.y_test,
            )
            self.ray_best_model.fit(X, y)

    @dispatch(object)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform classification on samples in X by Scikit-learn framework."""
        y_predict = self.model.predict(X)
        return y_predict

    @dispatch(object, bool)
    def predict(self, X: pd.DataFrame, is_automl: bool = False) -> np.ndarray:
        """Perform classification on samples in X by FLAML framework and RAY framework."""
        if self.naming not in RAY_FLAML:
            y_predict = self.automl.predict(X)
            return y_predict
        else:
            y_predict = self.ray_best_model.predict(X)
            return y_predict

    @property
    def auto_model(self) -> object:
        """Get AutoML trained model by FLAML framework and RAY framework."""
        if self.naming not in RAY_FLAML:
            return self.automl.model.estimator
        else:
            return self.ray_best_model

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        return dict()

    @property
    def customization(self) -> object:
        """The customized model of FLAML framework."""
        return object

    def ray_tune(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> object:
        """The customized model of FLAML framework and RAY framework."""
        return object

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    @staticmethod
    def _plot_predicted_vs_actual(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the predicted vs. actual diagram."""
        print("-----* Predicted vs. Actual Diagram *-----")
        plot_predicted_vs_actual(y_test_predict, y_test, algorithm_name)
        save_fig(f"Predicted vs. Actual Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([y_test, y_test_predict], axis=1)
        save_data(data, f"Predicted vs. Actual Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _plot_residuals(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the residuals diagram."""
        print("-----* Residuals Diagram *-----")
        residuals = plot_residuals(y_test_predict, y_test, algorithm_name)
        save_fig(f"Residuals Diagram - {algorithm_name}", local_path, mlflow_path)
        data = pd.concat([y_test, residuals], axis=1)
        save_data(data, f"Residuals Diagram - {algorithm_name}", local_path, mlflow_path)

    @staticmethod
    def _score(y_true: pd.DataFrame, y_predict: pd.DataFrame, algorithm_name: str, store_path: str) -> None:
        """Calculate the score of the model."""
        print("-----* Model Score *-----")
        scores = score(y_true, y_predict)
        scores_str = json.dumps(scores, indent=4)
        save_text(scores_str, f"Model Score - {algorithm_name}", store_path)
        mlflow.log_metrics(scores)

    @staticmethod
    def _cross_validation(trained_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, cv_num: int, algorithm_name: str, store_path: str) -> None:
        """Cross validation."""
        print("-----* Cross Validation *-----")
        print(f"K-Folds: {cv_num}")
        scores = cross_validation(trained_model, X_train, y_train, cv_num=cv_num)
        scores_str = json.dumps(scores, indent=4)
        save_text(scores_str, f"Cross Validation - {algorithm_name}", store_path)

    @dispatch()
    def common_components(self) -> None:
        """Invoke all common application functions for regression algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._score(
            y_true=RegressionWorkflowBase.y_test,
            y_predict=RegressionWorkflowBase.y_test_predict,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._cross_validation(
            trained_model=self.model,
            X_train=RegressionWorkflowBase.X_train,
            y_train=RegressionWorkflowBase.y_train,
            cv_num=10,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._plot_predicted_vs_actual(
            y_test_predict=RegressionWorkflowBase.y_test_predict,
            y_test=RegressionWorkflowBase.y_test,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_residuals(
            y_test_predict=RegressionWorkflowBase.y_test_predict,
            y_test=RegressionWorkflowBase.y_test,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_permutation_importance(
            X_test=RegressionWorkflowBase.X_test,
            y_test=RegressionWorkflowBase.y_test,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def common_components(self, is_automl: bool) -> None:
        """Invoke all common application functions for regression algorithms by FLAML framework."""
        GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._score(
            y_true=RegressionWorkflowBase.y_test,
            y_predict=RegressionWorkflowBase.y_test_predict,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._cross_validation(
            trained_model=self.auto_model,
            X_train=RegressionWorkflowBase.X_train,
            y_train=RegressionWorkflowBase.y_train,
            cv_num=10,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._plot_predicted_vs_actual(
            y_test_predict=RegressionWorkflowBase.y_test_predict,
            y_test=RegressionWorkflowBase.y_test,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_residuals(
            y_test_predict=RegressionWorkflowBase.y_test_predict,
            y_test=RegressionWorkflowBase.y_test,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_permutation_importance(
            X_test=RegressionWorkflowBase.X_test,
            y_test=RegressionWorkflowBase.y_test,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class PolynomialRegression(LinearWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Polynomial Regression algorithm to make insightful products."""

    name = "Polynomial Regression"
    special_function = ["Polynomial Regression Formula"]

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        order: str = "C",
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.order = order
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X, n_jobs=self.n_jobs)

        self._features_name = None
        self.naming = PolynomialRegression.name

    def poly(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Polynomial features."""
        poly_features = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias, interaction_only=self.interaction_only, order=self.order)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.fit_transform(X_test)
        poly_config = {type(poly_features).__name__: poly_features.get_params()}
        try:
            # scikit-learn >= 1.0
            self._features_name = poly_features.get_feature_names_out()
        except AttributeError:
            self._features_name = poly_features.get_feature_names()
        X_train_poly = pd.DataFrame(X_train_poly, columns=self._features_name)
        X_test_poly = pd.DataFrame(X_test_poly, columns=self._features_name)
        return poly_config, X_train_poly, X_test_poly

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = polynomial_regression_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            features_name=self._features_name,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )


class XGBoostRegression(TreeWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Xgboost algorithm to make insightful products."""

    name = "Xgboost"
    special_function = ["Feature Importance Diagram"]

    # In fact, it's used for type hint in the original xgboost package.
    # Hence, we have to copy it here again. Just ignore it
    _SklObjective = Optional[Union[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]]

    def __init__(
        self,
        max_depth: Optional[int] = 6,
        learning_rate: Optional[float] = 0.3,
        n_estimators: int = 100,
        verbosity: Optional[int] = 1,
        objective: _SklObjective = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = "auto",
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = 0,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = 0,
        subsample: Optional[float] = 1,
        colsample_bytree: Optional[float] = 1,
        colsample_bylevel: Optional[float] = 1,
        colsample_bynode: Optional[float] = 1,
        reg_alpha: Optional[float] = 0,
        reg_lambda: Optional[float] = 1,
        scale_pos_weight: Optional[float] = 1,
        base_score: Optional[float] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = 1,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = "gain",
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        # enable_categorical: bool = False,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        max_depth [default=6]
            Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
            0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
            exact tree method requires non-zero value.
            range: [0,∞]

        learning_rate [default=0.3]
            Step size shrinkage used in update to prevents overfitting.
            After each boosting step, we can directly get the weights of new features,
            and eta shrinks the feature weights to make the boosting process more conservative.
            range: [0,1]

        n_estimators : int
        Number of gradient boosted trees.  Equivalent to number of boosting rounds.

        objective : {SklObjective}
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).

        verbosity [default=1]
            Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
            Sometimes XGBoost tries to change configurations based on heuristics,
            which is displayed as warning message.
            If there’s unexpected behaviour, please try to increase value of verbosity.

        booster [default= gbtree ]
            Which booster to use. Can be gbtree, gblinear or dart;
            gbtree and dart use tree based models while gblinear uses linear functions.

        tree_method string [default= auto]
            The tree construction algorithm used in XGBoost. See description in the reference paper and Tree Methods.
            XGBoost supports approx, hist and gpu_hist for distributed training. Experimental support for external memory is available for approx and gpu_hist.
            Choices: auto, exact, approx, hist, gpu_hist, this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
                auto: Use heuristic to choose the fastest method.
                    For small dataset, exact greedy (exact) will be used.
                    For larger dataset, approximate algorithm (approx) will be chosen. It’s recommended to try hist and gpu_hist for higher performance with large dataset.
                      (gpu_hist)has support for external memory.
                    Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
                exact: Exact greedy algorithm. Enumerates all split candidates.
                approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
                hist: Faster histogram optimized approximate greedy algorithm.
                gpu_hist: GPU implementation of hist algorithm.

        n_jobs : Optional[int]
            Number of parallel threads used to run xgboost.  When used with other
            Scikit-Learn algorithms like grid search, you may choose which algorithm to
            parallelize and balance the threads.  Creating thread contention will
            significantly slow down both algorithms.

        gamma [default=0, alias: min_split_loss]
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            The larger gamma is, the more conservative the algorithm will be.
            range: [0,∞]

        min_child_weight [default=1]
            Minimum sum of instance weight (hessian) needed in a child.
            If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
            then the building process will give up further partitioning. In linear regression task,
            this simply corresponds to minimum number of instances needed to be in each node.
            The larger min_child_weight is, the more conservative the algorithm will be.
            range: [0,∞]

        max_delta_step [default=0]
            Maximum delta step we allow each leaf output to be.
            If the value is set to 0, it means there is no constraint.
            If it is set to a positive value, it can help making the update step more conservative.
            Usually this parameter is not needed, but it might help in logistic regression
            when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
            range: [0,∞]

        subsample [default=1]
            Subsample ratio of the training instances.
            Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
            and this will prevent overfitting.
            Subsampling will occur once in every boosting iteration.
            range: (0,1]

        colsample_bytree [default=1]
            colsample_bytree is the subsample ratio of columns when constructing each tree.
            Subsampling occurs once for every tree constructed.

        colsample_bylevel [default=1]
            colsample_bylevel is the subsample ratio of columns for each level.
            Subsampling occurs once for every new depth level reached in a tree.
            Columns are subsampled from the set of columns chosen for the current tree.

        colsample_bynode [default=1]
            colsample_bynode is the subsample ratio of columns for each node (split).
            Subsampling occurs once every time a new split is evaluated.
            Columns are subsampled from the set of columns chosen for the current level.

        reg_alpha [default=0]
            L1 regularization term on weights.
            Increasing this value will make model more conservative.

        reg_lambda [default=1, alias: reg_lambda]
            L2 regularization term on weights.
            Increasing this value will make model more conservative.

        scale_pos_weight [default=1]
            Control the balance of positive and negative weights, useful for unbalanced classes.

        predictor, [default= auto]
            The type of predictor algorithm to use.
            Provides the same results but allows the use of GPU or CPU.
                auto: Configure predictor based on heuristics.
                cpu_predictor: Multicore CPU prediction algorithm.
                gpu_predictor: Prediction using GPU. Used when tree_method is gpu_hist.
                    When predictor is set to default value auto, the gpu_hist tree method is able to provide GPU based prediction
                    without copying training data to GPU memory. If gpu_predictor is explicitly specified,
                    then all data is copied into GPU, only recommended for performing prediction tasks.

        base_score : Optional[float]
            The initial prediction score of all instances, global bias.

        random_state : Optional[Union[numpy.random.RandomState, int]]
            Random number seed.
            .. note::

               Using gblinear booster with shotgun updater is nondeterministic as
               it uses Hogwild algorithm.

        missing : float, default np.nan
        Value in the data which needs to be present as a missing value.

        num_parallel_tree: Optional[int]
            Used for boosting random forest.

        monotone_constraints : Optional[Union[Dict[str, int], str]]
            Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
            for more information.

        interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
            Constraints for interaction representing permitted interactions.  The
            constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
            3, 4]]``, where each inner list is a group of indices of features that are
            allowed to interact with each other.  See :doc:`tutorial
            </tutorials/feature_interaction_constraint>` for more information

        importance_type: Optional[str]
            The feature importance type for the feature_importances\\_ property:

            * For tree model, it's either "gain", "weight", "cover", "total_gain" or
              "total_cover".
            * For linear model, only "weight" is defined and it's the normalized coefficients
              without bias.

        gpu_id : Optional[int]
            Device ordinal.

        validate_parameters : Optional[bool]
            Give warnings for unknown parameter.

        eval_metric : Optional[Union[str, List[str], Callable]]

        early_stopping_rounds : Optional[int]

        References
        ----------
        [1] Xgboost Python API Reference - Scikit-Learn API
            https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

        [2] Xgboost API for the scikit-learn wrapper:
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
        # self.enable_categorical = enable_categorical
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
            # enable_categorical=self.enable_categorical,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.naming = XGBoostRegression.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": ["xgboost"],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f"{self.naming} - automl.log",  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = xgboost_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    # @staticmethod
    # def _histograms_feature_weights(X: pd.DataFrame, trained_model: object, image_config: dict, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
    #     """Histograms of feature weights plot."""
    #     histograms_feature_weights(X, trained_model, image_config)
    #     save_fig(f"Feature Importance Score - {algorithm_name}", local_path, mlflow_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=XGBoostRegression.X_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        # self._histograms_feature_weights(
        #     X=XGBoostRegression.X,
        #     trained_model=self.model,
        #     image_config=self.image_config,
        #     algorithm_name=self.naming,
        #     local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
        #     mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        # )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=XGBoostRegression.X_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        # self._histograms_feature_weights(
        #     X=XGBoostRegression.X,
        #     trained_model=self.auto_model,
        #     image_config=self.image_config,
        #     algorithm_name=self.naming,
        #     local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
        #     mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        # )


class DecisionTreeRegression(TreeWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Decision Tree algorithm to make insightful products."""

    name = "Decision Tree"
    special_function = ["Feature Importance Diagram", "Single Tree Diagram"]

    def __init__(
        self,
        criterion: str = "squared_error",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
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
        Scikit-learn API: sklearn.tree.DecisionTreeClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        """

        super().__init__()
        self.criterion = (criterion,)
        self.splitter = (splitter,)
        self.max_depth = (max_depth,)
        self.min_samples_split = (min_samples_split,)
        self.min_samples_leaf = (min_samples_leaf,)
        self.min_weight_fraction_leaf = (min_weight_fraction_leaf,)
        self.max_features = (max_features,)
        self.random_state = (random_state,)
        self.max_leaf_nodes = (max_leaf_nodes,)
        self.min_impurity_decrease = (min_impurity_decrease,)
        self.ccp_alpha = ccp_alpha

        self.model = DecisionTreeRegressor(
            criterion=self.criterion[0],
            splitter=self.splitter[0],
            max_depth=self.max_depth[0],
            min_samples_split=self.min_samples_split[0],
            min_samples_leaf=self.min_samples_leaf[0],
            min_weight_fraction_leaf=self.min_weight_fraction_leaf[0],
            max_features=self.max_features[0],
            random_state=self.random_state[0],
            max_leaf_nodes=self.max_leaf_nodes[0],
            min_impurity_decrease=self.min_impurity_decrease[0],
            ccp_alpha=self.ccp_alpha,
        )
        self.naming = DecisionTreeRegression.name
        self.customized = True
        self.customized_name = "Decision Tree"

    @property
    def settings(self) -> Dict:
        """The configuration of Decision Tree to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Decision Tree of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.tree import DecisionTreeRegressor

        class MyDTRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = DecisionTreeRegressor

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "criterion": {"domain": tune.choice(["squared_error", "friedman_mse", "absolute_error", "poisson"])},
                    "max_depth": {"domain": tune.randint(lower=2, upper=20), "init_value": 1, "low_cost_init_value": 1},
                    "min_samples_split": {
                        "domain": tune.randint(lower=2, upper=10),
                        "init_value": 2,
                        "low_cost_init_value": 2,
                    },
                    "min_samples_leaf": {"domain": tune.randint(lower=1, upper=10), "init_value": 1, "low_cost_init_value": 1},
                    "max_features": {"domain": tune.randint(lower=1, upper=10), "init_value": 1, "low_cost_init_value": 1},
                }
                return space

        return MyDTRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = decision_tree_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self):
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=DecisionTreeRegression.X_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=DecisionTreeRegression.X_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class ExtraTreesRegression(TreeWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Extra-Trees algorithm to make insightful products."""

    name = "Extra-Trees"
    special_function = ["Feature Importance Diagram", "Single Tree Diagram"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str] = "auto",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        #  min_impurity_split=None,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float] = None,
    ) -> None:
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
        Scikit-learn API: sklearn.ensemble.ExtraTreesRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extratreesregressor#sklearn.ensemble.ExtraTreesRegressor
        """

        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        # self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            # min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )

        self.naming = ExtraTreesRegression.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": ["extra_tree"],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = extra_trees_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=ExtraTreesRegression.X_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=ExtraTreesRegression.X_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class RandomForestRegression(TreeWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Random Forest algorithm to make insightful products."""

    name = "Random Forest"
    special_function = ["Feature Importance Diagram", "Single Tree Diagram"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float] = "sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,
        # class_weight=None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float] = None,
    ) -> None:
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
        Scikit-learn API: sklearn.ensemble.RandomForestRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor
        """

        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        # self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            # class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        self.naming = RandomForestRegression.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": ["rf"],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = random_forest_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=RandomForestRegression.X_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=RandomForestRegression.X_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class SVMRegression(RegressionWorkflowBase):
    """The automation workflow of using SVR algorithm to make insightful products."""

    name = "Support Vector Machine"
    special_function = []

    def __init__(
        self,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        # coef0: float = 0.0,
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
        Scikit-learn API: sklearn.svm.SVR
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        """

        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        # self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

        self.model = SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            # coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )

        self.naming = SVMRegression.name
        self.customized = True
        self.customized_name = "SVR"

    @property
    def settings(self) -> Dict:
        """The configuration of SVR to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SVR of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.svm import SVR

        class MySVMRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = SVR

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "C": {"domain": tune.uniform(lower=1, upper=data_size[0]), "init_value": 1, "low_cost_init_value": 1},
                    "kernel": {"domain": tune.choice(["poly", "rbf", "sigmoid"])},
                    "gamma": {"domain": tune.uniform(lower=1e-5, upper=10), "init_value": 1e-1, "low_cost_init_value": 1e-1},
                    "degree": {"domain": tune.quniform(lower=1, upper=5, q=1), "init_value": 3, "low_cost_init_value": 3},
                    "coef0": {"domain": tune.uniform(lower=0, upper=1), "init_value": 0, "low_cost_init_value": 0},
                    "shrinking": {"domain": tune.choice([True, False])},
                }
                return space

        return MySVMRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = svr_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs):
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        pass


class MLPRegression(RegressionWorkflowBase):
    """The automation workflow of using Multi-layer Perceptron algorithm to make insightful products."""

    name = "Multi-layer Perceptron"
    special_function = ["Loss Curve Diagram"]

    def __init__(
        self,
        hidden_layer_sizes: tuple = (50, 25, 5),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
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
            Tolerance for thfe optimization. When the loss or score is not improving
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
        ----------
        Scikit-learn API: sklearn.neural_network.MLPRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
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

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
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
            n_iter_no_change=self.n_iter_no_change,
        )

        self.naming = MLPRegression.name

    def ray_tune(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """The customized MLP of the combinations of Ray, FLAML and Scikit-learn framework."""

        from ray import tune
        from ray.air import session
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.search.flaml import BlendSearch
        from sklearn.metrics import mean_squared_error

        def customized_model(l1: int, l2: int, l3: int, batch: int) -> object:
            """The customized model by Scikit-learn framework."""
            return MLPRegressor(hidden_layer_sizes=(l1, l2, l3), batch_size=batch)

        def evaluate(l1: int, l2: int, l3: int, batch: int) -> float:
            """The evaluation function by simulating a long-running ML experiment
            to get the model's performance at every epoch."""
            regr = customized_model(l1, l2, l3, batch)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)  # Use RMSE score
            return rmse

        def objective(config: Dict) -> None:
            """Objective function takes a Tune config, evaluates the score of your experiment in a training loop,
            and uses session.report to report the score back to Tune."""
            for step in range(config["steps"]):
                score = evaluate(config["l1"], config["l2"], config["l3"], config["batch"])
                session.report({"iterations": step, "mean_loss": score})

        # Search space: The critical assumption is that the optimal hyper-parameters live within this space.
        search_config = {
            "l1": tune.randint(1, 20),
            "l2": tune.randint(1, 30),
            "l3": tune.randint(1, 20),
            "batch": tune.randint(20, 100),
        }

        # Define the time budget in seconds.
        time_budget_s = 30

        # Integrate with FLAML's BlendSearch to implement hyper-parameters optimization .
        algo = BlendSearch(metric="mean_loss", mode="min", space=search_config)
        algo.set_search_properties(config={"time_budget_s": time_budget_s})
        algo = ConcurrencyLimiter(algo, max_concurrent=4)

        # Use Ray Tune to  run the experiment to "min"imize the “mean_loss” of the "objective"
        # by searching "search_config" via "algo", "num_samples" times.
        tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                search_alg=algo,
                num_samples=-1,
                time_budget_s=time_budget_s,
            ),
            param_space={"steps": 100},
        )
        results = tuner.fit()

        # The hyper-parameters found to minimize the mean loss of the defined objective and the corresponding model.
        best_result = results.get_best_result(metric="mean_loss", mode="min")
        self.ray_best_model = customized_model(best_result.config["l1"], best_result.config["l2"], best_result.config["l3"], best_result.config["batch"])

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = multi_layer_perceptron_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @staticmethod
    def _plot_loss_curve(trained_model: object, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the learning curve of the trained model."""
        print("-----* Loss Curve Diagram *-----")
        data = pd.DataFrame(trained_model.loss_curve_, columns=["Loss"])
        data.plot(title="Loss")
        save_fig(f"Loss Curve Diagram - {algorithm_name}", local_path, mlflow_path)
        save_data(data, f"Loss Curve Diagram - {algorithm_name}", local_path, mlflow_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        if self.model.get_params()["solver"] in ["sgd", "adam"]:
            self._plot_loss_curve(
                trained_model=self.model,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        if self.model.get_params()["solver"] in ["sgd", "adam"]:
            self._plot_loss_curve(
                trained_model=self.auto_model,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )


class ClassicalLinearRegression(LinearWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Linear Regression algorithm to make insightful products."""

    name = "Linear Regression"
    special_function = ["Linear Regression Formula", "2D Scatter Diagram", "3D Scatter Diagram", "2D Line Diagram", "3D Surface Diagram"]

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        positive: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).

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
        Scikit-learn API: sklearn.linear_model.LinearRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression
        """
        super().__init__()
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

        self.model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )

        self.naming = ClassicalLinearRegression.name

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = linear_regression_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._show_formula(
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            features_name=ClassicalLinearRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = ClassicalLinearRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ClassicalLinearRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ClassicalLinearRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(ClassicalLinearRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=ClassicalLinearRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ClassicalLinearRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ClassicalLinearRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=ClassicalLinearRegression.X_test,
                target_data=ClassicalLinearRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=ClassicalLinearRegression.X_test,
                target_data=ClassicalLinearRegression.y_test,
                y_test_predict=ClassicalLinearRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=ClassicalLinearRegression.X_test,
                target_data=ClassicalLinearRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=ClassicalLinearRegression.X_test,
                target_data=ClassicalLinearRegression.y_test,
                y_test_predict=ClassicalLinearRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass


class KNNRegression(RegressionWorkflowBase):
    """The automation workflow of using KNN algorithm to make insightful products."""

    name = "K-Nearest Neighbors"
    special_function = []

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Optional[Dict] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use by default for :meth:`kneighbors` queries.

        weights : {'uniform', 'distance'}, callable or None, default='uniform'
            Weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights.  All points in each neighborhood
            are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
            array of distances, and returns an array of the same shape
            containing the weights.

            Uniform weights are used by default.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.

        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.  This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree.  The optimal value depends on the
            nature of the problem.

        p : int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric : str or callable, default='minkowski'
            Metric to use for distance computation. Default is "minkowski", which
            results in the standard Euclidean distance when p = 2. See the
            documentation of `scipy.spatial.distance
            <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
            the metrics listed in
            :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
            values.

            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square during fit. X may be a :term:`sparse graph`, in which
            case only "nonzero" elements may be considered neighbors.

            If metric is a callable function, it takes two arrays representing 1D
            vectors as inputs and must return one value indicating the distance
            between those vectors. This works for Scipy's metrics, but is less
            efficient than passing the metric name as a string.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
            Doesn't affect :meth:`fit` method.

        References
        ----------
        Scikit-learn API: sklearn.neighbors.KNeighborsRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )

        self.naming = KNNRegression.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": ["kneighbor"],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f"{self.naming} - automl.log",  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = knn_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        pass


class GradientBoostingRegression(TreeWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Gradient Boosting algorithm to make insightful products."""

    name = "Gradient Boosting"
    special_function = ["Feature Importance Diagram", "Single Tree Diagram"]

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: float = 3,
        min_impurity_decrease: float = 0.0,
        init: Optional[object] = None,
        random_state: Optional[int] = None,
        max_features: Union[str, int, float] = None,
        alpha: float = 0.9,
        verbose: int = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        loss : {'squared_error', 'absolute_error', 'huber', 'quantile'}, \
                default='squared_error'
            Loss function to be optimized. 'squared_error' refers to the squared
            error for regression. 'absolute_error' refers to the absolute error of
            regression and is a robust loss function. 'huber' is a
            combination of the two. 'quantile' allows quantile regression (use
            `alpha` to specify the quantile).

        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by `learning_rate`.
            There is a trade-off between learning_rate and n_estimators.
            Values must be in the range `[0.0, inf)`.

        n_estimators : int, default=100
            The number of boosting stages to perform. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.
            Values must be in the range `[1, inf)`.

        subsample : float, default=1.0
            The fraction of samples to be used for fitting the individual base
            learners. If smaller than 1.0 this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.
            Values must be in the range `(0.0, 1.0]`.

        criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
            The function to measure the quality of a split. Supported criteria are
            "friedman_mse" for the mean squared error with improvement score by
            Friedman, "squared_error" for mean squared error. The default value of
            "friedman_mse" is generally the best as it can provide a better
            approximation in some cases.

            .. versionadded:: 0.18

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, values must be in the range `[2, inf)`.
            - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
            will be `ceil(min_samples_split * n_samples)`.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, values must be in the range `[1, inf)`.
            - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
            will be `ceil(min_samples_leaf * n_samples)`.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.
            Values must be in the range `[0.0, 0.5]`.

        max_depth : int or None, default=3
            Maximum depth of the individual regression estimators. The maximum
            depth limits the number of nodes in the tree. Tune this parameter
            for best performance; the best value depends on the interaction
            of the input variables. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
            If int, values must be in the range `[1, inf)`.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            Values must be in the range `[0.0, inf)`.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        init : estimator or 'zero', default=None
            An estimator object that is used to compute the initial predictions.
            ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
            initial raw predictions are set to zero. By default a
            ``DummyEstimator`` is used, predicting either the average target value
            (for loss='squared_error'), or a quantile for the other losses.

        random_state : int, RandomState instance or None, default=None
            Controls the random seed given to each Tree estimator at each
            boosting iteration.
            In addition, it controls the random permutation of the features at
            each split (see Notes for more details).
            It also controls the random splitting of the training data to obtain a
            validation set if `n_iter_no_change` is not None.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        max_features : {'sqrt', 'log2'}, int or float, default=None
            The number of features to consider when looking for the best split:

            - If int, values must be in the range `[1, inf)`.
            - If float, values must be in the range `(0.0, 1.0]` and the features
            considered at each split will be `max(1, int(max_features * n_features_in_))`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Choosing `max_features < n_features` leads to a reduction of variance
            and an increase in bias.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        alpha : float, default=0.9
            The alpha-quantile of the huber loss function and the quantile
            loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
            Values must be in the range `(0.0, 1.0)`.

        verbose : int, default=0
            Enable verbose output. If 1 then it prints progress and performance
            once in a while (the more trees the lower the frequency). If greater
            than 1 then it prints progress and performance for every tree.
            Values must be in the range `[0, inf)`.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            Values must be in the range `[2, inf)`.
            If None, then unlimited number of leaf nodes.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just erase the
            previous solution. See :term:`the Glossary <warm_start>`.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Values must be in the range `(0.0, 1.0)`.
            Only used if ``n_iter_no_change`` is set to an integer.

            .. versionadded:: 0.20

        n_iter_no_change : int, default=None
            ``n_iter_no_change`` is used to decide if early stopping will be used
            to terminate training when validation score is not improving. By
            default it is set to None to disable early stopping. If set to a
            number, it will set aside ``validation_fraction`` size of the training
            data as validation and terminate training when validation score is not
            improving in all of the previous ``n_iter_no_change`` numbers of
            iterations.
            Values must be in the range `[1, inf)`.

            .. versionadded:: 0.20

        tol : float, default=1e-4
            Tolerance for the early stopping. When the loss is not improving
            by at least tol for ``n_iter_no_change`` iterations (if set to a
            number), the training stops.
            Values must be in the range `[0.0, inf)`.

            .. versionadded:: 0.20

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed.
            Values must be in the range `[0.0, inf)`.
            See :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.ensemble.GradientBoostingRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        """
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

        self.model = GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            init=self.init,
            random_state=self.random_state,
            max_features=self.max_features,
            alpha=self.alpha,
            verbose=self.verbose,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
        )

        self.naming = GradientBoostingRegression.name
        self.customized = True
        self.customized_name = "Gradient Boosting"

    @property
    def settings(self) -> Dict:
        """The configuration of Gradient Boosting to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Gradient Boosting of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.ensemble import GradientBoostingRegressor

        class MyGradientBoostingRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = GradientBoostingRegressor

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "n_estimators": {"domain": tune.lograndint(lower=4, upper=512), "init_value": 100},
                    "max_depth": {"domain": tune.randint(lower=1, upper=10), "init_value": 3},
                    "learning_rate": {"domain": tune.loguniform(lower=0.001, upper=1.0), "init_value": 0.1},
                    "subsample": {"domain": tune.uniform(lower=0.1, upper=1.0), "init_value": 1.0},
                    "min_samples_split": {"domain": tune.randint(lower=2, upper=20), "init_value": 2},
                    "min_samples_leaf": {"domain": tune.randint(lower=1, upper=20), "init_value": 1},
                    "min_impurity_decrease": {"domain": tune.loguniform(lower=1e-10, upper=1e-2), "init_value": 0.0},
                }
                return space

        return MyGradientBoostingRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = gradient_boosting_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=GradientBoostingRegression.X_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0][0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=GradientBoostingRegression.X_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0][0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class LassoRegression(LinearWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Lasso to make insightful products."""

    name = "Lasso Regression"
    special_function = ["Lasso Regression Formula", "2D Scatter Diagram", "3D Scatter Diagram", "2D Line Diagram", "3D Surface Diagram"]

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        precompute: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Optional[int] = None,
        selection: str = "cyclic",
    ) -> None:
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Constant that multiplies the L1 term, controlling regularization
            strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

            When `alpha = 0`, the objective is equivalent to ordinary least
            squares, solved by the :class:`LinearRegression` object. For numerical
            reasons, using `alpha = 0` with the `Lasso` object is not advised.
            Instead, you should use the :class:`LinearRegression` object.

        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).

        precompute : bool or array-like of shape (n_features, n_features),\
                    default=False
            Whether to use a precomputed Gram matrix to speed up
            calculations. The Gram matrix can also be passed as argument.
            For sparse input this option is always ``False`` to preserve sparsity.

        copy_X : bool, default=True
            If ``True``, X will be copied; else, it may be overwritten.

        max_iter : int, default=1000
            The maximum number of iterations.

        tol : float, default=1e-4
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``, see Notes below.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            See :term:`the Glossary <warm_start>`.

        positive : bool, default=False
            When set to ``True``, forces the coefficients to be positive.

        random_state : int, RandomState instance, default=None
            The seed of the pseudo random number generator that selects a random
            feature to update. Used when ``selection`` == 'random'.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        selection : {'cyclic', 'random'}, default='cyclic'
            If set to 'random', a random coefficient is updated every iteration
            rather than looping over features sequentially by default. This
            (setting to 'random') often leads to significantly faster convergence
            especially when tol is higher than 1e-4.

        References
        ----------
        Scikit-learn API: sklearn.linear_model.Lasso
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        """
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

        self.model = Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )

        self.naming = LassoRegression.name
        self.customized = True
        self.customized_name = "Lasso"

    @property
    def settings(self) -> Dict:
        """The configuration of Lasso to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Lasso of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.linear_model import Lasso

        class MyLassoRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = Lasso

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "alpha": {"domain": tune.uniform(lower=0.001, upper=10), "init_value": 1},
                    "fit_intercept": {"domain": tune.choice([True, False])},
                    "max_iter": {"domain": tune.randint(lower=500, upper=2000), "init_value": 1000},
                    "tol": {"domain": tune.uniform(lower=1e-5, upper=1e-3), "init_value": 1e-4},
                    "selection": {"domain": tune.choice(["cyclic", "random"])},
                }
                return space

        return MyLassoRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = lasso_regression_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.model.coef_],
            intercept=self.model.intercept_,
            features_name=LassoRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = LassoRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                y_test_predict=LassoRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                y_test_predict=LassoRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.auto_model.coef_],
            intercept=self.auto_model.intercept_,
            features_name=LassoRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = LassoRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LassoRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                y_test_predict=LassoRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=LassoRegression.X_test,
                target_data=LassoRegression.y_test,
                y_test_predict=LassoRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass


class ElasticNetRegression(LinearWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Elastic Net algorithm to make insightful products."""

    name = "Elastic Net"
    special_function = ["Elastic Net Formula", "2D Scatter Diagram", "3D Scatter Diagram", "2D Line Diagram", "3D Surface Diagram"]

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        precompute: bool = False,
        max_iter: int = 1000,
        copy_X: bool = True,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Optional[int] = None,
        selection: str = "cyclic",
    ) -> None:
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Constant that multiplies the penalty terms. Defaults to 1.0.
            See the notes for the exact mathematical meaning of this
            parameter. ``alpha = 0`` is equivalent to an ordinary least square,
            solved by the :class:`LinearRegression` object. For numerical
            reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
            Given this, you should use the :class:`LinearRegression` object.

        l1_ratio : float, default=0.5
            The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
            ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
            is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
            combination of L1 and L2.

        fit_intercept : bool, default=True
            Whether the intercept should be estimated or not. If ``False``, the
            data is assumed to be already centered.

        precompute : bool or array-like of shape (n_features, n_features),\
                    default=False
            Whether to use a precomputed Gram matrix to speed up
            calculations. The Gram matrix can also be passed as argument.
            For sparse input this option is always ``False`` to preserve sparsity.

        max_iter : int, default=1000
            The maximum number of iterations.

        copy_X : bool, default=True
            If ``True``, X will be copied; else, it may be overwritten.

        tol : float, default=1e-4
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``, see Notes below.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            See :term:`the Glossary <warm_start>`.

        positive : bool, default=False
            When set to ``True``, forces the coefficients to be positive.

        random_state : int, RandomState instance, default=None
            The seed of the pseudo random number generator that selects a random
            feature to update. Used when ``selection`` == 'random'.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        selection : {'cyclic', 'random'}, default='cyclic'
            If set to 'random', a random coefficient is updated every iteration
            rather than looping over features sequentially by default. This
            (setting to 'random') often leads to significantly faster convergence
            especially when tol is higher than 1e-4.

        References
        ----------
        Scikit-learn API: sklearn.linear_model.ElasticNet
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        """
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            max_iter=self.max_iter,
            copy_X=self.copy_X,
            tol=self.tol,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )

        self.naming = ElasticNetRegression.name
        self.customized = True
        self.customized_name = "Elastic Net"

    @property
    def settings(self) -> Dict:
        """The configuration of Elastic Net to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Elastic Net of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.linear_model import ElasticNet

        class MyElasticNetRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = ElasticNet

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "alpha": {"domain": tune.uniform(lower=0.001, upper=10), "init_value": 1},
                    "l1_ratio": {"domain": tune.uniform(lower=0.001, upper=1), "init_value": 0.5},
                    "fit_intercept": {"domain": tune.choice([True, False])},
                    "max_iter": {"domain": tune.randint(lower=500, upper=2000), "init_value": 1000},
                    "tol": {"domain": tune.uniform(lower=1e-5, upper=1e-3), "init_value": 1e-4},
                    "selection": {"domain": tune.choice(["cyclic", "random"])},
                }
                return space

        return MyElasticNetRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = elastic_net_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.model.coef_],
            intercept=self.model.intercept_,
            features_name=ElasticNetRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = ElasticNetRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                y_test_predict=ElasticNetRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                y_test_predict=ElasticNetRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.auto_model.coef_],
            intercept=self.auto_model.intercept_,
            features_name=ElasticNetRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = ElasticNetRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(ElasticNetRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                y_test_predict=ElasticNetRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=ElasticNetRegression.X_test,
                target_data=ElasticNetRegression.y_test,
                y_test_predict=ElasticNetRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass


class SGDRegression(LinearWorkflowMixin, RegressionWorkflowBase):
    """The automation workflow of using Stochastic Gradient Descent - SGD algorithm to make insightful products."""

    name = "SGD Regression"
    special_function = ["SGD Regression Formula", "2D Scatter Diagram", "3D Scatter Diagram", "2D Line Diagram", "3D Surface Diagram"]

    def __init__(
        self,
        loss: str = "squared_error",
        penalty: str = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: Union[float, None] = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        random_state: Optional[int] = None,
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        power_t: float = 0.25,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        warm_start: bool = False,
        average: Union[bool, int] = False,
    ) -> None:
        """
        Parameters
        ----------
        loss : str, default='squared_error'
            The loss function to be used. The possible values are 'squared_error',
            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
            The 'squared_error' refers to the ordinary least squares fit.
            'huber' modifies 'squared_error' to focus less on getting outliers
            correct by switching from squared to linear loss past a distance of
            epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
            linear past that; this is the loss function used in SVR.
            'squared_epsilon_insensitive' is the same but becomes squared loss past
            a tolerance of epsilon.
            More details about the losses formulas can be found in the
            :ref:`User Guide <sgd_mathematical_formulation>`.

        penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
            The penalty (aka regularization term) to be used. Defaults to 'l2'
            which is the standard regularizer for linear SVM models. 'l1' and
            'elasticnet' might bring sparsity to the model (feature selection)
            not achievable with 'l2'. No penalty is added when set to `None`.

        alpha : float, default=0.0001
            Constant that multiplies the regularization term. The higher the
            value, the stronger the regularization.
            Also used to compute the learning rate when set to `learning_rate` is
            set to 'optimal'.

        l1_ratio : float, default=0.15
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
            Only used if `penalty` is 'elasticnet'.

        fit_intercept : bool, default=True
            Whether the intercept should be estimated or not. If False, the
            data is assumed to be already centered.

        max_iter : int, default=1000
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the ``fit`` method, and not the
            :meth:`partial_fit` method.
            .. versionadded:: 0.19

        tol : float or None, default=1e-3
            The stopping criterion. If it is not None, training will stop
            when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
            epochs.
            Convergence is checked against the training loss or the
            validation loss depending on the `early_stopping` parameter.
            .. versionadded:: 0.19

        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch.

        verbose : int, default=0
            The verbosity level.

        epsilon : float, default=0.1
            Epsilon in the epsilon-insensitive loss functions; only if `loss` is
            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
            For 'huber', determines the threshold at which it becomes less
            important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current prediction
            and the correct label are ignored if they are less than this threshold.

        random_state : int, RandomState instance, default=None
            Used for shuffling the data, when ``shuffle`` is set to ``True``.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        learning_rate : str, default='invscaling'
            The learning rate schedule:
            - 'constant': `eta = eta0`
            - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
            where t0 is chosen by a heuristic proposed by Leon Bottou.
            - 'invscaling': `eta = eta0 / pow(t, power_t)`
            - 'adaptive': eta = eta0, as long as the training keeps decreasing.
            Each time n_iter_no_change consecutive epochs fail to decrease the
            training loss by tol or fail to increase validation score by tol if
            early_stopping is True, the current learning rate is divided by 5.
                .. versionadded:: 0.20
                    Added 'adaptive' option

        eta0 : float, default=0.01
            The initial learning rate for the 'constant', 'invscaling' or
            'adaptive' schedules. The default value is 0.01.

        power_t : float, default=0.25
            The exponent for inverse scaling learning rate.

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation
            score is not improving. If set to True, it will automatically set aside
            a fraction of training data as validation and terminate
            training when validation score returned by the `score` method is not
            improving by at least `tol` for `n_iter_no_change` consecutive
            epochs.
            .. versionadded:: 0.20
                Added 'early_stopping' option

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
            Only used if `early_stopping` is True.
            .. versionadded:: 0.20
                Added 'validation_fraction' option

        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before stopping
            fitting.
            Convergence is checked against the training loss or the
            validation loss depending on the `early_stopping` parameter.
            .. versionadded:: 0.20
                Added 'n_iter_no_change' option

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            See :term:`the Glossary <warm_start>`.
            Repeatedly calling fit or partial_fit when warm_start is True can
            result in a different solution than when calling fit a single time
            because of the way the data is shuffled.
            If a dynamic learning rate is used, the learning rate is adapted
            depending on the number of samples already seen. Calling ``fit`` resets
            this counter, while ``partial_fit``  will result in increasing the
            existing counter.

        average : bool or int, default=False
            When set to True, computes the averaged SGD weights across all
            updates and stores the result in the ``coef_`` attribute. If set to
            an int greater than 1, averaging will begin once the total number of
            samples seen reaches `average`. So ``average=10`` will begin
            averaging after seeing 10 samples.

        References
        ----------
        Scikit-learn API: sklearn.linear_model.SGDRegressor
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
        """
        super().__init__()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average

        self.model = SGDRegressor(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            warm_start=self.warm_start,
            average=self.average,
        )

        self.naming = SGDRegression.name
        self.customized = True
        self.customized_name = "SGD Regression"

    @property
    def settings(self) -> Dict:
        """The configuration of SGD to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "r2",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "regression",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SGD of FLAML framework."""
        from flaml import tune
        from flaml.data import REGRESSION
        from flaml.model import SKLearnEstimator
        from sklearn.linear_model import SGDRegressor

        class MySGDRegression(SKLearnEstimator):
            def __init__(self, task="regression", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = SGDRegressor

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "loss": {"domain": tune.choice(["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]), "init_value": "squared_error"},
                    "penalty": {"domain": tune.choice(["l2", "l1", "elasticnet", None]), "init_value": "l2"},
                    "alpha": {"domain": tune.loguniform(lower=0.0001, upper=1), "init_value": 0.0001},
                    "fit_intercept": {"domain": tune.choice([True, False]), "init_value": True},
                    "max_iter": {"domain": tune.randint(lower=50, upper=1000), "init_value": 1000},
                    "tol": {"domain": tune.loguniform(lower=0.000001, upper=0.001), "init_value": 0.001},
                    "shuffle": {"domain": tune.choice([True, False]), "init_value": True},
                    "learning_rate": {"domain": tune.choice(["constant", "optimal", "invscaling", "adaptive"]), "init_value": "invscaling"},
                    "eta0": {"domain": tune.loguniform(lower=0.0001, upper=0.1), "init_value": 0.01},
                    "power_t": {"domain": tune.uniform(lower=0.1, upper=0.9), "init_value": 0.25},
                    "l1_ratio": {"domain": tune.uniform(lower=0, upper=1), "init_value": 0.15},
                }
                return space

        return MySGDRegression

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
        hyper_parameters = sgd_regression_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.model.coef_],
            intercept=self.model.intercept_,
            features_name=SGDRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = SGDRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                y_test_predict=SGDRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                y_test_predict=SGDRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=[self.auto_model.coef_],
            intercept=self.auto_model.intercept_,
            features_name=SGDRegression.X_train.columns,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        columns_num = SGDRegression.X.shape[1]
        if columns_num > 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 2)
            self._plot_3d_scatter_diagram(
                feature_data=three_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(SGDRegression.X_test, 1)
            self._plot_2d_scatter_diagram(
                feature_data=two_dimen_data,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            # no need to choose
            self._plot_3d_scatter_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_3d_surface_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                y_test_predict=SGDRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_scatter_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
            self._plot_2d_line_diagram(
                feature_data=SGDRegression.X_test,
                target_data=SGDRegression.y_test,
                y_test_predict=SGDRegression.y_test_predict,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )
        else:
            pass
