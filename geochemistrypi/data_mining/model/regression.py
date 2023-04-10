# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from typing import Sequence
import numpy as np
import pandas as pd
import xgboost
from multipledispatch import dispatch
from flaml import AutoML

from ..global_variable import MODEL_OUTPUT_IMAGE_PATH, RAY_FLAML
from ..utils.base import save_fig
from ._base import WorkflowBase
from .func.algo_regression._common import plot_predicted_value_evaluation, plot_true_vs_predicted, score, cross_validation
from .func.algo_regression._polynomial_regression import show_formula, polynomial_regression_manual_hyper_parameters
from .func.algo_regression._rf import feature_importance__, box_plot, random_forest_manual_hyper_parameters
from .func.algo_regression._xgboost import feature_importance, histograms_feature_weights, permutation_importance_, xgboost_manual_hyper_parameters
from .func.algo_regression._linear_regression import show_formula, plot_2d_graph, plot_3d_graph, linear_regression_manual_hyper_parameters
from .func.algo_regression._extra_tree import feature_importances, extra_trees_manual_hyper_parameters
from .func.algo_regression._svr import plot_2d_decision_boundary, svr_manual_hyper_parameters
from .func.algo_regression._decision_tree import decision_tree_plot, decision_tree_manual_hyper_parameters
from .func.algo_regression._deep_neural_network import deep_neural_network_manual_hyper_parameters


class RegressionWorkflowBase(WorkflowBase):
    """The base workflow class of regression algorithms."""

    common_function = ['Model Score', 'Cross Validation', 'Model Prediction', 'Model Persistence']

    def __init__(self) -> None:
        super().__init__()
        # These two attributes are used for the customized models of FLAML framework
        self.customized = False
        self.customized_name = None

    @dispatch(object, object)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model by Scikit-learn framework."""
        self.model.fit(X, y)

    @dispatch(object, object, bool)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, is_automl: bool = False) -> None:
        """Fit the model by FLAML framework and RAY framework."""
        if self.naming not in RAY_FLAML:
            self.automl = AutoML()
            if self.customized:  # When the model is not built-in in FLAML framwork, use FLAML customization.
                self.automl.add_learner(learner_name=self.customized_name, learner_class=self.customization)
            if y.shape[1] == 1:  # FLAML's data format validation mechanism
                y = y.squeeze()  # Convert a single dataFrame column into a series
            self.automl.fit(X_train=X, y_train=y, **self.settings)
        else:
            # When the model is not built-in in FLAML framework, use RAY + FLAML customization.
            self.ray_tune(RegressionWorkflowBase.X_train, RegressionWorkflowBase.X_test,
                          RegressionWorkflowBase.y_train, RegressionWorkflowBase.y_test)
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

    def ray_tune(self, X_train, X_test, y_train, y_test) -> object:
        """The customized model of FLAML framework and RAY framework."""
        return object
    
    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    @staticmethod
    def _plot_predicted_value_evaluation(y_test: pd.DataFrame, y_test_predict: pd.DataFrame, algorithm_name: str, store_path: str) -> None:
        """Plot the predicted value evaluation."""
        print("-----* Predicted Value Evaluation *-----")
        plot_predicted_value_evaluation(y_test, y_test_predict)
        save_fig(f'Predicted Value Evaluation - {algorithm_name}', store_path)

    @staticmethod
    def _plot_true_vs_predicted(y_test_predict: pd.DataFrame, y_test: pd.DataFrame, algorithm_name: str, store_path: str) -> None:
        """Plot the true value vs. predicted value."""
        print("-----* True Value vs. Predicted Value *-----")
        plot_true_vs_predicted(y_test_predict, y_test, algorithm_name)
        save_fig(f'True Value vs. Predicted Value - {algorithm_name}', store_path)

    @staticmethod
    def _score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> None:
        """Calculate the score of the model."""
        print("-----* Model Score *-----")
        score(y_true, y_predict)

    @staticmethod
    def _cross_validation(trained_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, cv_num: int = 10) -> None:
        """Cross validation."""
        print("-----* Cross Validation *-----")
        print(f"K-Folds: {cv_num}")
        cross_validation(trained_model, X_train, y_train, cv_num=cv_num)

    # TODO(Sany sanyhew1097618435@163.com): How to prevent overfitting
    def is_overfitting(self):
        pass

    @dispatch()
    def common_components(self) -> None:
        """Invoke all common application functions for classification algorithms by Scikit-learn framework."""
        self._score(RegressionWorkflowBase.y_test, RegressionWorkflowBase.y_test_predict)
        self._cross_validation(self.model, RegressionWorkflowBase.X_train, RegressionWorkflowBase.y_train, 10)
        self._plot_predicted_value_evaluation(RegressionWorkflowBase.y_test, RegressionWorkflowBase.y_test_predict,
                                              self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._plot_true_vs_predicted(y_test_predict=RegressionWorkflowBase.y_test_predict,
                                     y_test=RegressionWorkflowBase.y_test, algorithm_name=self.naming,
                                     store_path=MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def common_components(self, is_automl: bool) -> None:
        """Invoke all common application functions for classification algorithms by FLAML framework."""
        self._score(RegressionWorkflowBase.y_test, RegressionWorkflowBase.y_test_predict)
        self._cross_validation(self.auto_model, RegressionWorkflowBase.X_train, RegressionWorkflowBase.y_train, 10)
        self._plot_predicted_value_evaluation(RegressionWorkflowBase.y_test, RegressionWorkflowBase.y_test_predict,
                                              self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._plot_true_vs_predicted(y_test_predict=RegressionWorkflowBase.y_test_predict,
                                     y_test=RegressionWorkflowBase.y_test, algorithm_name=self.naming,
                                     store_path=MODEL_OUTPUT_IMAGE_PATH)


class PolynomialRegression(RegressionWorkflowBase):
    """The automation workflow of using Polynomial Regression algorithm to make insightful products."""

    name = "Polynomial Regression"
    special_function = ["Polynomial Regression Formula"]

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        order: str = 'C',
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: Optional[int] = None
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

        self.model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs
        )

        self._features_name = None
        self.naming = PolynomialRegression.name

    def poly(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Polynomial features."""
        poly_features = PolynomialFeatures(degree=self.degree,
                                           include_bias=self.include_bias,
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
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = polynomial_regression_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _show_formula(coef: np.ndarray, intercept: np.ndarray, features_name: List) -> None:
        """Show the formula of the polynomial regression."""
        print("-----* Polynomial Regression Formula *-----")
        show_formula(coef, intercept, features_name)

    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._show_formula(coef=self.model.coef_, intercept=self.model.intercept_, features_name=self._features_name)


class XgboostRegression(RegressionWorkflowBase):
    """The automation workflow of using Xgboost algorithm to make insightful products."""

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
        max_depth: Optional[int] = 6,
        learning_rate: Optional[float] = 0.3,
        n_estimators: int = 100,
        verbosity: Optional[int] = 1,
        objective: _SklObjective = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = 'auto',
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
        importance_type: Optional[str] = 'gain',
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        #enable_categorical: bool = False,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any
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
                    For larger dataset, approximate algorithm (approx) will be chosen. It’s recommended to try hist and gpu_hist for higher performance with large dataset. (gpu_hist)has support for external memory.
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
        #self.enable_categorical = enable_categorical
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
            early_stopping_rounds=self.early_stopping_rounds
        )

        self.naming = XgboostRegression.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'r2',
            "estimator_list": ['xgboost'],  # list of ML learners
            "task": 'regression',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = xgboost_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _feature_importance(X: pd.DataFrame, trained_model: object, image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Feature importance plot."""
        print("-----* Feature Importance *-----")
        feature_importance(X, trained_model, image_config)
        save_fig(f"Feature Importance - {algorithm_name}", store_path)

    @staticmethod
    def _histograms_feature_weights(X: pd.DataFrame, trained_model: object, image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Histograms of feature weights plot."""
        histograms_feature_weights(X, trained_model,image_config)
        save_fig(f"Regression - {algorithm_name} - Feature Importance Score", store_path)

    @staticmethod
    def _permutation_importance(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object,
                                image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Permutation importance plot."""
        permutation_importance_(X, X_test, y_test, trained_model, image_config)
        save_fig(f"Regression - {algorithm_name} - Xgboost Feature Importance", store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importance(XgboostRegression.X, self.model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._histograms_feature_weights(XgboostRegression.X, self.model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._permutation_importance(XgboostRegression.X, XgboostRegression.X_test, XgboostRegression.y_test,
                                     self.model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importance(XgboostRegression.X, self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._histograms_feature_weights(XgboostRegression.X, self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._permutation_importance(XgboostRegression.X, XgboostRegression.X_test, XgboostRegression.y_test,
                                     self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class DecisionTreeRegression(RegressionWorkflowBase):
    """The automation workflow of using Decision Tree algorithm to make insightful products."""

    name = "Decision Tree"
    special_function = ["Decision Tree Plot"]

    def __init__(
        self,
        criterion: str = 'squared_error',
        splitter: str = 'best',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0
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
        self.criterion = criterion,
        self.splitter = splitter,
        # FIXME (Sany sanyhew1097618435@163.com): figure out why data type changes after assignment.
        # print("max_depth before: ", max_depth)
        # print("max_depth before type: ", type(max_depth))
        self.max_depth = max_depth,
        # print("max_depth after: ", self.max_depth)
        # print("max_depth after type: ", type(self.max_depth))
        self.min_samples_split = min_samples_split,
        self.min_samples_leaf = min_samples_leaf,
        self.min_weight_fraction_leaf = min_weight_fraction_leaf,
        self.max_features = max_features,
        self.random_state = random_state,
        self.max_leaf_nodes = max_leaf_nodes,
        self.min_impurity_decrease = min_impurity_decrease,
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
            ccp_alpha=self.ccp_alpha
        )
        self.naming = DecisionTreeRegression.name
        self.customized = True
        self.customized_name = 'Decision Tree'

    @property
    def settings(self) -> Dict:
        """The configuration of Decision Tree to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'r2',
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": 'regression',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration
    
    @property
    def customization(self) -> object:
        """The customized Decision Tree of FLAML framework."""
        from flaml.model import SKLearnEstimator
        from flaml import tune
        from flaml.data import REGRESSION
        from sklearn.tree import DecisionTreeRegressor

        class MyDTRegression(SKLearnEstimator):
            def __init__(self, task='regression', n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = DecisionTreeRegressor

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    'criterion': {'domain': tune.choice(["squared_error", "friedman_mse", "absolute_error", "poisson"])},
                    'max_depth': {'domain': tune.randint(lower=2, upper=20),
                                  'init_value': 1,
                                  'low_cost_init_value': 1},
                    'min_samples_split': {'domain': tune.randint(lower=2, upper=10),
                                          'init_value': 2,
                                          'low_cost_init_value': 2},
                    'min_samples_leaf': {'domain': tune.randint(lower=1, upper=10),
                                         'init_value': 1,
                                         'low_cost_init_value': 1},
                    'max_features': {'domain': tune.randint(lower=1, upper=10),
                                     'init_value': 1,
                                     'low_cost_init_value': 1},
                }
                return space

        return MyDTRegression

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = decision_tree_manual_hyper_parameters()
        return hyperparameters

    def _plot_tree_function(self, trained_model: object, image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Drawing decision tree diagrams."""
        print("-----* Decision Tree Plot *-----")
        decision_tree_plot(trained_model, image_config)
        save_fig(f"Regression - {algorithm_name} - Tree Graph", store_path)

    @dispatch()
    def special_components(self):
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._plot_tree_function(self.model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._plot_tree_function(self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        

class ExtraTreesRegression(RegressionWorkflowBase):
    """The automation workflow of using Extra-Trees algorithm to make insightful products."""
    
    name = "Extra-Trees"
    special_function = ["Feature Importance"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.,
        max_features: Union[int, float, str] = "auto",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.,
    #  min_impurity_split=None,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha: float = 0.0,
        max_samples=None
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
            "metric": 'r2',
            "estimator_list": ['extra_tree'],  # list of ML learners
            "task": 'regression',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = extra_trees_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _feature_importances(X_train: pd.DataFrame, trained_model: object,  image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Draw the feature importance bar diagram."""
        print("-----* Feature Importance *-----")
        feature_importances(X_train, trained_model, image_config)
        save_fig(f"Regression - {algorithm_name} - Feature Importance Plot", store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importances(ExtraTreesRegression.X_train, self.model,  self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importances(ExtraTreesRegression.X_train, self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class RandomForestRegression(RegressionWorkflowBase):
    """The automation workflow of using Random Forest algorithm to make insightful products."""

    name = "Random Forest"
    special_function = ["Feature Importance"]

    def __init__(   
        self,
        n_estimators=100,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        # class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
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
        self.max_samples=max_samples

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
            "metric": 'r2',
            "estimator_list": ['rf'],  # list of ML learners
            "task": 'regression',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = random_forest_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _feature_importances(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                             trained_model: object,  image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Feature importance plot."""
        print("-----* Feature Importance *-----")
        feature_importance__(X, X_test, y_test, trained_model, image_config)
        save_fig(f"Regression - {algorithm_name} - Feature Importance", store_path)

    @staticmethod
    def _box_plot(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 trained_model: object,  image_config: dict, algorithm_name: str, store_path: str) -> None:
        """Box plot."""
        print("-----* Box Plot *-----")
        box_plot(X, X_test, y_test, trained_model, image_config)
        save_fig(f"Regression - {algorithm_name} - Box Plot", store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importances(RandomForestRegression.X, RandomForestRegression.X_test,
                                  RandomForestRegression.y_test, self.model, self.image_config, self.naming,
                                  MODEL_OUTPUT_IMAGE_PATH)
        self._box_plot(RandomForestRegression.X, RandomForestRegression.X_test,
                                  RandomForestRegression.y_test, self.model, self.image_config, self.naming,
                                  MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importances(RandomForestRegression.X, RandomForestRegression.X_test,
                                  RandomForestRegression.y_test, self.auto_model, self.image_config, self.naming,
                                  MODEL_OUTPUT_IMAGE_PATH)
        self._box_plot(RandomForestRegression.X, RandomForestRegression.X_test,
                                  RandomForestRegression.y_test, self.auto_model, self.image_config, self.naming,
                                  MODEL_OUTPUT_IMAGE_PATH)


class SVMRegression(RegressionWorkflowBase):
    """The automation workflow of using SVR algorithm to make insightful products."""

    name = "Support Vector Machine"
    special_function = ['Two-dimensional Decision Boundary Diagram']

    def __init__(
        self,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: Union[str, float] = 'scale',
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
            max_iter=self.max_iter
        )

        self.naming = SVMRegression.name
        self.customized = True
        self.customized_name = 'SVR'

    @property
    def settings(self) -> Dict:
        """The configuration of SVR to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'r2',
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": 'regression',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SVR of FLAML framework."""
        from flaml.model import SKLearnEstimator
        from flaml import tune
        from flaml.data import REGRESSION
        from sklearn.svm import SVR

        class MySVMRegression(SKLearnEstimator):
            def __init__(self, task='regression', n_jobs=None, **config):
                super().__init__(task, **config)
                if task in REGRESSION:
                    self.estimator_class = SVR

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    'C': {
                        'domain': tune.uniform(lower=1, upper=data_size[0]),
                        'init_value': 1,
                        'low_cost_init_value': 1
                    },
                    'kernel': {'domain': tune.choice(['poly', 'rbf', 'sigmoid'])},
                    'gamma': {
                        'domain': tune.uniform(lower=1e-5, upper=10),
                        'init_value': 1e-1,
                        'low_cost_init_value': 1e-1
                    },
                    'degree': {
                        'domain': tune.quniform(lower=1, upper=5, q=1),
                        'init_value': 3,
                        'low_cost_init_value': 3
                    },
                    'coef0': {
                        'domain': tune.uniform(lower=0, upper=1),
                        'init_value': 0,
                        'low_cost_init_value': 0
                    },
                    'shrinking': {'domain': tune.choice([True, False])},
                }
                return space

        return MySVMRegression

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = svr_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: Any,
                                   image_config: dict, algorithm_name: str, store_path: str,
                                   contour_data: Optional[List[np.ndarray]] = None,
                                   labels: Optional[np.ndarray] = None) -> None:
        """Plot the decision boundary of the trained model with the testing data set below."""
        print("-----* Two-dimensional Decision Boundary Diagram *-----")
        plot_2d_decision_boundary(X, X_test, y_test, trained_model, image_config, algorithm_name)
        save_fig(f'Regression - {algorithm_name} - Decision Boundary', store_path)

    @dispatch()
    def special_components(self, **kwargs):
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""

        if SVMRegression.X.shape[1] == 2:
            self._plot_2d_decision_boundary(SVMRegression.X, SVMRegression.X_test, SVMRegression.y_test,
                                            self.model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""

        if SVMRegression.X.shape[1] == 2:
            self._plot_2d_decision_boundary(SVMRegression.X, SVMRegression.X_test, SVMRegression.y_test,
                                            self.auto_model, self.image_config, self.naming, MODEL_OUTPUT_IMAGE_PATH)



class DNNRegression(RegressionWorkflowBase):
    """The automation workflow of using Deep Neural Network algorithm to make insightful products."""

    name = "Deep Neural Network"
    special_function = ["Loss Record"]

    def __init__(
        self,
        hidden_layer_sizes: tuple = (50, 25, 5),
        activation: str = 'relu',
        solver: str ='adam',
        alpha: float = 0.0001,
        batch_size: Union[int, str] = 'auto',
        learning_rate: str = 'constant',
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
            n_iter_no_change=self.n_iter_no_change
        )

        self.naming = DNNRegression.name

    def ray_tune(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """The customized DNN of the combinations of Ray, FLAML and Scikit-learn framework."""

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
            "batch": tune.randint(20, 100)
        }

        # Define the time budget in seconds.
        time_budget_s = 30

        # Integrate with FLAML's BlendSearch to implement hyper-parameters optimization .
        algo = BlendSearch(
            metric="mean_loss",
            mode="min",
            space=search_config)
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
        best_result = results.get_best_result(metric='mean_loss', mode='min')
        self.ray_best_model = customized_model(best_result.config['l1'], best_result.config['l2'],
                                               best_result.config['l3'], best_result.config['batch'])

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = deep_neural_network_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _plot_learning_curve(trained_model: object, algorithm_name: str, store_path) -> None:
        """Plot the learning curve of the trained model."""
        print("-----* Loss Record *-----")
        pd.DataFrame(trained_model.loss_curve_).plot(title="Loss")
        save_fig(f'Loss Record - {algorithm_name}', store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._plot_learning_curve(self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._plot_learning_curve(self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class LinearRegression2(RegressionWorkflowBase):
    """The automation workflow of using Linear Regression algorithm to make insightful products."""

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
        Scikit-learn API: sklearn.linear_model.LinearRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression
        """
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            normalize=self.normalize,
            n_jobs=self.n_jobs
        )

        self.naming = LinearRegression2.name

    @staticmethod
    def manual_hyper_parameters() -> Dict:
        """Manual hyper-parameters specification."""
        print("-*-*- Hyper-parameters Specification -*-*-")
        hyperparameters = linear_regression_manual_hyper_parameters()
        return hyperparameters

    @staticmethod
    def _show_formula(coef, intercept, columns_name):
        """Show the formula of the linear regression model."""
        print("-----* Linear Regression Formula *-----")
        show_formula(coef, intercept, columns_name)

    @staticmethod
    def _plot_2d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str,
                       store_path: str):
        """Plot the 2D graph of the linear regression model."""
        print("-----* Plot 2D Graph *-----")
        plot_2d_graph(feature_data, target_data)
        save_fig(f"2D Scatter Graph - {algorithm_name}", store_path)

    @staticmethod
    def _plot_3d_graph(feature_data: pd.DataFrame, target_data: pd.DataFrame, algorithm_name: str,
                       store_path: str):
        """Plot the 3D graph of the linear regression model."""
        print("-----* Plot 3D Graph *-----")
        plot_3d_graph(feature_data, target_data)
        save_fig(f"3D Scatter Graph - {algorithm_name}", store_path)

    def special_components(self, **kwargs):
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._show_formula(coef=self.model.coef_, intercept=self.model.intercept_,
                           columns_name=LinearRegression2.X.columns)

        columns_num = LinearRegression2.X.shape[1]
        if columns_num > 2:
            # choose two of dimensions to draw
            three_dimen_axis_index, three_dimen_data = self.choose_dimension_data(LinearRegression2.X, 2)
            self._plot_3d_graph(feature_data=three_dimen_data, target_data=LinearRegression2.y,
                                algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LinearRegression2.X, 1)
            self._plot_2d_graph(feature_data=two_dimen_data, target_data=LinearRegression2.y,
                                algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        elif columns_num == 2:
            # choose one of dimensions to draw
            two_dimen_axis_index, two_dimen_data = self.choose_dimension_data(LinearRegression2.X, 1)
            self._plot_2d_graph(feature_data=two_dimen_data, target_data=LinearRegression2.y,
                                algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
            # no need to choose
            self._plot_3d_graph(feature_data=LinearRegression2.X, target_data=LinearRegression2.y,
                                algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        elif columns_num == 1:
            # no need to choose
            self._plot_2d_graph(feature_data=LinearRegression2.X, target_data=LinearRegression2.y,
                                algorithm_name=self.naming, store_path=MODEL_OUTPUT_IMAGE_PATH)
        else:
            pass
