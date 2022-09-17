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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from ._base import WorkflowBase
# sys.path.append("..")


class RegressionWorkflowBase(WorkflowBase):
    def __init__(self) -> None:
        super().__init__()
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


    @staticmethod
    def data_split(X, y, test_size=0.2, random_state=42):
        RegressionWorkflowBase.X = X  # child class is able to access to the data
        RegressionWorkflowBase.y = y
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

class SupportVectorRegression(RegressionWorkflowBase, BaseEstimator):
    name = "Support Vector Machine"
    special_function = ["Plot SVR Regression"]

    def __init__(self,
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
                 random_state: int = 42):
        super().__init__(random_state=42)
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

        self.model = SVR(
            kernel=self.kernel,
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

    def Plot_SVR_Regression(self):
        y = RegressionWorkflowBase().y
        X = RegressionWorkflowBase().X
        clf = self.model.fit(X, y)
        X_train, X_test, y_train, y_test = self.data_split(X, y)
        y_test_prediction = self.predict(X_test)
        y_test = np.array(y_test).reshape(1, len(y_test)).flatten()
        y_test_prediction = y_test_prediction.flatten()

        line_a = y_test.min()
        line_b = y_test.max()
        # the lien between a and b.
        plt.plot([line_a, line_b], [line_a, line_b], '-r', linewidth=1)
        plt.plot(y_test, y_test_prediction, 'o', color='gold', alpha=0.3)
        save_fig('Plot_SVR_Regression', MODEL_OUTPUT_IMAGE_PATH)


    def special_components(self):
        self.Plot_SVR_Regression()
        pass



class DNNRegression(RegressionWorkflowBase, BaseEstimator):

    name = "Deep Neural Networks"
    special_function = ["Loss Record"]

    def __init__(
            self,
            hidden_layer_sizes: tuple = (9, 9),
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
        super().__init__(random_state)
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

    def plot_learning_curve(self):
        pd.DataFrame(self.model.loss_curve_).plot(title="Loss")
        plt.show()
        save_fig("DNN_loss_record", MODEL_OUTPUT_IMAGE_PATH)

    def plot_pred(self):
        y_test_prediction = self.model.predict(RegressionWorkflowBase.X_test)
        y_test = np.array(RegressionWorkflowBase.y_test).reshape(1, len(RegressionWorkflowBase.y_test)).flatten()
        y_test_prediction = y_test_prediction.flatten()
        plt.plot([i for i in range(len(y_test))], y_test, label='true')
        plt.plot([i for i in range(len(y_test))], y_test_prediction, label='predict')
        plt.legend()
        plt.title('Ground Truth v.s. Prediction')
        plt.show()
        save_fig("DNN_predict", MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.plot_learning_curve()
        self.plot_pred()


