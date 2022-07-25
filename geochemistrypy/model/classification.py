# -*- coding: utf-8 -*-
# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, Sequence
# sys.path.append("..")


class ClassificationWorkflowBase(object):

    X = None
    y = None
    name = None
    common_function = ["Model Score", "Confusion Matrix"]
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
        self.naming = None

    @staticmethod
    def data_split(X_data, y_data, test_size=0.2, random_state=42):
        ClassificationWorkflowBase.X = X_data
        ClassificationWorkflowBase.y = y_data
        X_train, X_test, y_train, y_test = train_test_split(ClassificationWorkflowBase.X,
                                                            ClassificationWorkflowBase.y,
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
        print("-----* Model Score *-----")
        print(classification_report(y_test, y_test_prediction))

    def confusion_matrix_plot(self, X_test, y_test, y_test_prediction):
        print("-----* Confusion Matrix *-----")
        print(confusion_matrix(y_test, y_test_prediction))
        plot_confusion_matrix(self.model, X_test, y_test)
        # plt.show()
        save_fig(f"Confusion Matrix - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)


class SVMClassification(ClassificationWorkflowBase):

    name = "Support Vector Machine"
    special_function = ['Two-dimensional Decision Boundary Diagram']

    def __init__(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None
    ):
        ##############################################################
        #Support vector machine is used to classify the data
        ##############################################################
        """
        :param C:float, default=1.0 Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
        :param kernel:Specifies the kernel type to be used in the algorithm
        :param degree:Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        :param gamma:Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        :param coef0:Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        :param shrinking:Whether to use the shrinking heuristic. See the User Guide
        :param probability:Whether to enable probability estimates. This must be enabled prior to calling , will slow down that method as it internally uses 5-fold cross-validation, and may be inconsistent with .
        :param tol:Whether to enable probability estimates. This must be enabled prior to calling , will slow down that method as it internally uses 5-fold cross-validation, and may be inconsistent with .
        :param cache_size:Specify the size of the kernel cache (in MB).
        :param class_weight:Set the parameter C of class i to class_weight[i]*C for SVC.
        :param verbose:Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
        :param max_iter:Hard limit on iterations within solver, or -1 for no limit.
        :param decision_function_shape:Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, note that internally, one-vs-one (‘ovo’) is always used as a multi-class strategy to train models; an ovr matrix is only constructed from the ovo matrix. The parameter is ignored for binary classification.
        :param break_ties:If true, , and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict
        :param random_state:Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when is False. Pass an int for reproducible output across multiple function calls. See Glossary.

        References
        ----------------------------------------
        API design for machine learning software: experiences from the scikit-learn project.Buitinck, LarLouppe, GillesBlondel, MathieuPedregosa, FabianMueller, AndreasGrise, Olivierculae, VladPrettenhofer, PeterGramfort, AlexandreGrobler, JaquesLayton, RobertVanderplas, JakeJoly, ArnaudHolt, BrianVaroquaux, Gaël
        http://arxiv.org/abs/1309.0238
        """
        super().__init__(random_state)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

        self.model = SVC(C=self.C,
                         kernel=self.kernel,
                         degree=self.degree,
                         gamma=self.gamma,
                         coef0=self.coef0,
                         shrinking=self.shrinking,
                         probability=self.probability,
                         tol=self.tol,
                         cache_size=self.cache_size,
                         class_weight=self.class_weight,
                         verbose=self.verbose,
                         max_iter=self.max_iter,
                         decision_function_shape=self.decision_function_shape,
                         break_ties=self.break_ties,
                         random_state=self.random_state)
        self.naming = SVMClassification.name


    def plot_svc_function(self):
        """
        Dichotomize the two selected elements and draw an image

        """
        print("-----* Two-dimensional Decision Boundary Diagram *-----")
        y = np.array(ClassificationWorkflowBase().y)
        X = np.array(ClassificationWorkflowBase().X)
        y = np.squeeze(y)
        clf = self.model.fit(X,y)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k',cmap="rainbow")
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = clf.decision_function(xy).reshape(X.shape)
        ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        save_fig('plot_svc', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.plot_svc_function()


class DecisionTreeClassification(ClassificationWorkflowBase):

    name = "Decision Tree"
    special_function = ["Decision Tree Plot"]

    def __init__(
            self,
            criterion='gini',
            splitter='best',
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0
    ):
        ##############################################################################
        #Classification of data using machine learning models with decision trees
        ##############################################################################
        """
        :param criterion:The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain
        :param splitter:The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
        :param max_depth:The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        :param min_samples_split:The minimum number of samples required to split an internal node
        :param min_samples_leaf:The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
        :param min_weight_fraction_leaf:The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
        :param max_features:The number of features to consider when looking for the best split
        :param random_state:Controls the randomness of the estimator.
        :param max_leaf_nodes:Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
        :param min_impurity_decrease:A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        :param class_weight:Weights associated with classes in the form {class_label: weight}.
        :param ccp_alpha:Complexity parameter used for Minimal Cost-Complexity Pruning.

        References
        ----------------------------------------
        API design for machine learning software: experiences from the scikit-learn project.Buitinck, LarLouppe, GillesBlondel, MathieuPedregosa, FabianMueller, AndreasGrise, Olivierculae, VladPrettenhofer, PeterGramfort, AlexandreGrobler, JaquesLayton, RobertVanderplas, JakeJoly, ArnaudHolt, BrianVaroquaux, Gaël
        http://arxiv.org/abs/1309.0238
        """
        super().__init__(random_state)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.model = DecisionTreeClassifier(criterion=self.criterion,
                                            splitter=self.splitter,
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                            max_features=self.max_features,
                                            random_state=self.random_state,
                                            max_leaf_nodes=self.max_leaf_nodes,
                                            min_impurity_decrease=self.min_impurity_decrease,
                                            class_weight=self.class_weight,
                                            ccp_alpha=self.ccp_alpha)
        self.naming = DecisionTreeClassification.name

    def plot_tree_function(self):
        ###################################################
        #Drawing decision tree diagrams
        ###################################################
        print("-----* Decision Tree Plot *-----")
        y = ClassificationWorkflowBase().y
        X = ClassificationWorkflowBase().X
        clf = self.model.fit(X,y)
        tree.plot_tree(clf, filled=True)
        save_fig('plot_decision_tree_classification', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.plot_tree_function()


class RandomForestClassification(ClassificationWorkflowBase):
    name = "Random Forest"
    special_function = ['Feature Importance', "Random Forest's Tree Plot"]

    def __init__(
            self,
            n_estimators=100,
            criterion='gini',
            max_depth=4,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=3,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=10):
        ##############################################################
        # RandomForestRegression is used to classify the data
        ##############################################################
        """
        :param n_estimators:int, default=100.The number of trees in the forest.
        :param criterion:{“gini”, “entropy”, “log_loss”}, default=”gini”The function to measure the quality of a split.
        :param max_depthint, default=None.The maximum depth of the tree.
        :param min_samples_splitint or float, default=2
                The minimum number of samples required to split an internal node
        :param min_samples_leafint or float, default=1
                The minimum number of samples required to be at a leaf node.
        :param min_weight_fraction_leaffloat, default=0.0
                The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
        :param max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
                The number of features to consider when looking for the best split:
        :param max_leaf_nodesint, default=None
                Grow trees with max_leaf_nodes in best-first fashion.
        :param min_impurity_decreasefloat, default=0.0
                A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        :param bootstrapbool, default=True
                Whether bootstrap samples are used when building trees.
        :param oob_scorebool, default=False
                Whether to use out-of-bag samples to estimate the generalization score.
        :param n_jobsint, default=None
                The number of jobs to run in parallel.
        :param random_stateint, RandomState instance or None, default=None
                Controls both the randomness of the bootstrapping of the samples used when building trees
        :param verboseint, default=0
                Controls the verbosity when fitting and predicting.
        :param warm_startbool, default=False
                When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the Glossary.
        :param class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
                Weights associated with classes in the form {class_label: weight}.
        :param ccp_alphanon-negative float, default=0.0
                Complexity parameter used for Minimal Cost-Complexity Pruning.
        :param max_samplesint or float, default=None
                If bootstrap is True, the number of samples to draw from X to train each base estimator.

        References
        ----------------------------------------
        scikit API:sklearn.ensemble.RandomForestClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        super().__init__(random_state)
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
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
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
                                            class_weight=self.class_weight,
                                            ccp_alpha=self.ccp_alpha,
                                            max_samples=self.max_samples)

    def feature_importances(self):
        ###################################################
        # Drawing feature importances barh diagram
        ###################################################
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
        save_fig("RandomForest_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

    def plot(self):
        ###################################################
        # Drawing diagrams of the first decision tree of forest
        ###################################################
        print("-----* Random Forest's Tree Plot *-----")
        tree.plot_tree(self.model.estimators_[0])
        save_fig("RandomForest_tree", MODEL_OUTPUT_IMAGE_PATH)
        pass

    def special_components(self):
        self.feature_importances()
        self.plot()


class XgboostClassification(ClassificationWorkflowBase):
    # https: // xgboost.readthedocs.io / en / stable / python / python_api.html  # module-xgboost.sklearn
    _SklObjective = Optional[
        Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ]
    ]
    name = "Xgboost"
    special_function = ['Feature Importance']

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: Optional[int] = None,
            max_leaves: Optional[int] = None,
            max_bin: Optional[int] = None,
            grow_policy=1,
            learning_rate: Optional[float] = None,
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
            importance_type: Optional[str] = 'weight',
            gpu_id: Optional[int] = None,
            validate_parameters: Optional[bool] = None,
            predictor: Optional[str] = None,
            enable_categorical: bool = False,
            eval_metric: Optional[Union[str, List[str], Callable]] = None,
            early_stopping_rounds: Optional[int] = None,
            **kwargs: Any
    ):
        super().__init__(random_state=42)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
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

        self.model = xgboost.XGBClassifier(
            n_estimators=self.n_estimators,
            objective=self.objective,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            max_bin=self.max_bin,
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
            early_stopping_rounds=self.early_stopping_rounds,
            )
        self.naming = XgboostClassification.name

    def _feature_importance(self):
        print("-----* Feature Importance *-----")
        columns_name = ClassificationWorkflowBase.X.columns
        # print the feature importance value orderly
        for feature_name, score in zip(list(columns_name), self.model.feature_importances_):
            print(feature_name, ":", score)

        # histograms present feature weights for XGBoost predictions
        plt.figure(figsize=(16, 8))
        plt.bar(range(len(columns_name)), self.model.feature_importances_, tick_label=columns_name)
        save_fig("xgboost_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

        # feature importance map ranked by importance
        plt.rcParams["figure.figsize"] = (14, 8)
        xgboost.plot_importance(self.model)
        save_fig("xgboost_feature_importance_score", MODEL_OUTPUT_IMAGE_PATH)

    # def plot(self):
    # TODO(solve the problem of failed to execute WindowsPath('dot'), make sure the Graphviz executables are on your systems' PATH
    #     ###################################################
    #     # Drawing diagrams of the first decision tree of xgboost
    #     ###################################################
    #     print("-----* Xgboost's Tree Plot *-----")
    #     xgboost.plot_tree(self.model)
    #     # node_params = {
    #     #     'shape': 'box',
    #     #     'style': 'filled,rounded',
    #     #     'fillcolor': '#78bceb'
    #     # }
    #     # xgboost.to_graphviz(self.model, condition_node_params = node_params)
    #     save_fig('plot_xgboost_tree', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self._feature_importance()
        # self.plot()
