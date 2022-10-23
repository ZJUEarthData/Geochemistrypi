# -*- coding: utf-8 -*-
# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, Sequence, Set
from matplotlib.colors import ListedColormap
from ._base import WorkflowBase
from .func.algo_classification._svm import plot_2d_decision_boundary
# sys.path.append("..")


class ClassificationWorkflowBase(WorkflowBase):
    """The base workflow class of classification algorithms."""

    common_function = ["Model Score", "Confusion Matrix"]

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model."""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform classification on samples in X."""
        y_predict = self.model.predict(X)
        return y_predict

    @staticmethod
    def score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> None:
        print("-----* Model Score *-----")
        print(classification_report(y_true, y_predict))

    def confusion_matrix_plot(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_test_prediction: pd.DataFrame) -> None:
        print("-----* Confusion Matrix *-----")
        print(confusion_matrix(y_test, y_test_prediction))
        plt.figure()
        plot_confusion_matrix(self.model, X_test, y_test)
        save_fig(f"Confusion Matrix - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)

    @staticmethod
    def contour_data(X: pd.DataFrame, trained_model: Any) -> Tuple[List[np.ndarray], np.ndarray]:
        """Build up coordinate matrices as the data of contour plot.
        Parameters
        ----------
        X : pd.DataFrame (n_samples, n_components)
            The complete feature data.
        trained_model : Any
            Te algorithm model class from sklearn is trained.
        Returns
        -------
        matrices : List[np.ndarray]
            Coordinate matrices.
        labels : np.ndarray
            Predicted value by the trained model with coordinate data as input data.
        """

        # build up coordinate matrices from coordinate vectors.
        xi = [np.arange(X.iloc[:, i].min(), X.iloc[:, i].max(), (X.iloc[:, i].max()-X.iloc[:, i].min())/50)
              for i in range(X.shape[1])]
        ndim = len(xi)
        s0 = (1,) * ndim
        matrices = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]
        matrices[0].shape = (1, -1) + s0[2:]
        matrices[1].shape = (-1, 1) + s0[2:]
        matrices = np.broadcast_arrays(*matrices, subok=True)

        # get the labels of the coordinate matrices through the trained model
        input_array = np.column_stack((i.ravel() for i in matrices))
        labels = trained_model.predict(input_array).reshape(matrices[0].shape)

        return matrices, labels


class SVMClassification(ClassificationWorkflowBase):
    """The automation workflow of using SVC algorithm to make insightful products."""

    name = "Support Vector Machine"
    special_function = ['Two-dimensional Decision Boundary Diagram']

    def __init__(
            self,
            C: float = 1.0,
            kernel: Set = 'rbf',
            degree: int = 3,
            gamma: Set = "scale",
            coef0: float = 0.0,
            shrinking: bool = True,
            probability: bool = False,
            tol: float = 1e-3,
            cache_size: float = 200,
            class_weight: Union[Dict, str] = None,
            verbose: bool = False,
            max_iter: int = -1,
            decision_function_shape: Set = "ovr",
            break_ties: bool = False,
            random_state: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive. The penalty
            is a squared l2 penalty.
        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
            If none is given, 'rbf' will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix
            should be an array of shape ``(n_samples, n_samples)``.
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
        shrinking : bool, default=True
            Whether to use the shrinking heuristic.
            See the :ref:`User Guide <shrinking_svm>`.
        probability : bool, default=False
            Whether to enable probability estimates. This must be enabled prior
            to calling `fit`, will slow down that method as it internally uses
            5-fold cross-validation, and `predict_proba` may be inconsistent with
            `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.
        tol : float, default=1e-3
            Tolerance for stopping criterion.
        cache_size : float, default=200
            Specify the size of the kernel cache (in MB).
        class_weight : dict or 'balanced', default=None
            Set the parameter C of class i to class_weight[i]*C for
            SVC. If not given, all classes are supposed to have
            weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.
        verbose : bool, default=False
            Enable verbose output. Note that this setting takes advantage of a
            per-process runtime setting in libsvm that, if enabled, may not work
            properly in a multithreaded context.
        max_iter : int, default=-1
            Hard limit on iterations within solver, or -1 for no limit.
        decision_function_shape : {'ovo', 'ovr'}, default='ovr'
            Whether to return a one-vs-rest ('ovr') decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original
            one-vs-one ('ovo') decision function of libsvm which has shape
            (n_samples, n_classes * (n_classes - 1) / 2). However, note that
            internally, one-vs-one ('ovo') is always used as a multi-class strategy
            to train models; an ovr matrix is only constructed from the ovo matrix.
            The parameter is ignored for binary classification.
            .. versionchanged:: 0.19
                decision_function_shape is 'ovr' by default.
            .. versionadded:: 0.17
               *decision_function_shape='ovr'* is recommended.
            .. versionchanged:: 0.17
               Deprecated *decision_function_shape='ovo' and None*.
        break_ties : bool, default=False
            If true, ``decision_function_shape='ovr'``, and number of classes > 2,
            :term:`predict` will break ties according to the confidence values of
            :term:`decision_function`; otherwise the first class among the tied
            classes is returned. Please note that breaking ties comes at a
            relatively high computational cost compared to a simple predict.
            .. versionadded:: 0.22
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling the data for
            probability estimates. Ignored when `probability` is False.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        References
        ----------
        scikit API: sklearn.svm.SVC
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """
        super().__init__()
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

    # TODO(Sany sanyhew1097618435@163.com): think about What layout of the graph is more user-friendly.
    """
    def plot_svc_surface_function(self):
        # divide the two selected elements and draw an image
        print("-----* Two-dimensional Decision Surface Boundary Diagram *-----")
        plt.figure()
        y = np.array(ClassificationWorkflowBase().y)
        X = np.array(ClassificationWorkflowBase().X)
        X = PCA(n_components=2).fit_transform(X)
        y = np.squeeze(y)
        clf = self.model.fit(X, y)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=ListedColormap(['#FF0000', '#0000FF']),alpha=0.6)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        Z = clf.decision_function(np.c_[X.ravel(), Y.ravel()])
        Z = Z.reshape(X.shape)
        ax.contourf(X, Y, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        save_fig('SVC Surface Function Plot', MODEL_OUTPUT_IMAGE_PATH)
    """

    @staticmethod
    def _plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: Any,
                                   algorithm_name: str, store_path: str, contour_data: Optional[List[np.ndarray]] = None,
                                   labels: Optional[np.ndarray] = None) -> None:
        """Plot the decision boundary of the trained model with the testing data set below."""
        print("-----* Two-dimensional Decision Boundary Diagram *-----")
        plot_2d_decision_boundary(X, X_test, y_test, trained_model, algorithm_name)
        save_fig(f'2d Decision Boundary - {algorithm_name}', store_path)

    def special_components(self, **kwargs) -> None:
        if SVMClassification.X.shape[1] == 2:
            self._plot_2d_decision_boundary(SVMClassification.X, SVMClassification.X_test, SVMClassification.y_test,
                                            self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

            # TODO(Sany sanyhew1097618435@163.com): Check whether 3d decision boundary makes sense.
            #  If it does, use the code below.
            # contour_matrices, labels = self.contour_data(SVMClassification.X, self.model)
            # two_dimen_axis_index, two_dimen_X = self.choose_dimension_data(SVMClassification.X, 2)
            # two_dimen_X_test = SVMClassification.X_test.iloc[:, two_dimen_axis_index]
            # two_dimen_contour_matrices = [contour_matrices[i] for i in two_dimen_axis_index]
            # self._plot_2d_decision_boundary(two_dimen_X, two_dimen_X_test, SVMClassification.y_test, two_dimen_contour_matrices, labels)


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
        super().__init__()
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
        plt.figure()
        y = ClassificationWorkflowBase().y
        X = ClassificationWorkflowBase().X
        clf = self.model.fit(X,y)
        tree.plot_tree(clf, filled=True)
        save_fig('Decision Tree Classification Plot', MODEL_OUTPUT_IMAGE_PATH)

    def decision_surface_plot(self):
        #############################################################
        #Plot the decision surfaces of forests of the data
        #############################################################
        print("Decision_Surface_Plot", "Drawing Decision Surface Plot")
        plt.figure()
        y = np.array(ClassificationWorkflowBase().y)
        X = np.array(ClassificationWorkflowBase().X)
        X = PCA(n_components=2).fit_transform(X)
        self.model.fit(X,y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"], s=2)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1],c=y,cmap=ListedColormap(['#FF0000', '#0000FF']),alpha=0.6,s=20)
        plt.suptitle("Decision Surface Plot ", fontsize=12)
        plt.axis("tight")
        plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
        save_fig('Decision Surface Plot', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.plot_tree_function()
        self.decision_surface_plot()


class RandomForestClassification(ClassificationWorkflowBase):
    name = "Random Forest"
    special_function = ['Feature Importance', "Random Forest's Tree Plot", "Drawing Decision Surfaces Plot"]

    def __init__(
            self,
            n_estimators=100,
            criterion='gini',
            max_depth=4,
            min_samples_split=4,
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
        plt.figure()
        tree.plot_tree(self.model.estimators_[0])
        save_fig("RandomForest_tree", MODEL_OUTPUT_IMAGE_PATH)

    def decision_surfaces_plot(self):
        #############################################################
        #Plot the decision surfaces of forests of the data
        #############################################################
        print("-----* Decision Surfaces Plot *-----")
        plt.figure()
        y = np.array(ClassificationWorkflowBase().y)
        X = np.array(ClassificationWorkflowBase().X)
        X = PCA(n_components=2).fit_transform(X)
        self.model.fit(X,y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        estimator_alpha = 1.0 / len(self.model.estimators_)
        for tree in self.model.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),alpha=0.6, s=20)
        plt.suptitle("Decision Surfaces Plot ", fontsize=12)
        plt.axis("tight")
        plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
        save_fig('Decision Surfaces Plot - RandomForest', MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.feature_importances()
        self.plot()
        self.decision_surfaces_plot()


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
        super().__init__()
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


class LogisticRegressionClassification(ClassificationWorkflowBase):
    name = "Logistic Regression"
    special_function = ['Feature Importance']

    def __init__(
         self,
         penalty: str = 'l2',
         dual: bool = False,
         tol: float = 0.0001,
         C: float = 1.0,
         fit_intercept: bool = True,
         intercept_scaling: float = 1,
         class_weight: Optional[Union[Dict, str]] = None,
         random_state: Optional[int] = None,
         solver: str = 'lbfgs',
         max_iter: int = 100,
         multi_class: str = 'auto',
         verbose: int = 0,
         warm_start: bool = False,
         n_jobs: int = None,
         l1_ratio: float = None
    ) -> None:
        """
        Parameters
        ----------
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Specify the norm of the penalty:
            - `'none'`: no penalty is added;
            - `'l2'`: add a L2 penalty term and it is the default choice;
            - `'l1'`: add a L1 penalty term;
            - `'elasticnet'`: both L1 and L2 penalty terms are added.
            .. warning::
               Some penalties may not work with some solvers. See the parameter
               `solver` below, to know the compatibility between the penalty and
               solver.
            .. versionadded:: 0.19
               l1 penalty with SAGA solver (allowing 'multinomial' + L1)
        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for
            l2 penalty with liblinear solver. Prefer dual=False when
            n_samples > n_features.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        C : float, default=1.0
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger
            regularization.
        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.
        intercept_scaling : float, default=1
            Useful only when the solver 'liblinear' is used
            and self.fit_intercept is set to True. In this case, x becomes
            [x, self.intercept_scaling],
            i.e. a "synthetic" feature with constant value equal to
            intercept_scaling is appended to the instance vector.
            The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
            Note! the synthetic feature weight is subject to l1/l2 regularization
            as all other features.
            To lessen the effect of regularization on synthetic feature weight
            (and therefore on the intercept) intercept_scaling has to be increased.
        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
            .. versionadded:: 0.17
               *class_weight='balanced'*
        random_state : int, RandomState instance, default=None
            Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
            data. See :term:`Glossary <random_state>` for details.
        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
                default='lbfgs'
            Algorithm to use in the optimization problem. Default is 'lbfgs'.
            To choose a solver, you might want to consider the following aspects:
                - For small datasets, 'liblinear' is a good choice, whereas 'sag'
                  and 'saga' are faster for large ones;
                - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
                  'lbfgs' handle multinomial loss;
                - 'liblinear' is limited to one-versus-rest schemes.
            .. warning::
               The choice of the algorithm depends on the penalty chosen:
               Supported penalties by solver:
               - 'newton-cg'   -   ['l2', 'none']
               - 'lbfgs'       -   ['l2', 'none']
               - 'liblinear'   -   ['l1', 'l2']
               - 'sag'         -   ['l2', 'none']
               - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']
            .. note::
               'sag' and 'saga' fast convergence is only guaranteed on
               features with approximately the same scale. You can
               preprocess the data with a scaler from :mod:`sklearn.preprocessing`.
            .. seealso::
               Refer to the User Guide for more information regarding
               :class:`LogisticRegression` and more specifically the
               :ref:`Table <Logistic_regression>`
               summarizing solver/penalty supports.
            .. versionadded:: 0.17
               Stochastic Average Gradient descent solver.
            .. versionadded:: 0.19
               SAGA solver.
            .. versionchanged:: 0.22
                The default solver changed from 'liblinear' to 'lbfgs' in 0.22.
        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.
        multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
            If the option chosen is 'ovr', then a binary problem is fit for each
            label. For 'multinomial' the loss minimised is the multinomial loss fit
            across the entire probability distribution, *even when the data is
            binary*. 'multinomial' is unavailable when solver='liblinear'.
            'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
            and otherwise selects 'multinomial'.
            .. versionadded:: 0.18
               Stochastic Average Gradient descent solver for 'multinomial' case.
            .. versionchanged:: 0.22
                Default changed from 'ovr' to 'auto' in 0.22.
        verbose : int, default=0
            For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            Useless for liblinear solver. See :term:`the Glossary <warm_start>`.
            .. versionadded:: 0.17
               *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.
        n_jobs : int, default=None
            Number of CPU cores used when parallelizing over classes if
            multi_class='ovr'". This parameter is ignored when the ``solver`` is
            set to 'liblinear' regardless of whether 'multi_class' is specified or
            not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.
        l1_ratio : float, default=None
            The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
            used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
            to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
            to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
            combination of L1 and L2.
        References
        ----------
        scikit API: sklearn.linear_model.LogisticRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
        super().__init__()
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.l1_ratio = l1_ratio

        self.model = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            warm_start=self.warm_start,
            l1_ratio=self.l1_ratio,
            )

        self.naming = LogisticRegressionClassification.name

    def feature_importance(self):
        columns_name = ClassificationWorkflowBase.X.columns

        # print the feature coefficient value orderly
        print("-----* Feature Importance *-----")
        for feature_name, score in zip(list(columns_name), self.model.coef_.flatten()):
            print(feature_name, ":", score)

        # feature importance map ranked by coefficient
        coef_lr = pd.DataFrame({
            'var': columns_name,
            'coef': self.model.coef_.flatten()
        })
        index_sort = np.abs(coef_lr['coef']).sort_values().index
        coef_lr_sort = coef_lr.loc[index_sort, :]

        # Horizontal column chart plot
        fig, ax = plt.subplots(figsize=(14,8))
        x, y = coef_lr_sort['var'], coef_lr_sort['coef']
        rects = plt.barh(x, y, color='dodgerblue')
        plt.grid(linestyle="-.", axis='y', alpha=0.4)
        plt.tight_layout()

        # Add data labels
        for rect in rects:
            w = rect.get_width()
            ax.text(w, rect.get_y() + rect.get_height() / 2, '%.2f' % w, ha='left', va='center')
        save_fig("LogisticRegression_feature_importance", MODEL_OUTPUT_IMAGE_PATH)

    def special_components(self):
        self.feature_importance()