# -*- coding: utf-8 -*-
# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from ..utils.base import save_fig
from ..global_variable import MODEL_OUTPUT_IMAGE_PATH
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, Sequence, Set, Literal
from multipledispatch import dispatch
from flaml import AutoML
from ._base import WorkflowBase
from .func.algo_classification._common import confusion_matrix_plot, contour_data
from .func.algo_classification._svm import plot_2d_decision_boundary
from .func.algo_classification._xgboost import feature_importance_map, feature_importance_value, feature_weights_histograms
from .func.algo_classification._decision_tree import decision_tree_plot
from .func.algo_classification._logistic import logistic_importance_plot
from .func.algo_classification._rf import feature_importances


class ClassificationWorkflowBase(WorkflowBase):
    """The base workflow class of classification algorithms."""

    common_function = ["Model Score", "Confusion Matrix"]

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
        """Fit the model by FLAML framework."""
        self.automl = AutoML()
        if self.customized:  # When the model is not built-in in FLAML framwork
            self.automl.add_learner(learner_name=self.customized_name, learner_class=self.customization)
        if y.shape[1] == 1:  # FLAML's data format validation mechanism
            y = y.squeeze()  # Convert a single dataFrame column into a series
        self.automl.fit(X_train=X, y_train=y, **self.settings)

    @dispatch(object)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform classification on samples in X by Scikit-learn framework."""
        y_predict = self.model.predict(X)
        return y_predict

    @dispatch(object, bool)
    def predict(self, X: pd.DataFrame, is_automl: bool = False) -> np.ndarray:
        """Perform classification on samples in X by FLAML framework."""
        y_predict = self.automl.predict(X)
        return y_predict

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        return dict()

    @property
    def customization(self) -> object:
        """The customized model of FLAML framework."""
        return object

    @property
    def auto_model(self) -> object:
        """Get AutoML trained model by FLAML framework."""
        return self.automl.model.estimator

    @staticmethod
    def _score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> None:
        print("-----* Model Score *-----")
        print(classification_report(y_true, y_predict))

    @staticmethod
    def _confusion_matrix_plot(y_test: pd.DataFrame, y_test_predict: pd.DataFrame,
                               trained_model: object, algorithm_name: str, store_path: str) -> None:
        print("-----* Confusion Matrix *-----")
        confusion_matrix_plot(y_test, y_test_predict, trained_model)
        save_fig(f"Confusion Matrix - {algorithm_name}", store_path)

    @staticmethod
    def _contour_data(X: pd.DataFrame, trained_model: Any) -> Tuple[List[np.ndarray], np.ndarray]:
        """Build up coordinate matrices as the data of contour plot."""
        return contour_data(X, trained_model)

    @dispatch()
    def common_components(self) -> None:
        """Invoke all common application functions for classification algorithms by Scikit-learn framework."""
        self._score(ClassificationWorkflowBase.y_test, ClassificationWorkflowBase.y_test_predict)
        self._confusion_matrix_plot(ClassificationWorkflowBase.y_test, ClassificationWorkflowBase.y_test_predict,
                                    self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def common_components(self, is_automl: bool) -> None:
        """Invoke all common application functions for classification algorithms by FLAML framework."""
        self._score(ClassificationWorkflowBase.y_test, ClassificationWorkflowBase.y_test_predict)
        self._confusion_matrix_plot(ClassificationWorkflowBase.y_test, ClassificationWorkflowBase.y_test_predict,
                                    self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class SVMClassification(ClassificationWorkflowBase):
    """The automation workflow of using SVC algorithm to make insightful products."""

    name = "Support Vector Machine"
    special_function = ['Two-dimensional Decision Boundary Diagram']

    def __init__(
            self,
            C: float = 1.0,
            kernel: Union[str, Callable] = 'rbf',
            degree: int = 3,
            gamma: Union[str, float] = "scale",
            coef0: float = 0.0,
            shrinking: bool = True,
            probability: bool = False,
            tol: float = 1e-3,
            cache_size: float = 200,
            class_weight: Union[dict, str, None] = None,
            verbose: bool = False,
            max_iter: int = -1,
            decision_function_shape: Literal['ovo', 'ovr'] = "ovr",
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
        self.customized = True
        self.customized_name = 'SVC'

    @property
    def settings(self) -> Dict:
        """The configuration of SVC to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'accuracy',
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": 'classification',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SVC of FLAML framework."""
        from flaml.model import SKLearnEstimator
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from sklearn.svm import SVC

        class MySVMClassification(SKLearnEstimator):
            def __init__(self, task='binary', n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = SVC

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    'C': {'domain': tune.uniform(lower=1, upper=data_size[0]),
                          'init_value': 1,
                          'low_cost_init_value': 1}
                }
                return space

        return MySVMClassification

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
        save_fig(f'2D Decision Boundary - {algorithm_name}', store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
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

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        if SVMClassification.X.shape[1] == 2:
            self._plot_2d_decision_boundary(SVMClassification.X, SVMClassification.X_test, SVMClassification.y_test,
                                            self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class DecisionTreeClassification(ClassificationWorkflowBase):
    """A decision tree classifier"""

    name = "Decision Tree"
    special_function = ["Decision Tree Plot"]

    def __init__(
            self,
            criterion: str = 'gini',
            splitter: str = 'best',
            max_depth: Optional[int] = 3,
            min_samples_split: Union[int, float] = 2,
            min_samples_leaf: Union[int, float] = 1,
            min_weight_fraction_leaf: Union[int, float] = 0.0,
            max_features: Union[int, float, str, None] = None,
            random_state: Optional[int] = None,
            max_leaf_nodes: Optional[int] = None,
            min_impurity_decrease: float = 0.0,
            class_weight: Union[dict, list[dict], str, None] = None,
            ccp_alpha: float = 0.0
    ) -> None:
        """
        Parameters
        ----------
        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.

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
                  `int(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

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

        class_weight : dict, list of dict or "balanced", default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If None, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.

            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        References
        ----------
        sklearn.tree.DecisionTreeClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier
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

    def plot_tree_function(self, cls: object, trained_model: any, algorithm_name: str, store_path: str) -> None:
        #Drawing decision tree diagrams
        print("-----* Decision Tree Plot *-----")
        decision_tree_plot(cls, trained_model, algorithm_name)
        save_fig('Decision Tree Classification Plot', store_path)

    '''
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
       '''

    @staticmethod
    def dt_plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: Any,
                                     algorithm_name: str, store_path: str,
                                     contour_data: Optional[List[np.ndarray]] = None,
                                     labels: Optional[np.ndarray] = None) -> None:
        """Plot the decision boundary of the trained model with the testing data set below."""
        print("-----* Two-dimensional Decision Boundary Diagram *-----")
        plot_2d_decision_boundary(X, X_test, y_test, trained_model, algorithm_name)
        save_fig(f'2d Decision Boundary - {algorithm_name}', store_path)

    def special_components(self, **kwargs) -> None:
        self.plot_tree_function(ClassificationWorkflowBase, self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        if DecisionTreeClassification.X.shape[1] == 2:
            self.dt_plot_2d_decision_boundary(DecisionTreeClassification.X, DecisionTreeClassification.X_test,
                                              DecisionTreeClassification.y_test,
                                              self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)


class RandomForestClassification(ClassificationWorkflowBase):
    """The automation workflow of using Random Forest algorithm to make insightful products."""

    name = "Random Forest"
    special_function = ['Feature Importance', "Random Forest's Tree Plot", "Drawing Decision Surfaces Plot"]

    def __init__(
            self,
            n_estimators: int = 100,
            criterion: str = 'gini',
            max_depth: Optional[int] = 4,
            min_samples_split: Union[int, float] = 4,
            min_samples_leaf: Union[int, float] = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: Union[str, int, float] = 'sqrt',
            max_leaf_nodes: Optional[int] = 3,
            min_impurity_decrease: float = 0.0,
            bootstrap: bool = True,
            oob_score: bool = False,
            n_jobs: Optional[int] = -1,
            random_state: Optional[int] = 42,
            verbose: int = 0,
            warm_start: bool = False,
            class_weight: Union[str, dict, list[dict], None] = None,
            ccp_alpha: float = 0.0,
            max_samples: Union[int, float] = 10
    ) -> None:
        """
        A random forest classifier.

        A random forest is a meta estimator that fits a number of decision tree
        classifiers on various sub-samples of the dataset and uses averaging to
        improve the predictive accuracy and control over-fitting.
        The sub-sample size is controlled with the `max_samples` parameter if
        `bootstrap=True` (default), otherwise the whole dataset is used to build
        each tree.

        Read more in the :ref:`User Guide <forest>`.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.

            .. versionchanged:: 0.22
               The default value of ``n_estimators`` changed from 10 to 100
               in 0.22.

        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
            Note: this parameter is tree-specific.

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
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
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

        class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
                default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.

            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``

            The "balanced_subsample" mode is the same as "balanced" except that
            weights are computed based on the bootstrap sample for every tree
            grown.

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

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
        scikit API: sklearn.ensemble.RandomForestClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
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
        self.naming = RandomForestClassification.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'accuracy',
            "estimator_list": ['rf'],  # list of ML learners
            "task": 'classification',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @staticmethod
    def _feature_importances(X_train: pd.DataFrame, trained_model: object, algorithm_name: str, store_path: str) -> None:
        """Draw the feature importance bar diagram."""
        print("-----* Feature Importance *-----")
        feature_importances(X_train, trained_model)
        save_fig(f"Feature Importance - {algorithm_name}", store_path)

    @staticmethod
    def _tree_plot(trained_model: object, algorithm_name: str, store_path: str) -> None:
        """Draw a diagram of the first decision tree of the forest."""
        print("-----* Random Forest's Tree Plot *-----")
        plt.figure()
        tree.plot_tree(trained_model.estimators_[0])
        save_fig(f"First Decision Tree - {algorithm_name}", store_path)

    @staticmethod
    def _plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object,
                                   algorithm_name: str, store_path: str) -> None:
        """Plot the decision boundary of the trained model with the testing data set below."""
        print("-----* Two-dimensional Decision Boundary Diagram *-----")
        plot_2d_decision_boundary(X, X_test, y_test, trained_model, algorithm_name)
        save_fig(f'2D Decision Boundary - {algorithm_name}', store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importances(RandomForestClassification.X_train, self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._tree_plot(self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        if RandomForestClassification.X.shape[1] == 2:
            self._plot_2d_decision_boundary(RandomForestClassification.X, RandomForestClassification.X_test,
                                            RandomForestClassification.y_test, self.model, self.naming,
                                            MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importances(RandomForestClassification.X_train, self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        self._tree_plot(self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
        if RandomForestClassification.X.shape[1] == 2:
            self._plot_2d_decision_boundary(RandomForestClassification.X, RandomForestClassification.X_test,
                                            RandomForestClassification.y_test, self.auto_model, self.naming,
                                            MODEL_OUTPUT_IMAGE_PATH)


class XgboostClassification(ClassificationWorkflowBase):
    """The automation workflow of using Xgboost algorithm to make insightful products."""

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

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'accuracy',
            "estimator_list": ['xgboost'],  # list of ML learners
            "task": 'classification',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @staticmethod
    def _feature_importance_series(data: pd.DataFrame, trained_model: any, algorithm_name: str, store_path: str) -> None:
        print("-----* Feature Importance *-----")
        feature_importance_value(data, trained_model)
        feature_weights_histograms(data, trained_model, algorithm_name)
        save_fig(f"Feature Weights - Histograms Plot - {algorithm_name}", store_path)
        feature_importance_map(trained_model, algorithm_name)
        save_fig(f"Feature Importance Map Plot - {algorithm_name}", store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importance_series(XgboostClassification.X, self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importance_series(XgboostClassification.X, self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

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


class LogisticRegressionClassification(ClassificationWorkflowBase):
    """The automation workflow of using Logistic Regression algorithm to make insightful products."""

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

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": 'accuracy',
            "estimator_list": ['lrl2'],  # list of ML learners
            "task": 'classification',  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    def _feature_importance(self, data: pd.DataFrame, trained_model: any, algorithm_name: str, store_path: str) -> None:
        """Print the feature coefficient value orderly."""
        print("-----* Feature Importance *-----")
        logistic_importance_plot(data, trained_model, algorithm_name)
        save_fig("LogisticRegression_feature_importance", store_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        self._feature_importance(ClassificationWorkflowBase.X, self.model, self.naming, MODEL_OUTPUT_IMAGE_PATH)

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        self._feature_importance(ClassificationWorkflowBase.X, self.auto_model, self.naming, MODEL_OUTPUT_IMAGE_PATH)
