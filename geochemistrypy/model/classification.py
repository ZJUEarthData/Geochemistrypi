# -*- coding: utf-8 -*-
# import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from sklearn.model_selection import train_test_split
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
        save_fig(f"Confusion Matrix - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)


class SVMClassification(ClassificationWorkflowBase):

    name = "Support Vector Machine"
    special_function = ['two-dimensional decision boundary diagram']

    def __init__(
            self,
            C=1.0,
            kernel="rbf",
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

    def plot_ready(self):
        """
        Data processing preparation before drawing
        """
        self.X = ClassificationWorkflowBase().X
        self.y = ClassificationWorkflowBase().y
        y = np.array(self.y)
        X = np.array(self.X)
        y = np.squeeze(y)
        clf = self.model.fit(X,y)
        plt.scatter(X[:,0], X[:,1], c=y,edgecolors='k', s=50, cmap="rainbow")
        return clf


    def plot_svc_function(self,data, ax=None):
        """
        :param data: Data needed to draw two-dimensional decision boundary diagrams
        :param ax:Graph object
        """
        print("-----* Plot SVC Function *-----")
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = data.decision_function(xy).reshape(X.shape)
        ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        save_fig('plot_svc', MODEL_OUTPUT_IMAGE_PATH)
        

        
    def special_components(self):
        self.plot_svc_function(self.plot_ready())
