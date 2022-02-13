# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from model.classification import *


class ClassificationModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.clf_workflow = ClassificationWorkflowBase()

    def activate(self, X, y):
        X_train, X_test, y_train, y_test = self.clf_workflow.data_split(X, y)

        # model option
        if self.model == "Support Vector Machine":
            self.clf_workflow = SVMClassification()

        # common components for every classification algorithm
        self.clf_workflow.show_info()
        self.clf_workflow.fit(X_train, y_train)
        y_test_prediction = self.clf_workflow.predict(X_test)
        self.clf_workflow.score(y_test, y_test_prediction)
        self.clf_workflow.confusion_matrix_plot(X_test, y_test, y_test_prediction)

        # special components of different algorithms
        self.clf_workflow.special_components()