# -*- coding: utf-8 -*-
# import sys
from model.classification import ClassificationWorkflowBase, SVMClassification, DecisionTreeClassification, RandomForestClassification,XgboostClassification

# sys.path.append("..")


class ClassificationModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.clf_workflow = ClassificationWorkflowBase()

    def activate(self, X, y):
        X_train, X_test, y_train, y_test = self.clf_workflow.data_split(X, y)

        # model option
        if self.model == "Support Vector Machine":
            self.clf_workflow = SVMClassification()
        elif self.model == "Decision Tree":
            self.clf_workflow = DecisionTreeClassification()
        elif self.model == "Random Forest":
            self.clf_workflow = RandomForestClassification()
            self.clf_workflow.X_train = X_train
            self.clf_workflow.y_train = y_train
        elif self.model == "Xgboost":
            self.clf_workflow = XgboostClassification()
        # common components for every classification algorithm
        self.clf_workflow.show_info()
        self.clf_workflow.fit(X_train, y_train)
        y_test_prediction = self.clf_workflow.predict(X_test)
        self.clf_workflow.score(y_test, y_test_prediction)
        self.clf_workflow.confusion_matrix_plot(X_test, y_test, y_test_prediction)

        # special components of different algorithms
        self.clf_workflow.special_components()
