# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from model.regression import PolynomialRegression, XgboostRegression, RegressionWorkflowBase
from data.data_readiness import num_input

class RegressionModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.reg_workflow = RegressionWorkflowBase()

    def activate(self, X, y):
        X_train, X_test, y_train, y_test = self.reg_workflow.data_split(X, y)

        # model option
        if self.model == "Polynomial Regression":
            poly_degree = num_input("Please specify the maximal degree of the polynomial features.\n@Degree:")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
        elif self.model == "Xgboost":
            self.reg_workflow = XgboostRegression()

        # common components for every regression algorithm
        self.reg_workflow.show_info()
        self.reg_workflow.fit(X_train, y_train)
        y_test_prediction = self.reg_workflow.predict(X_test)
        self.reg_workflow.score(y_test, y_test_prediction)
        self.reg_workflow.cross_validation(X_train, y_train, cv_num=10)

        # special components of different algorithms
        self.reg_workflow.special_components()
