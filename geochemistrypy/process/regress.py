# -*- coding: utf-8 -*-
# import sys
from model.regression import PolynomialRegression, XgboostRegression, DecisionTreeRegression, ExtraTreeRegression,\
    RandomForestRegression, RegressionWorkflowBase
from data.data_readiness import num_input
from global_variable import SECTION
# sys.path.append("..")


class RegressionModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.reg_workflow = RegressionWorkflowBase()

    def activate(self, X, y):
        X_train, X_test, y_train, y_test = self.reg_workflow.data_split(X, y)

        # model option
        if self.model == "Polynomial Regression":
            print("Please specify the maximal degree of the polynomial features.")
            poly_degree = num_input(SECTION[2], "@Degree:")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
        elif self.model == "Xgboost":
            self.reg_workflow = XgboostRegression()
        elif self.model == "Decision Tree":
            print("Please specify the max depth of the decision tree regression.")
            dts_max_depth = num_input(SECTION[2], "@Max_depth:")
            self.reg_workflow = DecisionTreeRegression(max_depth=dts_max_depth)
        elif self.model == "Extra-Trees":
            self.reg_workflow = ExtraTreeRegression()
        elif self.model == "Random Forest":
            self.reg_workflow = RandomForestRegression()
        self.reg_workflow.X_train = X_train
        self.reg_workflow.y_train = y_train

        # common components for every regression algorithm
        self.reg_workflow.show_info()
        self.reg_workflow.fit(X_train, y_train)
        y_test_prediction = self.reg_workflow.predict(X_test)
        self.reg_workflow.score(y_test, y_test_prediction)
        self.reg_workflow.cross_validation(X_train, y_train, cv_num=10)

        # special components of different algorithms
        self.reg_workflow.special_components()
