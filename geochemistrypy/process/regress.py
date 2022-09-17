# -*- coding: utf-8 -*-
# import sys
from model.regression import PolynomialRegression, XgboostRegression, DecisionTreeRegression, ExtraTreeRegression,\
    RandomForestRegression, RegressionWorkflowBase, SupportVectorRegression, DNNRegression
from data.data_readiness import num_input, float_input, tuple_input
from global_variable import SECTION
import pandas as pd
from typing import Optional
# sys.path.append("..")


class RegressionModelSelection(object):

    def __init__(self, model):
        self.model = model
        self.reg_workflow = RegressionWorkflowBase()

    def activate(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
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
        elif self.model == "Support Vector Machine":
            self.reg_workflow = SupportVectorRegression()
        elif self.model == "Deep Neural Networks":
            print("Please specify the init learning rate of the the neural networks.")
            learning_rate = float_input(SECTION[2], "@Learning_rate:")
            print("Please specify the size of hidden layer and the number of neurons in the each hidden layer.")
            hidden_layer = tuple_input(SECTION[2], "@Hidden_layer_sizes:")
            self.reg_workflow = DNNRegression(learning_rate_init=learning_rate, hidden_layer_sizes=hidden_layer)

        self.reg_workflow.X_train = X_train
        self.reg_workflow.y_train = y_train

        # common components for every regression algorithm
        self.reg_workflow.show_info()
        self.reg_workflow.fit(X_train, y_train)
        y_test_prediction = self.reg_workflow.predict(X_test)
        self.reg_workflow.score(y_test, y_test_prediction)
        self.reg_workflow.cross_validation(X_train, y_train, cv_num=10)

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # special components of different algorithms
        self.reg_workflow.special_components()
