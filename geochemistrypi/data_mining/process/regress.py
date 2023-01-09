# -*- coding: utf-8 -*-
from ..model.regression import PolynomialRegression, XgboostRegression, DecisionTreeRegression, ExtraTreeRegression,\
    RandomForestRegression, RegressionWorkflowBase, SVMRegression, DNNRegression, LinearRegression2
from ..data.data_readiness import num_input, float_input, tuple_input
from ..global_variable import SECTION
from multipledispatch import dispatch


class RegressionModelSelection(object):
    """Simulate the normal way of training regression algorithms."""

    def __init__(self, model):
        self.model = model
        self.reg_workflow = RegressionWorkflowBase()

    @dispatch(object, object)
    def activate(self, X, y):
        """Train by Scikit-learn framework."""

        X_train, X_test, y_train, y_test = self.reg_workflow.data_split(X, y)
        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
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
            self.reg_workflow = SVMRegression()
        elif self.model == "Deep Neural Networks":
            print("Please specify the init learning rate of the the neural networks.")
            learning_rate = float_input(0.05, SECTION[2], "@Learning_rate:")
            print("Please specify the size of hidden layer and the number of neurons in the each hidden layer.")
            hidden_layer = tuple_input((50, 25, 5), SECTION[2], "@Hidden_layer_sizes:")
            self.reg_workflow = DNNRegression(learning_rate_init=learning_rate, hidden_layer_sizes=hidden_layer)
        elif self.model == "Linear Regression":
            self.reg_workflow = LinearRegression2()

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train)
        y_test_predict = self.reg_workflow.predict(X_test)
        y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
        self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Common components for every regression algorithm
        self.reg_workflow.common_components()

        # Special components of different algorithms
        self.reg_workflow.special_components()

        # Save the trained model
        self.reg_workflow.save_model()

    @dispatch(object, object, bool)
    def activate(self, X, y, is_automl):
        """Train by FLAML framework + RAY framework."""

        X_train, X_test, y_train, y_test = self.reg_workflow.data_split(X, y)
        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
        if self.model == "Polynomial Regression":
            # TODO(Sany sanyhew1097618435@163.com): Find the proper way for polynomial regression
            print("Please specify the maximal degree of the polynomial features.")
            poly_degree = num_input(SECTION[2], "@Degree:")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
        elif self.model == "Xgboost":
            self.reg_workflow = XgboostRegression()
        elif self.model == "Decision Tree":
            self.reg_workflow = DecisionTreeRegression()
        elif self.model == "Extra-Trees":
            self.reg_workflow = ExtraTreeRegression()
        elif self.model == "Random Forest":
            self.reg_workflow = RandomForestRegression()
        elif self.model == "Support Vector Machine":
            self.reg_workflow = SVMRegression()
        elif self.model == "Deep Neural Networks":
            self.reg_workflow = DNNRegression()
        elif self.model == "Linear Regression":
            self.reg_workflow = LinearRegression2()

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train, is_automl)
        y_test_predict = self.reg_workflow.predict(X_test, is_automl)
        y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
        self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Common components for every regression algorithm
        self.reg_workflow.common_components(is_automl)

        # Special components of different algorithms
        self.reg_workflow.special_components(is_automl)

        # Save the trained model
        self.reg_workflow.save_model(is_automl)
