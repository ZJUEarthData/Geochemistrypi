# -*- coding: utf-8 -*-
from ..model.regression import PolynomialRegression, XgboostRegression, DecisionTreeRegression, ExtraTreeRegression,\
    RandomForestRegression, RegressionWorkflowBase, SVMRegression, DNNRegression, LinearRegression2
from ..data.data_readiness import num_input, float_input, tuple_input, limit_num_input, str_input
from ..global_variable import SECTION, DATASET_OUTPUT_PATH
from multipledispatch import dispatch
import pandas as pd


class RegressionModelSelection(object):
    """Simulate the normal way of training regression algorithms."""

    def __init__(self, model):
        self.model = model
        self.reg_workflow = RegressionWorkflowBase()

    @dispatch(object, object, object, object, object, object)
    def activate(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame,
                 X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """Train by Scikit-learn framework."""

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
        if self.model == "Polynomial Regression":
            print("-*-*- Hyper-parameters Specification -*-*-")
            print("Please specify the maximal degree of the polynomial features.")
            poly_degree = num_input(SECTION[2], "@Degree: ")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
            self.reg_workflow.data_upload(X_train=X_train, X_test=X_test)
        elif self.model == "Xgboost":
            hyper_parameters = XgboostRegression.manual_hyper_parameters()
            self.reg_workflow = XgboostRegression(n_estimators=hyper_parameters['n_estimators'], learning_rate=hyper_parameters['learning_rate'], max_depth=hyper_parameters['max_depth'], subsample=hyper_parameters['subsample'], colsample_bytree=hyper_parameters['colsample_bytree'], alpha=hyper_parameters['alpha'], lambd=hyper_parameters['lambd'])
        elif self.model == "Decision Tree":
            hyper_parameters = DecisionTreeRegression.manual_hyper_parameters()
            self.reg_workflow = DecisionTreeRegression(criterion=hyper_parameters['criterion'], max_depth=hyper_parameters['max_depth'], min_samples_split=hyper_parameters['min_samples_split'], min_samples_leaf=hyper_parameters['min_samples_leaf'], max_features=hyper_parameters['max_features'])
        elif self.model == "Extra-Trees":
            hyper_parameters = ExtraTreeRegression.manual_hyper_parameters()
            self.reg_workflow = ExtraTreeRegression(n_estimators=hyper_parameters['n_estimators'], max_depth=hyper_parameters['max_depth'], min_samples_split=hyper_parameters['min_samples_split'], min_samples_leaf=hyper_parameters['min_samples_leaf'], max_features=hyper_parameters['max_features'], bootstrap=hyper_parameters['bootstrap'], oob_score=hyper_parameters['oob_score'])
        elif self.model == "Random Forest":
            hyper_parameters = RandomForestRegression.manual_hyper_parameters()
            self.reg_workflow = RandomForestRegression(n_estimators=hyper_parameters['n_estimators'], max_depth=hyper_parameters['max_depth'], min_samples_split=hyper_parameters['min_samples_split'], min_samples_leaf=hyper_parameters['min_samples_leaf'], max_features=hyper_parameters['max_features'], bootstrap=hyper_parameters['bootstrap'], oob_score=hyper_parameters['oob_score'])
        elif self.model == "Support Vector Machine":
            hyper_parameters = SVMRegression.manual_hyper_parameters()
            if hyper_parameters["kernel"] == "linear":
                self.reg_workflow = SVMRegression(kernel=hyper_parameters["kernel"], C=hyper_parameters["C"], shrinking=hyper_parameters["shrinking"])
            elif hyper_parameters["kernel"] == "poly":
                self.reg_workflow = SVMRegression(kernel=hyper_parameters["kernel"], degree=hyper_parameters["degree"], gamma=hyper_parameters["gamma"], C=hyper_parameters["C"], shrinking=hyper_parameters["shrinking"])
            elif hyper_parameters["kernel"] == "rbf":
                self.reg_workflow = SVMRegression(kernel=hyper_parameters["kernel"], gamma=hyper_parameters["gamma"], C=hyper_parameters["C"], shrinking=hyper_parameters["shrinking"])
            elif hyper_parameters["kernel"] == "sigmoid":
                self.reg_workflow = SVMRegression(kernel=hyper_parameters["kernel"], gamma=hyper_parameters["gamma"], C=hyper_parameters["C"], shrinking=hyper_parameters["shrinking"])
        elif self.model == "Deep Neural Networks":
            print("-*-*- Hyper-parameters Specification -*-*-")
            print("Learning Rate: It controls the step-size in updating the weights.")
            print("Please specify the initial learning rate of the the neural networks, such as 0.001.")
            learning_rate = float_input(0.05, SECTION[2], "@Learning Rate: ")
            print("Hidden Layer Sizes: The ith element represents the number of neurons in the ith hidden layer.")
            print("Please specify the size of hidden layer and the number of neurons in the each hidden layer.")
            hidden_layer = tuple_input((50, 25, 5), SECTION[2], "@Hidden Layer Sizes: ")
            # batch_size = limit_num_input()
            self.reg_workflow = DNNRegression(learning_rate_init=learning_rate,
                                              hidden_layer_sizes=hidden_layer)
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

        # Save the prediction result
        self.reg_workflow.data_save(y_test_predict, "Y Test Predict", DATASET_OUTPUT_PATH, "Model Prediction")

        # Save the trained model
        self.reg_workflow.save_model()

    @dispatch(object, object, object, object, object, object, bool)
    def activate(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.DataFrame, y_test: pd.DataFrame, is_automl: bool) -> None:
        """Train by FLAML framework + RAY framework."""

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

        # Save the prediction result
        self.reg_workflow.data_save(y_test_predict, "Y Test Predict", DATASET_OUTPUT_PATH, "Model Prediction")

        # Save the trained model
        self.reg_workflow.save_model(is_automl)
