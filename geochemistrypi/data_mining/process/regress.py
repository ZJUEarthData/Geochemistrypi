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
            print("-*-*- Hyper-parameters Specification -*-*-")
            print("N Estimators: The number of trees in the forest.")
            print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
            n_estimators = num_input(SECTION[2], "@N Estimators: ")
            print("Learning Rate: It controls the step-size in updating the weights. It shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.")
            print("Please specify the initial learning rate of Xgboost, such as 0.1.")
            learning_rate = float_input(0.01, SECTION[2], "@Learning Rate: ")
            print("Max Depth: The maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
            print("Please specify the maximum depth of a tree. A good starting range could be between 3 and 15, such as 4.")
            max_depth = num_input(SECTION[2], "@Max Depth: ")
            print("Subsample: The fraction of samples to be used for fitting the individual base learners.")
            print("Please specify the fraction of samples to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 0.8.")
            subsample = float_input(1, SECTION[2], "@Subsample: ")
            print("Colsample Bytree: The fraction of features to be used for fitting the individual base learners.")
            print("Please specify the fraction of features to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 1.")
            colsample_bytree = float_input(1, SECTION[2], "@Colsample Bytree: ")
            print("Alpha: L1 regularization term on weights.")
            print("Please specify the L1 regularization term on weights. A good starting range could be between 0 and 1.0, such as 0.")
            alpha = float_input(0, SECTION[2], "@Alpha: ")
            print("Lambda: L2 regularization term on weights.")
            print("Please specify the L2 regularization term on weights. A good starting range could be between 0 and 1.0, such as 1.")
            lambd = float_input(1, SECTION[2], "@Lambda: ")
            self.reg_workflow = XgboostRegression(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, alpha=alpha, lambd=lambd)
        elif self.model == "Decision Tree":
            print("-*-*- Hyper-parameters Specification -*-*-")
            print("Criterion: The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and “poisson” which uses reduction in Poisson deviance to find splits.")
            print("The default value is 'squared_error'. Optional criterions are 'friedman_mse', 'absolute_error' and 'poisson'.")
            criterions = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
            criterion = str_input(criterions, SECTION[2])
            print("Max Depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
            print("Please specify the maximum depth of the tree. A good starting range could be between 3 and 15, such as 4.")
            max_depth = num_input(SECTION[2], "@Max Depth: ")
            print("Min Samples Split: The minimum number of samples required to split an internal node.")
            print("Please specify the minimum number of samples required to split an internal node. A good starting range could be between 2 and 10, such as 3.")
            min_samples_split = num_input(SECTION[2], "@Min Samples Split: ")
            print("Min Samples Leaf: The minimum number of samples required to be at a leaf node.")
            print("Please specify the minimum number of samples required to be at a leaf node. A good starting range could be between 1 and 10, such as 2.")
            min_samples_leaf = num_input(SECTION[2], "@Min Samples Leaf: ")
            print("Max Features: The number of features to consider when looking for the best split.")
            print("Please specify the number of features to consider when looking for the best split. A good starting range could be between 1 and the total number of features in the dataset.")
            max_features = num_input(SECTION[2], "@Max Features: ")
            self.reg_workflow = DecisionTreeRegression(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        elif self.model == "Extra-Trees":
            self.reg_workflow = ExtraTreeRegression()
        elif self.model == "Random Forest":
            self.reg_workflow = RandomForestRegression()
        elif self.model == "Support Vector Machine":
            self.reg_workflow = SVMRegression()
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
