# -*- coding: utf-8 -*-
import pandas as pd
from multipledispatch import dispatch
from ..global_variable import DATASET_OUTPUT_PATH
from ..model.classification import ClassificationWorkflowBase, SVMClassification, DecisionTreeClassification,\
    RandomForestClassification, XgboostClassification, LogisticRegressionClassification
from ..data.data_readiness import num_input, float_input, str_input, limit_num_input
from ..global_variable import SECTION


class ClassificationModelSelection(object):
    """Simulate the normal way of training classification algorithms."""

    def __init__(self, model):
        self.model = model
        self.clf_workflow = ClassificationWorkflowBase()

    @dispatch(object, object, object, object, object, object)
    def activate(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame,
                 X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """Train by Scikit-learn framework."""

        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # model option
        if self.model == "Support Vector Machine":
            self.clf_workflow = SVMClassification()
        elif self.model == "Decision Tree":
            print("-*-*- Hyper-parameters Specification -*-*-")
            print("Criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain.")
            print("The default value is 'gini'. Optional criterions are 'entropy' and 'log_loss'.")
            criterions = ["gini", "entropy", "log_loss"]
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
            self.clf_workflow = DecisionTreeClassification(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        elif self.model == "Random Forest":
            self.clf_workflow = RandomForestClassification()
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
            self.clf_workflow = XgboostClassification(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, alpha=alpha, lambd=lambd)
        elif self.model == "Logistic Regression":
            self.clf_workflow = LogisticRegressionClassification()

        self.clf_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.clf_workflow.fit(X_train, y_train)
        y_test_predict = self.clf_workflow.predict(X_test)
        y_test_predict = self.clf_workflow.np2pd(y_test_predict, y_test.columns)
        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test,
                                      y_train=y_train, y_test=y_test, y_test_predict=y_test_predict)

        # Common components for every classification algorithm
        self.clf_workflow.common_components()

        # Special components of different algorithms
        self.clf_workflow.special_components()

        # Save the prediction result
        self.clf_workflow.data_save(y_test_predict, "Y Test Predict", DATASET_OUTPUT_PATH, "Model Prediction")

        # Save the trained model
        self.clf_workflow.save_model()

    @dispatch(object, object, object, object, object, object, bool)
    def activate(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.DataFrame, y_test: pd.DataFrame, is_automl: bool) -> None:
        """Train by FLAML framework + RAY framework."""

        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
        if self.model == "Support Vector Machine":
            self.clf_workflow = SVMClassification()
        elif self.model == "Decision Tree":
            self.clf_workflow = DecisionTreeClassification()
        elif self.model == "Random Forest":
            self.clf_workflow = RandomForestClassification()
        elif self.model == "Xgboost":
            self.clf_workflow = XgboostClassification()
        elif self.model == "Logistic Regression":
            self.clf_workflow = LogisticRegressionClassification()

        self.clf_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.clf_workflow.fit(X_train, y_train, is_automl)
        y_test_predict = self.clf_workflow.predict(X_test, is_automl)
        y_test_predict = self.clf_workflow.np2pd(y_test_predict, y_test.columns)
        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test,
                                      y_train=y_train, y_test=y_test, y_test_predict=y_test_predict)

        # Common components for every classification algorithm
        self.clf_workflow.common_components(is_automl)

        # Special components of different algorithms
        self.clf_workflow.special_components(is_automl)

        # Save the prediction result
        self.clf_workflow.data_save(y_test_predict, "y test predict", DATASET_OUTPUT_PATH, "Model Prediction")

        # Save the trained model
        self.clf_workflow.save_model(is_automl)