# -*- coding: utf-8 -*-
import os

import pandas as pd
from multipledispatch import dispatch

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..model.classification import (
    ClassificationWorkflowBase,
    DecisionTreeClassification,
    ExtraTreesClassification,
    LogisticRegressionClassification,
    MLPClassification,
    RandomForestClassification,
    SVMClassification,
    XgboostClassification,
)
from ._base import ModelSelectionBase


class ClassificationModelSelection(ModelSelectionBase):
    """Simulate the normal way of training classification algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.clf_workflow = ClassificationWorkflowBase()

    @dispatch(object, object, object, object, object, object)
    def activate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """Train by Scikit-learn framework."""

        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Sample balance
        X_train, y_train = self.clf_workflow.sample_balance(X_train, y_train, os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH)

        # Model option
        if self.model_name == "Support Vector Machine":
            hyper_parameters = SVMClassification.manual_hyper_parameters()
            self.clf_workflow = SVMClassification(
                kernel=hyper_parameters["kernel"],
                degree=hyper_parameters["degree"],
                gamma=hyper_parameters["gamma"],
                C=hyper_parameters["C"],
                shrinking=hyper_parameters["shrinking"],
            )
        elif self.model_name == "Decision Tree":
            hyper_parameters = DecisionTreeClassification.manual_hyper_parameters()
            self.clf_workflow = DecisionTreeClassification(
                criterion=hyper_parameters["criterion"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
            )
        elif self.model_name == "Random Forest":
            hyper_parameters = RandomForestClassification.manual_hyper_parameters()
            self.clf_workflow = RandomForestClassification(
                n_estimators=hyper_parameters["n_estimators"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
                bootstrap=hyper_parameters["bootstrap"],
                oob_score=hyper_parameters["oob_score"],
                max_samples=hyper_parameters["max_samples"],
            )
        elif self.model_name == "Xgboost":
            hyper_parameters = XgboostClassification.manual_hyper_parameters()
            self.clf_workflow = XgboostClassification(
                n_estimators=hyper_parameters["n_estimators"],
                learning_rate=hyper_parameters["learning_rate"],
                max_depth=hyper_parameters["max_depth"],
                subsample=hyper_parameters["subsample"],
                colsample_bytree=hyper_parameters["colsample_bytree"],
                alpha=hyper_parameters["alpha"],
                lambd=hyper_parameters["lambd"],
            )
        elif self.model_name == "Logistic Regression":
            hyper_parameters = LogisticRegressionClassification.manual_hyper_parameters()
            self.clf_workflow = LogisticRegressionClassification(
                penalty=hyper_parameters["penalty"],
                C=hyper_parameters["C"],
                solver=hyper_parameters["solver"],
                max_iter=hyper_parameters["max_iter"],
                class_weight=hyper_parameters["class_weight"],
                l1_ratio=hyper_parameters["l1_ratio"],
            )
        elif self.model_name == "Multi-layer Perceptron":
            hyper_parameters = MLPClassification.manual_hyper_parameters()
            self.clf_workflow = MLPClassification(
                hidden_layer_sizes=hyper_parameters["hidden_layer_sizes"],
                activation=hyper_parameters["activation"],
                solver=hyper_parameters["solver"],
                alpha=hyper_parameters["alpha"],
                learning_rate=hyper_parameters["learning_rate"],
                max_iter=hyper_parameters["max_iter"],
            )
        elif self.model_name == "Extra-Trees":
            hyper_parameters = ExtraTreesClassification.manual_hyper_parameters()
            self.clf_workflow = ExtraTreesClassification(
                n_estimators=hyper_parameters["n_estimators"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
                bootstrap=hyper_parameters["bootstrap"],
                oob_score=hyper_parameters["oob_score"],
                max_samples=hyper_parameters["max_samples"],
            )

        self.clf_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.clf_workflow.fit(X_train, y_train)
        y_test_predict = self.clf_workflow.predict(X_test)
        y_test_predict = self.clf_workflow.np2pd(y_test_predict, y_test.columns)
        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        self.clf_workflow.save_hyper_parameters(hyper_parameters, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every classification algorithm
        self.clf_workflow.common_components()

        # Special components of different algorithms
        self.clf_workflow.special_components()

        # Save the prediction result
        self.clf_workflow.data_save(y_test_predict, "Y Test Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Prediction")

        # Save the trained model
        self.clf_workflow.save_model()

    @dispatch(object, object, object, object, object, object, bool)
    def activate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        is_automl: bool,
    ) -> None:
        """Train by FLAML framework + RAY framework."""

        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Sample balance
        X_train, y_train = self.clf_workflow.sample_balance(X_train, y_train, os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH)

        # Model option
        if self.model_name == "Support Vector Machine":
            self.clf_workflow = SVMClassification()
        elif self.model_name == "Decision Tree":
            self.clf_workflow = DecisionTreeClassification()
        elif self.model_name == "Random Forest":
            self.clf_workflow = RandomForestClassification()
        elif self.model_name == "Xgboost":
            self.clf_workflow = XgboostClassification()
        elif self.model_name == "Logistic Regression":
            self.clf_workflow = LogisticRegressionClassification()
        elif self.model_name == "Multi-layer Perceptron":
            self.clf_workflow = MLPClassification()
        elif self.model_name == "Extra-Trees":
            self.clf_workflow = ExtraTreesClassification()

        self.clf_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.clf_workflow.fit(X_train, y_train, is_automl)
        y_test_predict = self.clf_workflow.predict(X_test, is_automl)
        y_test_predict = self.clf_workflow.np2pd(y_test_predict, y_test.columns)
        self.clf_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        if self.clf_workflow.ray_best_model is not None:
            self.clf_workflow.save_hyper_parameters(self.clf_workflow.ray_best_model.get_params(), self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))
        else:
            self.clf_workflow.save_hyper_parameters(self.clf_workflow.automl.best_config, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every classification algorithm
        self.clf_workflow.common_components(is_automl)

        # Special components of different algorithms
        self.clf_workflow.special_components(is_automl)

        # Save the prediction result
        self.clf_workflow.data_save(y_test_predict, "y test predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Prediction")

        # Save the trained model
        self.clf_workflow.save_model(is_automl)
