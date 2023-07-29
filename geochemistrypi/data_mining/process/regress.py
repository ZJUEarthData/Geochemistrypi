# -*- coding: utf-8 -*-
import os

import pandas as pd
from multipledispatch import dispatch
from rich import print

from ..constants import MLFLOW_ARTIFACT_DATA_PATH, SECTION
from ..data.data_readiness import num_input
from ..model.regression import (
    ClassicalLinearRegression,
    DecisionTreeRegression,
    DNNRegression,
    ExtraTreesRegression,
    PolynomialRegression,
    RandomForestRegression,
    RegressionWorkflowBase,
    SVMRegression,
    XgboostRegression,
)
from ._base import ModelSelectionBase


class RegressionModelSelection(ModelSelectionBase):
    """Simulate the normal way of training regression algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.reg_workflow = RegressionWorkflowBase()

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

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
        if self.model_name == "Polynomial Regression":
            hyper_parameters = PolynomialRegression.manual_hyper_parameters()
            self.reg_workflow = PolynomialRegression(
                degree=hyper_parameters["degree"],
                interaction_only=hyper_parameters["interaction_only"],
                include_bias=hyper_parameters["include_bias"],
            )
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
            self.reg_workflow.data_upload(X_train=X_train, X_test=X_test)
        elif self.model_name == "Xgboost":
            hyper_parameters = XgboostRegression.manual_hyper_parameters()
            self.reg_workflow = XgboostRegression(
                n_estimators=hyper_parameters["n_estimators"],
                learning_rate=hyper_parameters["learning_rate"],
                max_depth=hyper_parameters["max_depth"],
                subsample=hyper_parameters["subsample"],
                colsample_bytree=hyper_parameters["colsample_bytree"],
                alpha=hyper_parameters["alpha"],
                lambd=hyper_parameters["lambd"],
            )
        elif self.model_name == "Decision Tree":
            hyper_parameters = DecisionTreeRegression.manual_hyper_parameters()
            self.reg_workflow = DecisionTreeRegression(
                criterion=hyper_parameters["criterion"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
            )
        elif self.model_name == "Extra-Trees":
            hyper_parameters = ExtraTreesRegression.manual_hyper_parameters()
            self.reg_workflow = ExtraTreesRegression(
                n_estimators=hyper_parameters["n_estimators"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
                bootstrap=hyper_parameters["bootstrap"],
                oob_score=hyper_parameters["oob_score"],
            )
        elif self.model_name == "Random Forest":
            hyper_parameters = RandomForestRegression.manual_hyper_parameters()
            self.reg_workflow = RandomForestRegression(
                n_estimators=hyper_parameters["n_estimators"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
                bootstrap=hyper_parameters["bootstrap"],
                oob_score=hyper_parameters["oob_score"],
            )
        elif self.model_name == "Support Vector Machine":
            hyper_parameters = SVMRegression.manual_hyper_parameters()
            if hyper_parameters["kernel"] == "linear":
                self.reg_workflow = SVMRegression(kernel=hyper_parameters["kernel"], C=hyper_parameters["C"], shrinking=hyper_parameters["shrinking"])
            elif hyper_parameters["kernel"] == "poly":
                self.reg_workflow = SVMRegression(
                    kernel=hyper_parameters["kernel"],
                    degree=hyper_parameters["degree"],
                    gamma=hyper_parameters["gamma"],
                    C=hyper_parameters["C"],
                    shrinking=hyper_parameters["shrinking"],
                )
            elif hyper_parameters["kernel"] == "rbf":
                self.reg_workflow = SVMRegression(
                    kernel=hyper_parameters["kernel"],
                    gamma=hyper_parameters["gamma"],
                    C=hyper_parameters["C"],
                    shrinking=hyper_parameters["shrinking"],
                )
            elif hyper_parameters["kernel"] == "sigmoid":
                self.reg_workflow = SVMRegression(
                    kernel=hyper_parameters["kernel"],
                    gamma=hyper_parameters["gamma"],
                    C=hyper_parameters["C"],
                    shrinking=hyper_parameters["shrinking"],
                )
        elif self.model_name == "Deep Neural Network":
            hyper_parameters = DNNRegression.manual_hyper_parameters()
            self.reg_workflow = DNNRegression(
                hidden_layer_sizes=hyper_parameters["hidden_layer_sizes"],
                activation=hyper_parameters["activation"],
                solver=hyper_parameters["solver"],
                alpha=hyper_parameters["alpha"],
                learning_rate=hyper_parameters["learning_rate"],
                max_iter=hyper_parameters["max_iter"],
            )
        elif self.model_name == "Linear Regression":
            hyper_parameters = ClassicalLinearRegression.manual_hyper_parameters()
            self.reg_workflow = ClassicalLinearRegression(fit_intercept=hyper_parameters["fit_intercept"], normalize=hyper_parameters["normalize"])

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train)
        y_test_predict = self.reg_workflow.predict(X_test)
        y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
        self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        self.reg_workflow.save_hyper_parameters(hyper_parameters, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every regression algorithm
        self.reg_workflow.common_components()

        # Special components of different algorithms
        self.reg_workflow.special_components()

        # Save the prediction result
        self.reg_workflow.data_save(y_test_predict, "Y Test Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Prediction")

        # Save the trained model
        self.reg_workflow.save_model()

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

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Model option
        if self.model_name == "Polynomial Regression":
            # TODO(Sany sanyhew1097618435@163.com): Find the proper way for polynomial regression
            print("Please specify the maximal degree of the polynomial features.")
            poly_degree = num_input(SECTION[2], "@Degree:")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            X_train, X_test = self.reg_workflow.poly(X_train, X_test)
        elif self.model_name == "Xgboost":
            self.reg_workflow = XgboostRegression()
        elif self.model_name == "Decision Tree":
            self.reg_workflow = DecisionTreeRegression()
        elif self.model_name == "Extra-Trees":
            self.reg_workflow = ExtraTreesRegression()
        elif self.model_name == "Random Forest":
            self.reg_workflow = RandomForestRegression()
        elif self.model_name == "Support Vector Machine":
            self.reg_workflow = SVMRegression()
        elif self.model_name == "Deep Neural Network":
            self.reg_workflow = DNNRegression()
        elif self.model_name == "Linear Regression":
            self.reg_workflow = ClassicalLinearRegression()

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train, is_automl)
        y_test_predict = self.reg_workflow.predict(X_test, is_automl)
        y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
        self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        if self.reg_workflow.ray_best_model is not None:
            self.reg_workflow.save_hyper_parameters(self.reg_workflow.ray_best_model.get_params(), self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))
        else:
            self.reg_workflow.save_hyper_parameters(self.reg_workflow.automl.best_config, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every regression algorithm
        self.reg_workflow.common_components(is_automl)

        # Special components of different algorithms
        self.reg_workflow.special_components(is_automl)

        # Save the prediction result
        self.reg_workflow.data_save(y_test_predict, "Y Test Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Prediction")

        # Save the trained model
        self.reg_workflow.save_model(is_automl)
