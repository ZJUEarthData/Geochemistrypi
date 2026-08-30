# -*- coding: utf-8 -*-
import os
from typing import Dict

import pandas as pd
from multipledispatch import dispatch
from rich import print

from ...scientific_execution import active_scientific_execution, save_scientific_execution_attestation
from ..constants import MLFLOW_ARTIFACT_DATA_PATH, SECTION
from ..data.data_readiness import num_input
from ..model.regression import (
    BayesianRidgeRegression,
    ClassicalLinearRegression,
    DecisionTreeRegression,
    ElasticNetRegression,
    ExtraTreesRegression,
    GradientBoostingRegression,
    KNNRegression,
    LassoRegression,
    MLPRegression,
    PolynomialRegression,
    RandomForestRegression,
    RegressionWorkflowBase,
    RidgeRegression,
    SGDRegression,
    SVMRegression,
    XGBoostRegression,
)
from ._base import ModelSelectionBase


class RegressionModelSelection(ModelSelectionBase):
    """Simulate the normal way of training regression algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.reg_workflow = RegressionWorkflowBase()
        self.transformer_config = {}
        self.scientific_execution = active_scientific_execution()

    def _constructor_parameters(self, method: str, legacy: Dict) -> Dict:
        if self.scientific_execution is None:
            return legacy
        return self.scientific_execution.constructor_parameters(method, legacy)

    @dispatch(object, object, object, object, object, object, object, object, object)
    def activate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        name_train: pd.Series,
        name_test: pd.Series,
        name_all: pd.Series,
    ) -> None:
        """Train by Scikit-learn framework."""

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name_train=name_train, name_test=name_test)
        # Model option
        if self.model_name == "Polynomial Regression":
            hyper_parameters = PolynomialRegression.manual_hyper_parameters()
            self.reg_workflow = PolynomialRegression(
                degree=hyper_parameters["degree"],
                interaction_only=hyper_parameters["interaction_only"],
                include_bias=hyper_parameters["include_bias"],
            )
            poly_config, X_train, X_test = self.reg_workflow.poly(X_train, X_test)
            self.transformer_config.update(poly_config)
            self.reg_workflow.data_upload(X_train=X_train, X_test=X_test)
        elif self.model_name == "XGBoost":
            hyper_parameters = XGBoostRegression.manual_hyper_parameters()
            self.reg_workflow = XGBoostRegression(
                **self._constructor_parameters(
                    "xgboost",
                    {
                        "n_estimators": hyper_parameters["n_estimators"],
                        "learning_rate": hyper_parameters["learning_rate"],
                        "max_depth": hyper_parameters["max_depth"],
                        "subsample": hyper_parameters["subsample"],
                        "colsample_bytree": hyper_parameters["colsample_bytree"],
                        "reg_alpha": hyper_parameters["alpha"],
                        "reg_lambda": hyper_parameters["lambd"],
                    },
                )
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
                **self._constructor_parameters(
                    "extra_trees",
                    {
                        "n_estimators": hyper_parameters["n_estimators"],
                        "max_depth": hyper_parameters["max_depth"],
                        "min_samples_split": hyper_parameters["min_samples_split"],
                        "min_samples_leaf": hyper_parameters["min_samples_leaf"],
                        "max_features": hyper_parameters["max_features"],
                        "bootstrap": hyper_parameters["bootstrap"],
                        "oob_score": hyper_parameters["oob_score"],
                        "max_samples": hyper_parameters["max_samples"],
                    },
                )
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
                max_samples=hyper_parameters["max_samples"],
            )
        elif self.model_name == "Support Vector Machine":
            hyper_parameters = SVMRegression.manual_hyper_parameters()
            self.reg_workflow = SVMRegression(
                kernel=hyper_parameters["kernel"],
                degree=hyper_parameters["degree"],
                gamma=hyper_parameters["gamma"],
                C=hyper_parameters["C"],
                shrinking=hyper_parameters["shrinking"],
            )
        elif self.model_name == "Multi-layer Perceptron":
            hyper_parameters = MLPRegression.manual_hyper_parameters()
            self.reg_workflow = MLPRegression(
                hidden_layer_sizes=hyper_parameters["hidden_layer_sizes"],
                activation=hyper_parameters["activation"],
                solver=hyper_parameters["solver"],
                alpha=hyper_parameters["alpha"],
                learning_rate=hyper_parameters["learning_rate"],
                max_iter=hyper_parameters["max_iter"],
            )
        elif self.model_name == "Linear Regression":
            hyper_parameters = ClassicalLinearRegression.manual_hyper_parameters()
            self.reg_workflow = ClassicalLinearRegression(fit_intercept=hyper_parameters["fit_intercept"])
        elif self.model_name == "K-Nearest Neighbors":
            hyper_parameters = KNNRegression.manual_hyper_parameters()
            self.reg_workflow = KNNRegression(
                n_neighbors=hyper_parameters["n_neighbors"],
                weights=hyper_parameters["weights"],
                algorithm=hyper_parameters["algorithm"],
                leaf_size=hyper_parameters["leaf_size"],
                p=hyper_parameters["p"],
                metric=hyper_parameters["metric"],
            )
        elif self.model_name == "Gradient Boosting":
            hyper_parameters = GradientBoostingRegression.manual_hyper_parameters()
            self.reg_workflow = GradientBoostingRegression(
                n_estimators=hyper_parameters["n_estimators"],
                learning_rate=hyper_parameters["learning_rate"],
                max_depth=hyper_parameters["max_depth"],
                min_samples_split=hyper_parameters["min_samples_split"],
                min_samples_leaf=hyper_parameters["min_samples_leaf"],
                max_features=hyper_parameters["max_features"],
                subsample=hyper_parameters["subsample"],
                loss=hyper_parameters["loss"],
            )
        elif self.model_name == "Lasso Regression":
            hyper_parameters = LassoRegression.manual_hyper_parameters()
            self.reg_workflow = LassoRegression(
                alpha=hyper_parameters["alpha"],
                fit_intercept=hyper_parameters["fit_intercept"],
                max_iter=hyper_parameters["max_iter"],
                tol=hyper_parameters["tol"],
                selection=hyper_parameters["selection"],
            )
        elif self.model_name == "Elastic Net":
            hyper_parameters = ElasticNetRegression.manual_hyper_parameters()
            self.reg_workflow = ElasticNetRegression(
                alpha=hyper_parameters["alpha"],
                l1_ratio=hyper_parameters["l1_ratio"],
                fit_intercept=hyper_parameters["fit_intercept"],
                max_iter=hyper_parameters["max_iter"],
                tol=hyper_parameters["tol"],
                selection=hyper_parameters["selection"],
            )
        elif self.model_name == "SGD Regression":
            hyper_parameters = SGDRegression.manual_hyper_parameters()
            self.reg_workflow = SGDRegression(
                loss=hyper_parameters["loss"],
                penalty=hyper_parameters["penalty"],
                alpha=hyper_parameters["alpha"],
                l1_ratio=hyper_parameters["l1_ratio"],
                fit_intercept=hyper_parameters["fit_intercept"],
                max_iter=hyper_parameters["max_iter"],
                tol=hyper_parameters["tol"],
                shuffle=hyper_parameters["shuffle"],
                learning_rate=hyper_parameters["learning_rate"],
                eta0=hyper_parameters["eta0"],
                power_t=hyper_parameters["power_t"],
            )
        elif self.model_name == "BayesianRidge Regression":
            hyper_parameters = BayesianRidgeRegression.manual_hyper_parameters()
            self.reg_workflow = BayesianRidgeRegression(
                tol=hyper_parameters["tol"],
                alpha_1=hyper_parameters["alpha_1"],
                alpha_2=hyper_parameters["alpha_2"],
                lambda_1=hyper_parameters["lambda_1"],
                lambda_2=hyper_parameters["lambda_2"],
                alpha_init=hyper_parameters["alpha_init"],
                lambda_init=hyper_parameters["lambda_init"],
                compute_score=hyper_parameters["compute_score"],
                fit_intercept=hyper_parameters["fit_intercept"],
                copy_X=hyper_parameters["copy_X"],
                verbose=hyper_parameters["verbose"],
            )
        elif self.model_name == "Ridge Regression":
            hyper_parameters = RidgeRegression.manual_hyper_parameters()
            self.reg_workflow = RidgeRegression(
                alpha=hyper_parameters["alpha"],
                fit_intercept=hyper_parameters["fit_intercept"],
                max_iter=hyper_parameters["max_iter"],
                tol=hyper_parameters["tol"],
            )

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train)
        save_scientific_execution_attestation(
            self.reg_workflow.model,
            os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"),
        )
        y_train_predict = self.reg_workflow.predict(X_train)
        # Support multiple Y columns: Make sure the number of columns in the predicted results matches the number of columns in the original Y
        if y_train_predict.ndim == 1 and y_train.shape[1] > 1:
            # If the predicted result is 1D but Y is multi-column, reshape it
            y_train_predict = y_train_predict.reshape(-1, y_train.shape[1])
        y_train_predict = self.reg_workflow.np2pd(y_train_predict, y_train.columns)
        y_train_predict = y_train_predict.dropna()
        y_train_predict = y_train_predict.reset_index(drop=True)
        self.reg_workflow.data_upload(y_train_predict=y_train_predict)
        external_labeled_evaluation = self.scientific_execution is not None and self.scientific_execution.evaluation_mode == "external_labeled"
        if external_labeled_evaluation:
            y_test_predict = pd.DataFrame(columns=y_test.columns)
            self.reg_workflow.data_upload(y_test_predict=y_test_predict)
        else:
            y_test_predict = self.reg_workflow.predict(X_test)
            # Support multiple Y columns: Make sure the number of columns in the predicted results matches the number of columns in the original Y
            if y_test_predict.ndim == 1 and y_test.shape[1] > 1:
                # If the predicted result is 1D but Y is multi-column, reshape it
                y_test_predict = y_test_predict.reshape(-1, y_test.shape[1])
            y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
            y_test_predict = y_test_predict.dropna()
            y_test_predict = y_test_predict.reset_index(drop=True)
            self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        effective_hyper_parameters = self.reg_workflow.model.get_params(deep=False) if self.scientific_execution is not None else hyper_parameters
        self.reg_workflow.save_hyper_parameters(
            effective_hyper_parameters,
            self.model_name,
            os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"),
        )

        # An external labeled cohort replaces the internal holdout.  Only
        # training/CV evidence is valid before the independent evaluation.
        if external_labeled_evaluation:
            self.reg_workflow.training_components()
        else:
            self.reg_workflow.common_components()

        # Special components of different algorithms
        self.reg_workflow.special_components()

        # Save the prediction result
        self.reg_workflow.data_save(y_train_predict, name_train, "Y Train Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Train Prediction")
        if not external_labeled_evaluation:
            self.reg_workflow.data_save(y_test_predict, name_test, "Y Test Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Test Prediction")

        # Save the trained model
        self.reg_workflow.model_save()

    @dispatch(object, object, object, object, object, object, object, object, object, bool)
    def activate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        name_train: pd.Series,
        name_test: pd.Series,
        name_all: pd.Series,
        is_automl: bool,
    ) -> None:
        """Train by FLAML framework + RAY framework."""

        self.reg_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name_train=name_train, name_test=name_test)

        # Model option
        if self.model_name == "Polynomial Regression":
            print("Please specify the maximal degree of the polynomial features.")
            poly_degree = num_input(SECTION[2], "@Degree:")
            self.reg_workflow = PolynomialRegression(degree=poly_degree)
            poly_config, X_train, X_test = self.reg_workflow.poly(X_train, X_test)
            self.transformer_config.update(poly_config)
            self.reg_workflow.data_upload(X_train=X_train, X_test=X_test)
        elif self.model_name == "XGBoost":
            self.reg_workflow = XGBoostRegression()
        elif self.model_name == "Decision Tree":
            self.reg_workflow = DecisionTreeRegression()
        elif self.model_name == "Extra-Trees":
            self.reg_workflow = ExtraTreesRegression()
        elif self.model_name == "Random Forest":
            self.reg_workflow = RandomForestRegression()
        elif self.model_name == "Support Vector Machine":
            self.reg_workflow = SVMRegression()
        elif self.model_name == "Multi-layer Perceptron":
            self.reg_workflow = MLPRegression()
        elif self.model_name == "Linear Regression":
            self.reg_workflow = ClassicalLinearRegression()
        elif self.model_name == "K-Nearest Neighbors":
            self.reg_workflow = KNNRegression()
        elif self.model_name == "Gradient Boosting":
            self.reg_workflow = GradientBoostingRegression()
        elif self.model_name == "Lasso Regression":
            self.reg_workflow = LassoRegression()
        elif self.model_name == "Elastic Net":
            self.reg_workflow = ElasticNetRegression()
        elif self.model_name == "SGD Regression":
            self.reg_workflow = SGDRegression()
        elif self.model_name == "BayesianRidge Regression":
            self.reg_workflow = BayesianRidgeRegression()
        elif self.model_name == "Ridge Regression":
            self.reg_workflow = RidgeRegression()

        self.reg_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.reg_workflow.fit(X_train, y_train, is_automl)
        y_train_predict = self.reg_workflow.predict(X_train, is_automl)
        # Support multiple Y columns: Make sure the number of columns in the predicted results matches the number of columns in the original Y
        if y_train_predict.ndim == 1 and y_train.shape[1] > 1:
            # If the predicted result is 1D but Y is multi-column, reshape it
            y_train_predict = y_train_predict.reshape(-1, y_train.shape[1])
        y_train_predict = self.reg_workflow.np2pd(y_train_predict, y_train.columns)
        y_train_predict = y_train_predict.dropna()
        y_train_predict = y_train_predict.reset_index(drop=True)
        self.reg_workflow.data_upload(y_train_predict=y_train_predict)
        y_test_predict = self.reg_workflow.predict(X_test, is_automl)
        # Support multiple Y columns: Make sure the number of columns in the predicted results matches the number of columns in the original Y
        if y_test_predict.ndim == 1 and y_test.shape[1] > 1:
            # If the predicted result is 1D but Y is multi-column, reshape it
            y_test_predict = y_test_predict.reshape(-1, y_test.shape[1])
        y_test_predict = self.reg_workflow.np2pd(y_test_predict, y_test.columns)
        y_test_predict = y_test_predict.dropna()
        y_test_predict = y_test_predict.reset_index(drop=True)
        self.reg_workflow.data_upload(y_test_predict=y_test_predict)

        # Save the model hyper-parameters
        if self.reg_workflow.ray_best_model is not None:
            self.reg_workflow.save_hyper_parameters(self.reg_workflow.ray_best_model.get_params(), self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))
        else:
            self.reg_workflow.save_hyper_parameters(self.reg_workflow.auto_best_config, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every regression algorithm
        self.reg_workflow.common_components(is_automl)

        # Special components of different algorithms
        self.reg_workflow.special_components(is_automl)

        # Save the prediction result
        self.reg_workflow.data_save(y_train_predict, name_train, "Y Train Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Train Prediction")
        self.reg_workflow.data_save(y_test_predict, name_test, "Y Test Predict", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Model Test Prediction")

        # Save the trained model
        self.reg_workflow.model_save(is_automl)
