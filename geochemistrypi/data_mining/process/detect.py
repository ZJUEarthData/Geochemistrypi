# -*- coding: utf-8 -*-
import os

import pandas as pd

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..model.detection import AnomalyDetectionWorkflowBase, IsolationForestAnomalyDetection, LocalOutlierFactorAnomalyDetection
from ._base import ModelSelectionBase


class AnomalyDetectionModelSelection(ModelSelectionBase):
    """Simulate the normal way of invoking scikit-learn anomaly detection algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.ad_workflow = AnomalyDetectionWorkflowBase()
        self.transformer_config = {}

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

        self.ad_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name_train=name_train, name_test=name_test, name_all=name_all)

        # Model option
        if self.model_name == "Isolation Forest":
            hyper_parameters = IsolationForestAnomalyDetection.manual_hyper_parameters()
            self.ad_workflow = IsolationForestAnomalyDetection(
                n_estimators=hyper_parameters["n_estimators"],
                contamination=hyper_parameters["contamination"],
                max_features=hyper_parameters["max_features"],
                bootstrap=hyper_parameters["bootstrap"],
                max_samples=hyper_parameters["max_samples"],
            )

        if self.model_name == "Local Outlier Factor":
            hyper_parameters = LocalOutlierFactorAnomalyDetection.manual_hyper_parameters()
            self.ad_workflow = LocalOutlierFactorAnomalyDetection(
                n_neighbors=hyper_parameters["n_neighbors"],
                contamination=hyper_parameters["contamination"],
                leaf_size=hyper_parameters["leaf_size"],
                n_jobs=hyper_parameters["n_jobs"],
                p=hyper_parameters["p"],
            )

        self.ad_workflow.show_info()

        # Use Scikit-learn style API to process input data
        self.ad_workflow.fit(X)
        y_predict = self.ad_workflow.predict(X)
        X_anomaly_detection, X_normal, X_abnormal, name_normal, name_abnormal = self.ad_workflow._detect_data(X, name_all, y_predict)

        self.ad_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name_train=name_train, name_test=name_test, name_all=name_all)

        # Save the model hyper-parameters
        self.ad_workflow.save_hyper_parameters(hyper_parameters, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # Common components for every anomaly detection algorithm
        self.ad_workflow.common_components()

        # special components of different algorithms
        self.ad_workflow.special_components()

        # Save abnormal detection result
        self.ad_workflow.data_save(X_anomaly_detection, name_all, "X Abnormal Detection", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Abnormal Detection Data")
        self.ad_workflow.data_save(X_normal, name_normal, "X Normal", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Normal Data")
        self.ad_workflow.data_save(X_abnormal, name_abnormal, "X Abnormal", os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"), MLFLOW_ARTIFACT_DATA_PATH, "Abnormal Data")

        # Save the trained model
        self.ad_workflow.model_save()
