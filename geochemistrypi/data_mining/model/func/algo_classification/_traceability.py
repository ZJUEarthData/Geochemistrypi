# -*- coding: utf-8 -*-
import json
import os
from typing import Optional

import pandas as pd

from ....utils.base import save_data, save_data_without_data_identifier, save_text


def _safe_json_value(value):
    if value == float("inf"):
        return "Infinity"
    if value == -float("inf"):
        return "-Infinity"
    if isinstance(value, dict):
        return {str(key): _safe_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return value


def save_target_transform_configuration(label_config: Optional[dict], metric_average: Optional[str], local_path: str, mlflow_path: Optional[str] = None) -> None:
    if not label_config or not local_path:
        return
    config = dict(label_config)
    if metric_average:
        config["metric_average"] = metric_average
    save_text(json.dumps(_safe_json_value(config), indent=4), "Classification Target Traceability", local_path, mlflow_path)


def save_class_counts(y_train: pd.DataFrame, y_test: pd.DataFrame, local_path: str, mlflow_path: Optional[str] = None) -> None:
    if not local_path:
        return
    train_counts = y_train.iloc[:, 0].value_counts().sort_index().rename_axis("encoded_label").reset_index(name="count")
    test_counts = y_test.iloc[:, 0].value_counts().sort_index().rename_axis("encoded_label").reset_index(name="count")
    save_data_without_data_identifier(train_counts, "Y Train Class Counts", local_path, mlflow_path)
    save_data_without_data_identifier(test_counts, "Y Test Class Counts", local_path, mlflow_path)


def decode_predictions(predictions: pd.DataFrame, label_config: Optional[dict]) -> pd.DataFrame:
    decoded = predictions.copy()
    if not label_config:
        return decoded
    code_to_label = label_config.get("code_to_custom_label", {})
    pred_col = decoded.columns[0]
    decoded[f"{pred_col}_decoded"] = decoded[pred_col].map(lambda value: code_to_label.get(str(int(value)), code_to_label.get(str(value), value)))
    return decoded


def save_decoded_predictions(predictions: pd.DataFrame, name_column: pd.Series, label_config: Optional[dict], df_name: str, local_path: str, mlflow_path: Optional[str] = None) -> None:
    if not local_path or not label_config:
        return
    decoded = decode_predictions(predictions, label_config)
    save_data(decoded, name_column, df_name, local_path, mlflow_path)


def save_metric_configuration(algorithm_name: str, metric_average: Optional[str], local_path: str, mlflow_path: Optional[str] = None) -> None:
    if not local_path:
        return
    config = {
        "algorithm": algorithm_name,
        "metric_average": metric_average or "binary_or_default",
    }
    save_text(json.dumps(config, indent=4), f"Metric Configuration - {algorithm_name}", local_path, mlflow_path)


def save_skipped_binary_plot_notice(algorithm_name: str, class_count: int, local_path: str, mlflow_path: Optional[str] = None) -> None:
    if not local_path:
        return
    notice = (
        "Binary-only classification plots were skipped.\n\n"
        f"Algorithm: {algorithm_name}\n"
        f"Class count: {class_count}\n\n"
        "Skipped plots:\n"
        "- Precision-Recall Curve\n"
        "- Precision-Recall Threshold Diagram\n"
        "- ROC Curve\n\n"
        "Reason:\n"
        "The current implementation of these plots supports binary classification only.\n"
        "For multiclass classification, use confusion matrix, classification report, cross validation,\n"
        "and permutation/feature importance outputs.\n"
    )
    save_text(notice, f"Skipped Binary Plots - {algorithm_name}", local_path, mlflow_path)
