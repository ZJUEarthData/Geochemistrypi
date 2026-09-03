# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import xgboost
from flaml import AutoML
from multipledispatch import dispatch
from rich import print
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ...scientific_execution import active_scientific_execution, semantic_label_identity
from ..constants import CUSTOMIZE_LABEL_STRATEGY, MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH, OPTION, RAY_FLAML, SAMPLE_BALANCE_STRATEGY, SECTION
from ..data.data_readiness import limit_num_input, num2option, num_input
from ..plot.statistic_plot import basic_statistic
from ..utils.base import clear_output, save_data, save_data_without_data_identifier, save_fig, save_text
from ._base import LinearWorkflowMixin, TreeWorkflowMixin, WorkflowBase
from .func.algo_classification._adaboost import adaboost_manual_hyper_parameters
from .func.algo_classification._common import (
    cross_validation,
    plot_2d_decision_boundary,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    plot_precision_recall,
    plot_precision_recall_threshold,
    plot_ROC,
    resampler,
    score,
)
from .func.algo_classification._decision_tree import decision_tree_manual_hyper_parameters
from .func.algo_classification._enum import (
    AdaBoostSpecialFunction,
    ClassificationCommonFunction,
    DecisionTreeSpecialFunction,
    ExtraTreesSpecialFunction,
    GradientBoostingSpecialFunction,
    LogisticRegressionSpecialFunction,
    MLPSpecialFunction,
    RandomForestSpecialFunction,
    SGDSpecialFunction,
    XGBoostSpecialFunction,
)
from .func.algo_classification._extra_trees import extra_trees_manual_hyper_parameters
from .func.algo_classification._gradient_boosting import gradient_boosting_manual_hyper_parameters
from .func.algo_classification._knn import knn_manual_hyper_parameters
from .func.algo_classification._logistic_regression import logistic_regression_manual_hyper_parameters, plot_logistic_importance
from .func.algo_classification._multi_layer_perceptron import multi_layer_perceptron_manual_hyper_parameters
from .func.algo_classification._rf import random_forest_manual_hyper_parameters
from .func.algo_classification._sgd_classification import sgd_classificaiton_manual_hyper_parameters
from .func.algo_classification._svc import svc_manual_hyper_parameters
from .func.algo_classification._traceability import save_metric_configuration, save_skipped_binary_plot_notice
from .func.algo_classification._xgboost import xgboost_manual_hyper_parameters


def _configured_cross_validation_folds() -> int:
    contract = active_scientific_execution()
    return contract.cross_validation_folds if contract is not None else 10


class ClassificationWorkflowBase(WorkflowBase):
    """The base workflow class of classification algorithms."""

    common_function = [func.value for func in ClassificationCommonFunction]

    def __init__(self) -> None:
        super().__init__()
        # These two attributes are used for the customized models of FLAML framework
        self.customized = False
        self.customized_name = None
        self.mode = "Classification"

    @dispatch(object, object)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        """Fit the model by Scikit-learn framework."""
        self.model.fit(X, y)

    @dispatch(object, object, bool)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, is_automl: bool = False) -> None:
        """Fit the model by FLAML framework."""
        self._fit_automl(X, y)

    def _fit_automl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit the configured classifier through the shared AutoML workflow."""
        if self.naming not in RAY_FLAML:
            self.automl = AutoML()
            if self.customized:  # When the model is not built-in in FLAML framwork
                self.automl.add_learner(learner_name=self.customized_name, learner_class=self.customization)
            if y.shape[1] == 1:  # FLAML's data format validation mechanism
                y = y.squeeze()  # Convert a single dataFrame column into a series
            self.automl.fit(X_train=X, y_train=y, **self._prepare_automl_settings(self.settings))
        else:
            # When the model is not built-in in FLAML framework, use RAY + FLAML customization.
            self.ray_tune(
                ClassificationWorkflowBase.X_train,
                ClassificationWorkflowBase.X_test,
                ClassificationWorkflowBase.y_train,
                ClassificationWorkflowBase.y_test,
            )
            self.ray_best_model.fit(X, y)

    @dispatch(object)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Perform classification on samples in X by Scikit-learn framework."""
        y_predict = self.model.predict(X)
        return y_predict

    @dispatch(object, bool)
    def predict(self, X: pd.DataFrame, is_automl: bool = False) -> np.ndarray:
        """Perform classification on samples in X by FLAML framework."""
        if self.naming not in RAY_FLAML:
            y_predict = self.automl.predict(X)
            return y_predict
        else:
            y_predict = self.ray_best_model.predict(X)
            return y_predict

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        return dict()

    @property
    def customization(self) -> object:
        """The customized model of FLAML framework."""
        return object

    @property
    def auto_model(self) -> object:
        """Get AutoML trained model by FLAML framework."""
        if self.naming not in RAY_FLAML:
            from sklearn.multioutput import MultiOutputRegressor

            if isinstance(self.automl, MultiOutputRegressor):
                return self.automl
            return self.automl.model.estimator
        else:
            return self.ray_best_model

    @property
    def auto_best_config(self) -> Dict:
        """Get the best AutoML hyper-parameter configuration."""
        if self.naming not in RAY_FLAML:
            from sklearn.multioutput import MultiOutputRegressor

            if isinstance(self.automl, MultiOutputRegressor):
                estimators = getattr(self.automl, "estimators_", None)
                if estimators:
                    best_configs = [getattr(est, "best_config", None) for est in estimators]
                    best_configs = [config for config in best_configs if config is not None]
                    if len(best_configs) == 1:
                        return best_configs[0]
                    if best_configs:
                        return {"best_config_per_output": best_configs}
                return {}
            return self.automl.best_config
        return {}

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        return dict()

    @staticmethod
    def _score(
        y_true: pd.DataFrame,
        y_predict: pd.DataFrame,
        algorithm_name: str,
        store_path: str,
        func_name: str,
        average: Optional[str] = None,
        interactive: bool = True,
        metric_configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Print the classification score report of the model."""
        print(f"-----* {func_name} *-----")
        average, scores = score(
            y_true,
            y_predict,
            average=average,
            interactive=interactive,
            metric_configuration=metric_configuration,
        )
        scores_str = json.dumps(scores, indent=4)
        save_text(scores_str, f"{func_name} - {algorithm_name}", store_path)
        mlflow.log_metrics({key: value for key, value in scores.items() if isinstance(value, (int, float))})
        return average

    @staticmethod
    def _classification_report(y_true: pd.DataFrame, y_predict: pd.DataFrame, algorithm_name: str, store_path: str, func_name: str) -> None:
        """Print the classification report of the model."""
        print(f"-----* {func_name} *-----")
        print(classification_report(y_true, y_predict, zero_division=0))
        scores = classification_report(y_true, y_predict, output_dict=True, zero_division=0)
        scores_str = json.dumps(scores, indent=4)
        save_text(scores_str, f"{func_name} - {algorithm_name}", store_path)
        mlflow.log_artifact(os.path.join(store_path, f"{func_name} - {algorithm_name}.txt"))

    @staticmethod
    def _cross_validation(
        trained_model: object,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        graph_name: str,
        average: str,
        cv_num: int,
        algorithm_name: str,
        store_path: str,
        metric_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Perform cross validation on the model."""
        print(f"-----* {graph_name} *-----")
        print(f"K-Folds: {cv_num}")
        scores = cross_validation(
            trained_model,
            X_train,
            y_train,
            average=average,
            cv_num=cv_num,
            metric_configuration=metric_configuration,
        )
        scores_str = json.dumps(scores, indent=4)
        save_text(scores_str, f"{graph_name} - {algorithm_name}", store_path)

    @staticmethod
    def _plot_confusion_matrix(
        y_test: pd.DataFrame, y_test_predict: pd.DataFrame, name_column: str, graph_name: str, trained_model: object, algorithm_name: str, local_path: str, mlflow_path: str
    ) -> None:
        """Plot the confusion matrix of the model."""
        print(f"-----* {graph_name} *-----")
        data = plot_confusion_matrix(y_test, y_test_predict, trained_model)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        labels = getattr(trained_model, "classes_", None)
        if labels is None or len(labels) != data.shape[0]:
            labels = pd.unique(pd.concat([pd.Series(np.ravel(y_test)), pd.Series(np.ravel(y_test_predict))], ignore_index=True))
        if len(labels) != data.shape[0]:
            labels = range(data.shape[0])
        index = [f"true_{label}" for label in labels]
        columns = [f"pred_{label}" for label in labels]
        data = pd.DataFrame(data, columns=columns, index=index)
        save_data(data, name_column, f"{graph_name} - {algorithm_name}", local_path, mlflow_path, True)

    @staticmethod
    def _plot_normalized_confusion_matrix(
        y_test: pd.DataFrame,
        y_test_predict: pd.DataFrame,
        name_column: str,
        trained_model: object,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
        normalization: str,
    ) -> None:
        graph_name = f"Normalized Confusion Matrix ({normalization})"
        print(f"-----* {graph_name} *-----")
        data = plot_normalized_confusion_matrix(
            y_test,
            y_test_predict,
            trained_model,
            normalization,
        )
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        labels = getattr(trained_model, "classes_", None)
        if labels is None or len(labels) != data.shape[0]:
            labels = pd.unique(
                pd.concat(
                    [pd.Series(np.ravel(y_test)), pd.Series(np.ravel(y_test_predict))],
                    ignore_index=True,
                )
            )
        if len(labels) != data.shape[0]:
            labels = range(data.shape[0])
        normalized = pd.DataFrame(
            data,
            columns=[f"pred_{label}" for label in labels],
            index=[f"true_{label}" for label in labels],
        )
        save_data(
            normalized,
            name_column,
            f"{graph_name} - {algorithm_name}",
            local_path,
            mlflow_path,
            True,
        )

    @staticmethod
    def _count_unique_labels(y_values: Optional[pd.DataFrame]) -> Optional[int]:
        if y_values is None:
            return None
        labels = pd.Series(np.ravel(y_values)).dropna()
        if labels.empty:
            return None
        return int(labels.nunique())

    @classmethod
    def _get_total_class_count(cls, label_config: Optional[Dict[str, Any]] = None) -> int:
        if label_config and label_config.get("num_classes") is not None:
            return int(label_config["num_classes"])
        for y_values in (getattr(cls, "y", None), getattr(cls, "y_train", None), getattr(cls, "y_test", None)):
            class_count = cls._count_unique_labels(y_values)
            if class_count is not None:
                return class_count
        return 0

    @staticmethod
    def _plot_precision_recall(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        name_column: str,
        trained_model: object,
        graph_name: str,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
        metric_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        print(f"-----* {graph_name} *-----")
        y_probs, precisions, recalls, thresholds = plot_precision_recall(
            X_test,
            y_test,
            trained_model,
            graph_name,
            algorithm_name,
            metric_configuration=metric_configuration,
        )
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        y_probs = pd.DataFrame(y_probs, columns=["Probabilities"])
        precisions = pd.DataFrame(precisions, columns=["Precisions"])
        recalls = pd.DataFrame(recalls, columns=["Recalls"])
        thresholds = pd.DataFrame(thresholds, columns=["Thresholds"])
        save_data(precisions, name_column, f"{graph_name} - Precisions", local_path, mlflow_path)
        save_data(recalls, name_column, f"{graph_name} - Recalls", local_path, mlflow_path)

    @staticmethod
    def _plot_precision_recall_threshold(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        name_column: str,
        trained_model: object,
        graph_name: str,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
        metric_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        print(f"-----* {graph_name} *-----")
        y_probs, precisions, recalls, thresholds = plot_precision_recall_threshold(
            X_test,
            y_test,
            trained_model,
            graph_name,
            algorithm_name,
            metric_configuration=metric_configuration,
        )
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        y_probs = pd.DataFrame(y_probs, columns=["Probabilities"])
        precisions = pd.DataFrame(precisions, columns=["Precisions"])
        recalls = pd.DataFrame(recalls, columns=["Recalls"])
        thresholds = pd.DataFrame(thresholds, columns=["Thresholds"])
        save_data(y_probs, name_column, f"{graph_name} - Probabilities", local_path, mlflow_path)
        save_data(precisions, name_column, f"{graph_name} - Precisions", local_path, mlflow_path)
        save_data(recalls, name_column, f"{graph_name} - Recalls", local_path, mlflow_path)
        save_data(thresholds, name_column, f"{graph_name} - Thresholds", local_path, mlflow_path)

    @staticmethod
    def _plot_ROC(
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        name_column: str,
        graph_name: str,
        trained_model: object,
        algorithm_name: str,
        local_path: str,
        mlflow_path: str,
        metric_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        print(f"-----* {graph_name} *-----")
        y_probs, fpr, tpr, thresholds = plot_ROC(
            X_test,
            y_test,
            trained_model,
            graph_name,
            algorithm_name,
            metric_configuration=metric_configuration,
        )
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        y_probs = pd.DataFrame(y_probs, columns=["Probabilities"])
        fpr = pd.DataFrame(fpr, columns=["False Positive Rate"])
        tpr = pd.DataFrame(tpr, columns=["True Positive Rate"])
        thresholds = pd.DataFrame(thresholds, columns=["Thresholds"])
        save_data(y_probs, name_column, f"{graph_name} - Probabilities", local_path, mlflow_path)
        save_data(fpr, name_column, f"{graph_name} - False Positive Rate", local_path, mlflow_path)
        save_data(tpr, name_column, f"{graph_name} - True Positive Rate", local_path, mlflow_path)
        save_data(thresholds, name_column, f"{graph_name} - Thresholds", local_path, mlflow_path)

    @staticmethod
    def _plot_2d_decision_boundary(
        X: pd.DataFrame, X_test: pd.DataFrame, name_column1: str, name_column2: str, trained_model: object, graph_name: str, image_config: dict, algorithm_name: str, local_path: str, mlflow_path: str
    ) -> None:
        """Plot the decision boundary of the trained model with the testing data set below."""
        print(f"-----* {graph_name} *-----")
        plot_2d_decision_boundary(X, X_test, trained_model, image_config)
        save_fig(f"{graph_name} - {algorithm_name}", local_path, mlflow_path)
        save_data(X, name_column1, f"{graph_name} - X", local_path, mlflow_path)
        save_data(X_test, name_column2, f"{graph_name} - X Test", local_path, mlflow_path)

    @staticmethod
    def sample_balance(X_train: pd.DataFrame, y_train: pd.DataFrame, name_column: str, local_path: str, mlflow_path: str) -> tuple:
        """Use this method when the sample size is unbalanced."""
        print("[bold green]-*-*- Sample Balance on Train Set -*-*-[/bold green]")
        num2option(OPTION)
        is_sample_balance = limit_num_input(OPTION, SECTION[1], num_input)
        if is_sample_balance == 1:
            print("Which strategy do you want to apply?")
            num2option(SAMPLE_BALANCE_STRATEGY)
            sample_balance_num = limit_num_input(SAMPLE_BALANCE_STRATEGY, SECTION[1], num_input)
            sample_balance_config, X_train, y_train = resampler(X_train, y_train, SAMPLE_BALANCE_STRATEGY, sample_balance_num - 1)
            train_set_resampled = pd.concat([X_train, y_train], axis=1)
            print("Train Set After Resampling:")
            print(train_set_resampled)
            print("Basic Statistical Information: ")
            basic_statistic(train_set_resampled)
            save_data(X_train, name_column, "X Train After Sample Balance", local_path, mlflow_path)
            save_data(y_train, name_column, "Y Train After Sample Balance", local_path, mlflow_path)
        else:
            sample_balance_config = None
        clear_output()
        return sample_balance_config, X_train, y_train

    @staticmethod
    def customize_label(
        y: pd.DataFrame,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
        name_column1: Optional[pd.Series] = None,
        name_column2: Optional[pd.Series] = None,
        name_column3: Optional[pd.Series] = None,
        local_path: Optional[str] = None,
        mlflow_path: Optional[str] = None,
        label_mapping: Optional[dict] = None,
        interactive: bool = True,
        return_config: bool = False,
    ):
        if not isinstance(y, pd.DataFrame) or y.shape[1] != 1:
            raise ValueError("Classification target Y must be exactly one column.")
        if y.isnull().any().any():
            raise ValueError("Classification target Y contains missing values. Please handle missing labels before training.")

        target_column = y.columns[0]
        y_original = y.copy()

        def _split_semicolon(raw: str) -> List[str]:
            return [item.strip() for item in raw.split(";") if item.strip()]

        def _default_labels(num_classes: int) -> List[str]:
            return [f"Class_{idx}" for idx in range(num_classes)]

        def _parse_labels(num_classes: int) -> List[str]:
            raw = input("Input class labels separated by ';' or press Enter to use Class_0, Class_1, ...\n@Labels: ").strip()
            labels = _split_semicolon(raw) if raw else _default_labels(num_classes)
            if len(labels) != num_classes:
                raise ValueError(f"The number of labels must be {num_classes}, but got {len(labels)}.")
            if len(set(labels)) != len(labels):
                raise ValueError("Class labels must be unique.")
            return labels

        def _ordered_unique(series: pd.Series) -> List[Any]:
            return list(pd.Series(series).drop_duplicates())

        def _final_label_order(y_custom: pd.DataFrame, extra: Optional[dict] = None) -> List[Any]:
            observed_labels = _ordered_unique(y_custom[target_column])
            preferred_labels = (extra or {}).get("label_order")
            if not preferred_labels:
                return observed_labels
            ordered_labels = [label for label in preferred_labels if label in observed_labels]
            ordered_labels.extend(label for label in observed_labels if label not in ordered_labels)
            return ordered_labels

        def _encode_df(df: pd.DataFrame, custom_label_to_code: Dict[Any, int]) -> pd.DataFrame:
            encoded = df.copy()
            encoded[target_column] = encoded[target_column].map(custom_label_to_code)
            if encoded[target_column].isnull().any():
                missing = df.loc[encoded[target_column].isnull(), target_column].drop_duplicates().tolist()
                raise ValueError(f"Some labels were not encoded: {missing}")
            encoded[target_column] = encoded[target_column].astype(int)
            return encoded

        def _build_config(strategy: str, y_custom: pd.DataFrame, extra: Optional[dict] = None) -> dict:
            custom_labels = _final_label_order(y_custom, extra)
            if len(custom_labels) < 2:
                raise ValueError("Classification requires at least two final classes.")
            custom_label_to_code = {label: idx for idx, label in enumerate(custom_labels)}
            class_counts = y_custom[target_column].value_counts(sort=False).to_dict()
            config = {
                "target_transform_version": 2,
                "strategy": strategy,
                "target_column": target_column,
                "num_classes": len(custom_labels),
                "classes": [str(label) for label in custom_labels],
                "custom_label_to_code": {str(label): code for label, code in custom_label_to_code.items()},
                "code_to_custom_label": {str(code): str(label) for label, code in custom_label_to_code.items()},
                "class_counts": {str(label): int(count) for label, count in class_counts.items()},
                "typed_label_records": [
                    {
                        "semantic_label": semantic_label_identity(label),
                        "encoded_label": code,
                        "count": int(class_counts[label]),
                    }
                    for label, code in custom_label_to_code.items()
                ],
            }
            if extra:
                config.update(extra)
            return config

        def _save_outputs(y_custom: pd.DataFrame, y_encoded: pd.DataFrame, config: dict) -> None:
            if not local_path:
                return
            os.makedirs(local_path, exist_ok=True)
            mapping_rows = [
                {
                    "semantic_label_type": record["semantic_label"]["type"],
                    "custom_label": record["semantic_label"]["value"],
                    "encoded_label": record["encoded_label"],
                }
                for record in config["typed_label_records"]
            ]
            mapping_df = pd.DataFrame(mapping_rows)
            counts_df = pd.DataFrame(
                [
                    {
                        "semantic_label_type": record["semantic_label"]["type"],
                        "custom_label": record["semantic_label"]["value"],
                        "count": record["count"],
                    }
                    for record in config["typed_label_records"]
                ]
            )
            save_data(y_original, name_column1, "Y Raw Before Customizing Label", local_path, mlflow_path)
            save_data(y_custom, name_column1, "Y Human-Readable After Customizing Label", local_path, mlflow_path)
            save_data(y_encoded, name_column1, "Y Encoded After Customizing Label", local_path, mlflow_path)
            save_data_without_data_identifier(mapping_df, "Target Label Mapping", local_path, mlflow_path)
            save_data_without_data_identifier(counts_df, "Target Class Counts", local_path, mlflow_path)
            save_text(json.dumps(config, indent=4), "Target Transform Configuration", local_path, mlflow_path)

        def _apply_mapping(mapping: dict, source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
            mapping_type = mapping.get("type")
            y_custom = source.copy()
            extra = {}
            if mapping_type == "interval":
                bins = mapping.get("bins")
                labels = mapping.get("labels")
                if not bins or not labels or len(labels) != len(bins) - 1:
                    raise ValueError("Interval mapping requires bins and exactly len(bins)-1 labels.")
                if not pd.api.types.is_numeric_dtype(source[target_column]):
                    raise ValueError("Interval mapping requires a numeric target column.")
                if any(bins[idx] >= bins[idx + 1] for idx in range(len(bins) - 1)):
                    raise ValueError("Interval bins must be strictly increasing.")
                y_custom[target_column] = pd.cut(source[target_column], bins=bins, labels=labels, include_lowest=True)
                extra = {"bins": bins, "interval_closed": "right", "label_order": labels}
            elif mapping_type == "quantile":
                num_classes = int(mapping.get("num_classes", 2))
                labels = mapping.get("labels") or _default_labels(num_classes)
                if len(labels) != num_classes:
                    raise ValueError("Quantile mapping requires labels to match num_classes.")
                if not pd.api.types.is_numeric_dtype(source[target_column]):
                    raise ValueError("Quantile mapping requires a numeric target column.")
                _, bins = pd.qcut(source[target_column], q=num_classes, retbins=True)
                y_custom[target_column] = pd.cut(source[target_column], bins=bins, labels=labels, include_lowest=True)
                extra = {"bins": [float(value) for value in bins], "interval_closed": "right", "label_order": labels}
            elif mapping_type == "dict":
                mapping_dict = mapping.get("mapping", {})
                observed = set(source[target_column].drop_duplicates())
                missing = [label for label in observed if label not in mapping_dict and str(label) not in mapping_dict]
                if missing:
                    raise ValueError(f"Mapping does not cover all original labels. Missing labels: {missing}")
                y_custom[target_column] = source[target_column].map(lambda value: mapping_dict.get(value, mapping_dict.get(str(value))))
                label_order = []
                for value in mapping_dict.values():
                    if value not in label_order:
                        label_order.append(value)
                extra = {"original_to_custom_label": {str(key): str(value) for key, value in mapping_dict.items()}, "label_order": label_order}
            else:
                raise ValueError(f"Unsupported label mapping type: {mapping_type}")
            if y_custom[target_column].isnull().any():
                raise ValueError("Label customization produced missing labels. Please check bins or mapping rules.")
            return y_custom, extra

        if label_mapping is not None:
            y_custom, extra = _apply_mapping(label_mapping, y)
            strategy = label_mapping.get("type", "custom")
        elif interactive:
            print("[bold green]-*-*- Classification Label Customization -*-*-[/bold green]")
            print("Choose how to define the final classification labels.")
            num2option(CUSTOMIZE_LABEL_STRATEGY)
            strategy_num = limit_num_input(CUSTOMIZE_LABEL_STRATEGY, SECTION[1], num_input)
            y_custom = y.copy()
            extra = {}
            if strategy_num == 1:
                strategy = "encode_original"
            elif strategy_num == 2:
                strategy = "dict"
                original_labels = _ordered_unique(y[target_column])
                print("Original labels:")
                for label in original_labels:
                    print(f"- {label}")
                raw = input("Map every original label with format original:new; original:new\n@Mapping: ").strip()
                mapping_dict = {}
                for item in _split_semicolon(raw):
                    if ":" not in item:
                        raise ValueError("Each mapping item must use original:new format.")
                    original, new = item.split(":", 1)
                    mapping_dict[original.strip()] = new.strip()
                y_custom, extra = _apply_mapping({"type": "dict", "mapping": mapping_dict}, y)
            elif strategy_num == 3:
                strategy = "interval"
                if not pd.api.types.is_numeric_dtype(y[target_column]):
                    raise ValueError("Numeric interval bins require a numeric target column.")
                num_classes = num_input(SECTION[1], "@Number of Classes: ")
                if num_classes < 2:
                    raise ValueError("The number of classes must be at least 2.")
                labels = _parse_labels(num_classes)
                raw_edges = input(f"Input {num_classes - 1} internal cut points separated by ';'. The full data range will be covered automatically.\n@Cut Points: ").strip()
                edges = [float(value) for value in _split_semicolon(raw_edges)]
                if len(edges) != num_classes - 1:
                    raise ValueError(f"Expected {num_classes - 1} cut points, but got {len(edges)}.")
                if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
                    raise ValueError("Cut points must be strictly increasing.")
                bins = [-float("inf")] + edges + [float("inf")]
                y_custom, extra = _apply_mapping({"type": "interval", "bins": bins, "labels": labels}, y)
            elif strategy_num == 4:
                strategy = "quantile"
                if not pd.api.types.is_numeric_dtype(y[target_column]):
                    raise ValueError("Quantile bins require a numeric target column.")
                num_classes = num_input(SECTION[1], "@Number of Classes: ")
                if num_classes < 2:
                    raise ValueError("The number of classes must be at least 2.")
                if y[target_column].nunique() < num_classes:
                    raise ValueError("The number of classes cannot exceed the number of unique target values.")
                labels = _parse_labels(num_classes)
                y_custom, extra = _apply_mapping({"type": "quantile", "num_classes": num_classes, "labels": labels}, y)
            else:
                raise ValueError("Unsupported classification label strategy.")
        else:
            if return_config:
                strategy = "unchanged"
                y_custom = y.copy()
                extra = {}
            else:
                return y, y_train, y_test

        config = _build_config(strategy, y_custom, extra)
        custom_label_to_code = {label: idx for idx, label in enumerate(_final_label_order(y_custom, extra))}
        y_encoded = _encode_df(y_custom, custom_label_to_code)
        _save_outputs(y_custom, y_encoded, config)
        print("------------------------------------")
        print("Original label VS Human-readable label VS Encoded label:")
        y_show = pd.concat([y_original, y_custom, y_encoded], axis=1).drop_duplicates().reset_index(drop=True)
        y_show.columns = ["original_label", "custom_label", "encoded_label"]
        print(y_show)

        if y_train is not None and y_test is not None:
            if label_mapping is not None:
                y_train_custom, _ = _apply_mapping(label_mapping, y_train)
                y_test_custom, _ = _apply_mapping(label_mapping, y_test)
            else:
                y_train_custom = y_train.copy()
                y_test_custom = y_test.copy()
            y_train_encoded = _encode_df(y_train_custom, custom_label_to_code)
            y_test_encoded = _encode_df(y_test_custom, custom_label_to_code)
            if local_path:
                save_data(y_train_encoded, name_column2, "Y Train Encoded After Customizing Label", local_path, mlflow_path)
                save_data(y_test_encoded, name_column3, "Y Test Encoded After Customizing Label", local_path, mlflow_path)
            if return_config:
                return y_encoded, y_train_encoded, y_test_encoded, config
            return y_encoded, y_train_encoded, y_test_encoded

        if return_config:
            return y_encoded, config
        return y_encoded

    @dispatch()
    def common_components(self) -> None:
        """Invoke all common application functions for classification algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        average = self._score(
            y_true=ClassificationWorkflowBase.y_test,
            y_predict=ClassificationWorkflowBase.y_test_predict,
            func_name=ClassificationCommonFunction.MODEL_SCORE.value,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
            average=getattr(self, "metric_average", None),
            interactive=getattr(self, "metric_average", None) is None,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._classification_report(
            y_true=ClassificationWorkflowBase.y_test,
            y_predict=ClassificationWorkflowBase.y_test_predict,
            func_name=ClassificationCommonFunction.CLASSIFICATION_REPORT.value,
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._cross_validation(
            trained_model=self.model,
            X_train=ClassificationWorkflowBase.X_train,
            y_train=ClassificationWorkflowBase.y_train,
            graph_name=ClassificationCommonFunction.CROSS_VALIDATION.value,
            average=average,
            cv_num=_configured_cross_validation_folds(),
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._plot_confusion_matrix(
            y_test=ClassificationWorkflowBase.y_test,
            y_test_predict=ClassificationWorkflowBase.y_test_predict,
            name_column=ClassificationWorkflowBase.name_test,
            trained_model=self.model,
            graph_name=ClassificationCommonFunction.CONFUSION_MATRIX.value,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        scientific_execution = active_scientific_execution()
        if scientific_execution is not None and scientific_execution.confusion_matrix_normalization is not None:
            self._plot_normalized_confusion_matrix(
                y_test=ClassificationWorkflowBase.y_test,
                y_test_predict=ClassificationWorkflowBase.y_test_predict,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.model,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                normalization=scientific_execution.confusion_matrix_normalization,
            )
        class_count = self._get_total_class_count(getattr(self, "label_config", None))
        if class_count == 2:
            self._plot_precision_recall(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.model,
                graph_name=ClassificationCommonFunction.PRECISION_RECALL_CURVE.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
            self._plot_precision_recall_threshold(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.model,
                graph_name=ClassificationCommonFunction.PRECISION_RECALL_THRESHOLD_DIAGRAM.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
            self._plot_ROC(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.model,
                graph_name=ClassificationCommonFunction.ROC_CURVE.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
        else:
            save_skipped_binary_plot_notice(self.naming, class_count, GEOPI_OUTPUT_METRICS_PATH)
        save_metric_configuration(
            self.naming,
            average,
            GEOPI_OUTPUT_METRICS_PATH,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._plot_permutation_importance(
            X_test=ClassificationWorkflowBase.X_test,
            y_test=ClassificationWorkflowBase.y_test,
            name_column=ClassificationWorkflowBase.name_test,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            graph_name=ClassificationCommonFunction.PERMUTATION_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        if ClassificationWorkflowBase.X.shape[1] == 2:
            self._plot_2d_decision_boundary(
                X=ClassificationWorkflowBase.X,
                X_test=ClassificationWorkflowBase.X_test,
                name_column1=ClassificationWorkflowBase.name_all,
                name_column2=ClassificationWorkflowBase.name_test,
                trained_model=self.model,
                image_config=self.image_config,
                graph_name=ClassificationCommonFunction.TWO_DIMENSIONAL_DECISION_BOUNDARY_DIAGRAM.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

    @dispatch(bool)
    def common_components(self, is_automl: bool) -> None:
        """Invoke all common application functions for classification algorithms by FLAML framework."""
        GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        average = self._score(
            y_true=ClassificationWorkflowBase.y_test,
            y_predict=ClassificationWorkflowBase.y_test_predict,
            algorithm_name=self.naming,
            func_name=ClassificationCommonFunction.MODEL_SCORE.value,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
            average=getattr(self, "metric_average", None),
            interactive=getattr(self, "metric_average", None) is None,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._classification_report(
            y_true=ClassificationWorkflowBase.y_test,
            y_predict=ClassificationWorkflowBase.y_test_predict,
            algorithm_name=self.naming,
            func_name=ClassificationCommonFunction.CLASSIFICATION_REPORT.value,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
        )
        self._cross_validation(
            trained_model=self.auto_model,
            X_train=ClassificationWorkflowBase.X_train,
            y_train=ClassificationWorkflowBase.y_train,
            graph_name=ClassificationCommonFunction.CROSS_VALIDATION.value,
            average=average,
            cv_num=_configured_cross_validation_folds(),
            algorithm_name=self.naming,
            store_path=GEOPI_OUTPUT_METRICS_PATH,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._plot_confusion_matrix(
            y_test=ClassificationWorkflowBase.y_test,
            y_test_predict=ClassificationWorkflowBase.y_test_predict,
            name_column=ClassificationWorkflowBase.name_test,
            trained_model=self.auto_model,
            graph_name=ClassificationCommonFunction.CONFUSION_MATRIX.value,
            algorithm_name=self.naming,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        class_count = self._get_total_class_count(getattr(self, "label_config", None))
        if class_count == 2:
            self._plot_precision_recall(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.auto_model,
                graph_name=ClassificationCommonFunction.PRECISION_RECALL_CURVE.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
            self._plot_precision_recall_threshold(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.auto_model,
                graph_name=ClassificationCommonFunction.PRECISION_RECALL_THRESHOLD_DIAGRAM.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
            self._plot_ROC(
                X_test=ClassificationWorkflowBase.X_test,
                y_test=ClassificationWorkflowBase.y_test,
                name_column=ClassificationWorkflowBase.name_test,
                trained_model=self.auto_model,
                graph_name=ClassificationCommonFunction.ROC_CURVE.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
                metric_configuration=getattr(self, "metric_configuration", None),
            )
        else:
            save_skipped_binary_plot_notice(self.naming, class_count, GEOPI_OUTPUT_METRICS_PATH)
        save_metric_configuration(
            self.naming,
            average,
            GEOPI_OUTPUT_METRICS_PATH,
            metric_configuration=getattr(self, "metric_configuration", None),
        )
        self._plot_permutation_importance(
            X_test=ClassificationWorkflowBase.X_test,
            y_test=ClassificationWorkflowBase.y_test,
            name_column=ClassificationWorkflowBase.name_test,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            graph_name=ClassificationCommonFunction.PERMUTATION_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        if ClassificationWorkflowBase.X.shape[1] == 2:
            self._plot_2d_decision_boundary(
                X=ClassificationWorkflowBase.X,
                X_test=ClassificationWorkflowBase.X_test,
                name_column1=ClassificationWorkflowBase.name_all,
                name_column2=ClassificationWorkflowBase.name_test,
                trained_model=self.auto_model,
                image_config=self.image_config,
                graph_name=ClassificationCommonFunction.TWO_DIMENSIONAL_DECISION_BOUNDARY_DIAGRAM.value,
                algorithm_name=self.naming,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )


class SVMClassification(ClassificationWorkflowBase):
    """The automation workflow of using SVC algorithm to make insightful products."""

    name = "Support Vector Machine"
    special_function = []

    def __init__(
        self,
        C: float = 1.0,
        kernel: Union[str, Callable] = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        class_weight: Union[dict, str, None] = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        break_ties: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive. The penalty
            is a squared l2 penalty.

        kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
            If none is given, 'rbf' will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix
            should be an array of shape ``(n_samples, n_samples)``.

        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.

        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

            - if ``gamma='scale'`` (default) is passed then it uses
              1 / (n_features * X.var()) as value of gamma,
            - if 'auto', uses 1 / n_features.

            .. versionchanged:: 0.22
               The default value of ``gamma`` changed from 'auto' to 'scale'.

        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.

        shrinking : bool, default=True
            Whether to use the shrinking heuristic.
            See the :ref:`User Guide <shrinking_svm>`.

        probability : bool, default=False
            Whether to enable probability estimates. This must be enabled prior
            to calling `fit`, will slow down that method as it internally uses
            5-fold cross-validation, and `predict_proba` may be inconsistent with
            `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

        tol : float, default=1e-3
            Tolerance for stopping criterion.

        cache_size : float, default=200
            Specify the size of the kernel cache (in MB).

        class_weight : dict or 'balanced', default=None
            Set the parameter C of class i to class_weight[i]*C for
            SVC. If not given, all classes are supposed to have
            weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.

        verbose : bool, default=False
            Enable verbose output. Note that this setting takes advantage of a
            per-process runtime setting in libsvm that, if enabled, may not work
            properly in a multithreaded context.

        max_iter : int, default=-1
            Hard limit on iterations within solver, or -1 for no limit.

        decision_function_shape : {'ovo', 'ovr'}, default='ovr'
            Whether to return a one-vs-rest ('ovr') decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original
            one-vs-one ('ovo') decision function of libsvm which has shape
            (n_samples, n_classes * (n_classes - 1) / 2). However, note that
            internally, one-vs-one ('ovo') is always used as a multi-class strategy
            to train models; an ovr matrix is only constructed from the ovo matrix.
            The parameter is ignored for binary classification.

            .. versionchanged:: 0.19
                decision_function_shape is 'ovr' by default.

            .. versionadded:: 0.17
               *decision_function_shape='ovr'* is recommended.

            .. versionchanged:: 0.17
               Deprecated *decision_function_shape='ovo' and None*.

        break_ties : bool, default=False
            If true, ``decision_function_shape='ovr'``, and number of classes > 2,
            :term:`predict` will break ties according to the confidence values of
            :term:`decision_function`; otherwise the first class among the tied
            classes is returned. Please note that breaking ties comes at a
            relatively high computational cost compared to a simple predict.

            .. versionadded:: 0.22

        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling the data for
            probability estimates. Ignored when `probability` is False.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        References
        ----------
        Scikit-learn API: sklearn.svm.SVC
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """

        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            verbose=self.verbose,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state,
        )

        self.naming = SVMClassification.name
        self.customized = True
        self.customized_name = "SVC"

    @property
    def settings(self) -> Dict:
        """The configuration of SVC to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SVC of FLAML framework."""
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from flaml.model import SKLearnEstimator
        from sklearn.svm import SVC

        class MySVMClassification(SKLearnEstimator):
            def __init__(self, task="classification", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = SVC

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "C": {"domain": tune.uniform(lower=1, upper=data_size[0]), "init_value": 1, "low_cost_init_value": 1},
                    "kernel": {"domain": tune.choice(["poly", "rbf", "sigmoid"])},
                    "gamma": {"domain": tune.uniform(lower=1e-5, upper=10), "init_value": 1e-1, "low_cost_init_value": 1e-1},
                    "degree": {"domain": tune.quniform(lower=1, upper=5, q=1), "init_value": 3, "low_cost_init_value": 3},
                    "coef0": {"domain": tune.uniform(lower=0, upper=1), "init_value": 0, "low_cost_init_value": 0},
                    "shrinking": {"domain": tune.choice([True, False])},
                    "probability": {"domain": tune.choice([True])},
                }
                return space

        return MySVMClassification

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = svc_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        pass


class DecisionTreeClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Decision Tree algorithm to make insightful products."""

    name = "Decision Tree"
    special_function = [func.value for func in DecisionTreeSpecialFunction]

    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = 3,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Union[dict, List[dict], str, None] = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator. The features are always
            randomly permuted at each split, even if ``splitter`` is set to
            ``"best"``. When ``max_features < n_features``, the algorithm will
            select ``max_features`` at random at each split before finding the best
            split among them. But the best found split may vary across different
            runs, even if ``max_features=n_features``. That is the case, if the
            improvement of the criterion is identical for several splits and one
            split has to be selected at random. To obtain a deterministic behaviour
            during fitting, ``random_state`` has to be fixed to an integer.
            See :term:`Glossary <random_state>` for details.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        class_weight : dict, list of dict or "balanced", default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If None, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.

            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.tree.DecisionTreeClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier
        """

        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )

        self.naming = DecisionTreeClassification.name
        self.customized = True
        self.customized_name = "Decision Tree"

    @property
    def settings(self) -> Dict:
        """The configuration of Decision Tree to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Decision Tree of FLAML framework."""
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from flaml.model import SKLearnEstimator
        from sklearn.tree import DecisionTreeClassifier

        class MyDTClassification(SKLearnEstimator):
            def __init__(self, task="classification", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = DecisionTreeClassifier

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "criterion": {"domain": tune.choice(["gini", "entropy", "log_loss"])},
                    "max_depth": {"domain": tune.randint(lower=2, upper=20), "init_value": 1, "low_cost_init_value": 1},
                    "min_samples_split": {
                        "domain": tune.randint(lower=2, upper=10),
                        "init_value": 2,
                        "low_cost_init_value": 2,
                    },
                    "min_samples_leaf": {"domain": tune.randint(lower=1, upper=10), "init_value": 1, "low_cost_init_value": 1},
                    "max_features": {"domain": tune.randint(lower=1, upper=10), "init_value": 1, "low_cost_init_value": 1},
                }
                return space

        return MyDTClassification

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = decision_tree_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=DecisionTreeClassification.X_train,
            name_column=DecisionTreeClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=DecisionTreeSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=DecisionTreeSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=DecisionTreeClassification.X_train,
            name_column=DecisionTreeClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=DecisionTreeSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=DecisionTreeSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class RandomForestClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Random Forest algorithm to make insightful products."""

    name = "Random Forest"
    special_function = [func.value for func in RandomForestSpecialFunction]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = 4,
        min_samples_split: Union[int, float] = 4,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[str, int, float] = "sqrt",
        max_leaf_nodes: Optional[int] = 3,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Union[str, dict, list[dict], None] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float] = 10,
    ) -> None:
        """
        A random forest classifier.

        A random forest is a meta estimator that fits a number of decision tree
        classifiers on various sub-samples of the dataset and uses averaging to
        improve the predictive accuracy and control over-fitting.
        The sub-sample size is controlled with the `max_samples` parameter if
        `bootstrap=True` (default), otherwise the whole dataset is used to build
        each tree.

        Read more in the :ref:`User Guide <forest>`.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.

            .. versionchanged:: 0.22
               The default value of ``n_estimators`` changed from 10 to 100
               in 0.22.

        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
            Note: this parameter is tree-specific.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `round(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees. If False, the
            whole dataset is used to build each tree.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score.
            Only available if bootstrap=True.

        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls both the randomness of the bootstrapping of the samples used
            when building trees (if ``bootstrap=True``) and the sampling of the
            features to consider when looking for the best split at each node
            (if ``max_features < n_features``).
            See :term:`Glossary <random_state>` for details.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`the Glossary <warm_start>`.

        class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
                default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.

            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``

            The "balanced_subsample" mode is the same as "balanced" except that
            weights are computed based on the bootstrap sample for every tree
            grown.

            For multi-output, the weights of each column of y will be multiplied.

            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.

            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0.0, 1.0]`.

            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.ensemble.RandomForestClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
        """

        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )

        self.naming = RandomForestClassification.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": ["rf"],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = random_forest_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=RandomForestClassification.X_train,
            name_column=RandomForestClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=RandomForestSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=RandomForestSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=RandomForestClassification.X_train,
            name_column=RandomForestClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=RandomForestSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=RandomForestSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class XGBoostClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using XGBoost algorithm to make insightful products."""

    name = "XGBoost"
    special_function = [func.value for func in XGBoostSpecialFunction]

    # https: // xgboost.readthedocs.io / en / stable / python / python_api.html  # module-xgboost.sklearn
    _SklObjective = Optional[Union[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        grow_policy=1,
        learning_rate: Optional[float] = None,
        verbosity: Optional[int] = None,
        objective: _SklObjective = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = "weight",
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        max_depth [default=6]
            Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
            0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
            exact tree method requires non-zero value.
            range: [0,∞]

        learning_rate [default=0.3]
            Step size shrinkage used in update to prevents overfitting.
            After each boosting step, we can directly get the weights of new features,
            and eta shrinks the feature weights to make the boosting process more conservative.
            range: [0,1]

        n_estimators : int
        Number of gradient boosted trees.  Equivalent to number of boosting rounds.

        objective : {SklObjective}
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).

        verbosity [default=1]
            Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
            Sometimes XGBoost tries to change configurations based on heuristics,
            which is displayed as warning message.
            If there’s unexpected behaviour, please try to increase value of verbosity.

        booster [default= gbtree ]
            Which booster to use. Can be gbtree, gblinear or dart;
            gbtree and dart use tree based models while gblinear uses linear functions.

        tree_method string [default= auto]
            The tree construction algorithm used in XGBoost. See description in the reference paper and Tree Methods.
            XGBoost supports approx, hist and gpu_hist for distributed training. Experimental support for external memory is available for approx and gpu_hist.
            Choices: auto, exact, approx, hist, gpu_hist, this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
                auto: Use heuristic to choose the fastest method.
                    For small dataset, exact greedy (exact) will be used.
                    For larger dataset, approximate algorithm (approx) will be chosen. It’s recommended to try hist and gpu_hist for higher performance with large dataset.
                      (gpu_hist)has support for external memory.
                    Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
                exact: Exact greedy algorithm. Enumerates all split candidates.
                approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
                hist: Faster histogram optimized approximate greedy algorithm.
                gpu_hist: GPU implementation of hist algorithm.

        n_jobs : Optional[int]
            Number of parallel threads used to run xgboost.  When used with other
            Scikit-Learn algorithms like grid search, you may choose which algorithm to
            parallelize and balance the threads.  Creating thread contention will
            significantly slow down both algorithms.

        gamma [default=0, alias: min_split_loss]
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
            The larger gamma is, the more conservative the algorithm will be.
            range: [0,∞]

        min_child_weight [default=1]
            Minimum sum of instance weight (hessian) needed in a child.
            If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
            then the building process will give up further partitioning. In linear regression task,
            this simply corresponds to minimum number of instances needed to be in each node.
            The larger min_child_weight is, the more conservative the algorithm will be.
            range: [0,∞]

        max_delta_step [default=0]
            Maximum delta step we allow each leaf output to be.
            If the value is set to 0, it means there is no constraint.
            If it is set to a positive value, it can help making the update step more conservative.
            Usually this parameter is not needed, but it might help in logistic regression
            when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
            range: [0,∞]

        subsample [default=1]
            Subsample ratio of the training instances.
            Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
            and this will prevent overfitting.
            Subsampling will occur once in every boosting iteration.
            range: (0,1]

        colsample_bytree [default=1]
            colsample_bytree is the subsample ratio of columns when constructing each tree.
            Subsampling occurs once for every tree constructed.

        colsample_bylevel [default=1]
            colsample_bylevel is the subsample ratio of columns for each level.
            Subsampling occurs once for every new depth level reached in a tree.
            Columns are subsampled from the set of columns chosen for the current tree.

        colsample_bynode [default=1]
            colsample_bynode is the subsample ratio of columns for each node (split).
            Subsampling occurs once every time a new split is evaluated.
            Columns are subsampled from the set of columns chosen for the current level.

        reg_alpha [default=0]
            L1 regularization term on weights.
            Increasing this value will make model more conservative.

        reg_lambda [default=1, alias: reg_lambda]
            L2 regularization term on weights.
            Increasing this value will make model more conservative.

        scale_pos_weight [default=1]
            Control the balance of positive and negative weights, useful for unbalanced classes.

        predictor, [default= auto]
            The type of predictor algorithm to use.
            Provides the same results but allows the use of GPU or CPU.
                auto: Configure predictor based on heuristics.
                cpu_predictor: Multicore CPU prediction algorithm.
                gpu_predictor: Prediction using GPU. Used when tree_method is gpu_hist.
                    When predictor is set to default value auto, the gpu_hist tree method is able to provide GPU based prediction
                    without copying training data to GPU memory. If gpu_predictor is explicitly specified,
                    then all data is copied into GPU, only recommended for performing prediction tasks.

        base_score : Optional[float]
            The initial prediction score of all instances, global bias.

        random_state : Optional[Union[numpy.random.RandomState, int]]
            Random number seed.
            .. note::

               Using gblinear booster with shotgun updater is nondeterministic as
               it uses Hogwild algorithm.

        missing : float, default np.nan
        Value in the data which needs to be present as a missing value.

        num_parallel_tree: Optional[int]
            Used for boosting random forest.

        monotone_constraints : Optional[Union[Dict[str, int], str]]
            Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
            for more information.

        interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
            Constraints for interaction representing permitted interactions.  The
            constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
            3, 4]]``, where each inner list is a group of indices of features that are
            allowed to interact with each other.  See :doc:`tutorial
            </tutorials/feature_interaction_constraint>` for more information

        importance_type: Optional[str]
            The feature importance type for the feature_importances\\_ property:

            * For tree model, it's either "gain", "weight", "cover", "total_gain" or
              "total_cover".
            * For linear model, only "weight" is defined and it's the normalized coefficients
              without bias.

        gpu_id : Optional[int]
            Device ordinal.

        validate_parameters : Optional[bool]
            Give warnings for unknown parameter.

        eval_metric : Optional[Union[str, List[str], Callable]]

        early_stopping_rounds : Optional[int]

        References
        ----------
        [1] XGBoost Python API Reference - Scikit-Learn API
            https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

        [2] XGBoost API for the scikit-learn wrapper:
            https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

        if kwargs:
            self.kwargs = kwargs

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = xgboost.XGBClassifier(
            n_estimators=self.n_estimators,
            objective=self.objective,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            max_bin=self.max_bin,
            learning_rate=self.learning_rate,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters,
            predictor=self.predictor,
            enable_categorical=self.enable_categorical,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.naming = XGBoostClassification.name

    @dispatch(object, object)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        y_series = pd.Series(np.ravel(y))
        classes = sorted(y_series.dropna().unique().tolist())
        expected_classes = list(range(len(classes)))
        if classes != expected_classes:
            raise ValueError(f"XGBoost classification labels must be contiguous integers {expected_classes}, but got {classes}.")
        if len(classes) > 2:
            if self.objective not in (None, "multi:softprob", "multi:softmax"):
                raise ValueError("XGBoost multiclass classification requires objective 'multi:softprob' or 'multi:softmax'.")
            self.model.set_params(objective=self.objective or "multi:softprob", num_class=len(classes), eval_metric=self.eval_metric or "mlogloss")
        else:
            self.model.set_params(eval_metric=self.eval_metric or "logloss")
        self.model.fit(X, y_series)

    @dispatch(object, object, bool)
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, is_automl: bool = False) -> None:
        """Fit XGBoost through the shared FLAML workflow in AutoML mode."""
        self._fit_automl(X, y)

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": ["xgboost"],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = xgboost_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    # @staticmethod
    # def _plot_tree(trained_model: object, algorithm_name: str, local_path: str, mlflow_path: str) -> None:
    #     # TODO: (solve the problem of failed to execute WindowsPath('dot'), make sure the Graphviz executables are on your systems' PATH
    #     # Drawing diagrams of the first decision tree of xgboost
    #     print("-----* XGBoost's Single Tree Diagram *-----")
    #     xgboost.plot_tree(trained_model)
    #     # node_params = {
    #     #     'shape': 'box',
    #     #     'style': 'filled,rounded',
    #     #     'fillcolor': '#78bceb'
    #     # }
    #     # xgboost.to_graphviz(self.model, condition_node_params = node_params)
    #     save_fig(f"Single Tree Diagram - {algorithm_name}", local_path, mlflow_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        # self._plot_tree(
        #     trained_model=self.model,
        #     algorithm_name=self.naming,
        #     local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
        #     mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        # )
        self._plot_feature_importance(
            X_train=XGBoostClassification.X_train,
            name_column=XGBoostClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=XGBoostSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=XGBoostClassification.X_train,
            name_column=XGBoostClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=XGBoostSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class LogisticRegressionClassification(LinearWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Logistic Regression algorithm to make insightful products."""

    name = "Logistic Regression"
    special_function = [func.value for func in LogisticRegressionSpecialFunction]

    def __init__(
        self,
        penalty: str = "l2",
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Optional[Union[Dict, str]] = None,
        random_state: Optional[int] = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int = None,
        l1_ratio: float = None,
    ) -> None:
        """
        Parameters
        ----------
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Specify the norm of the penalty:

            - `'none'`: no penalty is added;
            - `'l2'`: add a L2 penalty term and it is the default choice;
            - `'l1'`: add a L1 penalty term;
            - `'elasticnet'`: both L1 and L2 penalty terms are added.

            .. warning::
               Some penalties may not work with some solvers. See the parameter
               `solver` below, to know the compatibility between the penalty and
               solver.

            .. versionadded:: 0.19
               l1 penalty with SAGA solver (allowing 'multinomial' + L1)

        dual : bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for
            l2 penalty with liblinear solver. Prefer dual=False when
            n_samples > n_features.

        tol : float, default=1e-4
            Tolerance for stopping criteria.

        C : float, default=1.0
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger
            regularization.

        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.

        intercept_scaling : float, default=1
            Useful only when the solver 'liblinear' is used
            and self.fit_intercept is set to True. In this case, x becomes
            [x, self.intercept_scaling],
            i.e. a "synthetic" feature with constant value equal to
            intercept_scaling is appended to the instance vector.
            The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
            Note! the synthetic feature weight is subject to l1/l2 regularization
            as all other features.
            To lessen the effect of regularization on synthetic feature weight
            (and therefore on the intercept) intercept_scaling has to be increased.

        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
            .. versionadded:: 0.17
               *class_weight='balanced'*

        random_state : int, RandomState instance, default=None
            Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
            data. See :term:`Glossary <random_state>` for details.

        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
                default='lbfgs'

            Algorithm to use in the optimization problem. Default is 'lbfgs'.
            To choose a solver, you might want to consider the following aspects:

                - For small datasets, 'liblinear' is a good choice, whereas 'sag'
                  and 'saga' are faster for large ones;
                - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
                  'lbfgs' handle multinomial loss;
                - 'liblinear' is limited to one-versus-rest schemes.

            .. warning::
               The choice of the algorithm depends on the penalty chosen:
               Supported penalties by solver:
               - 'newton-cg'   -   ['l2', 'none']
               - 'lbfgs'       -   ['l2', 'none']
               - 'liblinear'   -   ['l1', 'l2']
               - 'sag'         -   ['l2', 'none']
               - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']

            .. note::
               'sag' and 'saga' fast convergence is only guaranteed on
               features with approximately the same scale. You can
               preprocess the data with a scaler from :mod:`sklearn.preprocessing`.

            .. seealso::
               Refer to the User Guide for more information regarding
               :class:`LogisticRegression` and more specifically the
               :ref:`Table <Logistic_regression>`
               summarizing solver/penalty supports.

            .. versionadded:: 0.17
               Stochastic Average Gradient descent solver.
            .. versionadded:: 0.19
               SAGA solver.
            .. versionchanged:: 0.22
                The default solver changed from 'liblinear' to 'lbfgs' in 0.22.

        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.

        multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
            If the option chosen is 'ovr', then a binary problem is fit for each
            label. For 'multinomial' the loss minimised is the multinomial loss fit
            across the entire probability distribution, *even when the data is
            binary*. 'multinomial' is unavailable when solver='liblinear'.
            'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
            and otherwise selects 'multinomial'.
            .. versionadded:: 0.18
               Stochastic Average Gradient descent solver for 'multinomial' case.
            .. versionchanged:: 0.22
                Default changed from 'ovr' to 'auto' in 0.22.

        verbose : int, default=0
            For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

            .. versionadded:: 0.17
               *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

        n_jobs : int, default=None
            Number of CPU cores used when parallelizing over classes if
            multi_class='ovr'". This parameter is ignored when the ``solver`` is
            set to 'liblinear' regardless of whether 'multi_class' is specified or
            not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors.
            See :term:`Glossary <n_jobs>` for more details.

        l1_ratio : float, default=None
            The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
            used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
            to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
            to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
            combination of L1 and L2.

        References
        ----------
        Scikit-learn API: sklearn.linear_model.LogisticRegression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
        super().__init__()
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.l1_ratio = l1_ratio

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            warm_start=self.warm_start,
            l1_ratio=self.l1_ratio,
        )

        self.naming = LogisticRegressionClassification.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": ["lrl2"],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = logistic_regression_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @staticmethod
    def _plot_feature_importance(columns_name: np.ndarray, name_column: str, trained_model: any, algorithm_name: str, local_path: str, mlflow_path: str, func_name: str) -> None:
        """Print the feature coefficient value orderly."""
        print(f"-----* {func_name} *-----")
        data = plot_logistic_importance(columns_name, trained_model)
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        save_data(data, name_column, f"{func_name} - {algorithm_name}", local_path, mlflow_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            features_name=LogisticRegressionClassification.X.columns,
            regression_classification="Classification",
            y_train=LogisticRegressionClassification.y,
            algorithm_name=self.naming,
            func_name=LogisticRegressionSpecialFunction.LOGISTIC_REGRESSION_FORMULA.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        self._plot_feature_importance(
            columns_name=LogisticRegressionClassification.X.columns,
            name_column=LogisticRegressionClassification.name_all,
            trained_model=self.model,
            algorithm_name=self.naming,
            func_name=LogisticRegressionSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=self.auto_model.coef_,
            intercept=self.auto_model.intercept_,
            features_name=LogisticRegressionClassification.X.columns,
            regression_classification="Classification",
            y_train=LogisticRegressionClassification.y,
            algorithm_name=self.naming,
            func_name=LogisticRegressionSpecialFunction.LOGISTIC_REGRESSION_FORMULA.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
        self._plot_feature_importance(
            columns_name=LogisticRegressionClassification.X.columns,
            name_column=LogisticRegressionClassification.name_all,
            trained_model=self.auto_model,
            algorithm_name=self.naming,
            func_name=LogisticRegressionSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class MLPClassification(ClassificationWorkflowBase):
    """The automation workflow of using Multi-layer Perceptron algorithm to make insightful products."""

    name = "Multi-layer Perceptron"
    special_function = [func.value for func in MLPSpecialFunction]

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        *,
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ):
        """
        Parameters
        ----------
        hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
            The ith element represents the number of neurons in the ith
            hidden layer.

        activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            Activation function for the hidden layer.
            - 'identity', no-op activation, useful to implement linear bottleneck,
            returns f(x) = x
            - 'logistic', the logistic sigmoid function,
            returns f(x) = 1 / (1 + exp(-x)).
            - 'tanh', the hyperbolic tan function,
            returns f(x) = tanh(x).
            - 'relu', the rectified linear unit function,
            returns f(x) = max(0, x)

        solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
            The solver for weight optimization.
            - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
            - 'sgd' refers to stochastic gradient descent.
            - 'adam' refers to a stochastic gradient-based optimizer proposed
            by Kingma, Diederik, and Jimmy Ba
            Note: The default solver 'adam' works pretty well on relatively
            large datasets (with thousands of training samples or more) in terms of
            both training time and validation score.
            For small datasets, however, 'lbfgs' can converge faster and perform
            better.

        alpha : float, default=0.0001
            Strength of the L2 regularization term. The L2 regularization term
            is divided by the sample size when added to the loss.

        batch_size : int, default='auto'
            Size of minibatches for stochastic optimizers.
            If the solver is 'lbfgs', the classifier will not use minibatch.
            When set to "auto", `batch_size=min(200, n_samples)`.

        learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
            Learning rate schedule for weight updates.
            - 'constant' is a constant learning rate given by
            'learning_rate_init'.
            - 'invscaling' gradually decreases the learning rate at each
            time step 't' using an inverse scaling exponent of 'power_t'.
            effective_learning_rate = learning_rate_init / pow(t, power_t)
            - 'adaptive' keeps the learning rate constant to
            'learning_rate_init' as long as training loss keeps decreasing.
            Each time two consecutive epochs fail to decrease training loss by at
            least tol, or fail to increase validation score by at least tol if
            'early_stopping' is on, the current learning rate is divided by 5.
            Only used when ``solver='sgd'``.

        learning_rate_init : float, default=0.001
            The initial learning rate used. It controls the step-size
            in updating the weights. Only used when solver='sgd' or 'adam'.

        power_t : float, default=0.5
            The exponent for inverse scaling learning rate.
            It is used in updating effective learning rate when the learning_rate
            is set to 'invscaling'. Only used when solver='sgd'.

        max_iter : int, default=200
            Maximum number of iterations. The solver iterates until convergence
            (determined by 'tol') or this number of iterations. For stochastic
            solvers ('sgd', 'adam'), note that this determines the number of epochs
            (how many times each data point will be used), not the number of
            gradient steps.

        shuffle : bool, default=True
            Whether to shuffle samples in each iteration. Only used when
            solver='sgd' or 'adam'.

        random_state : int, RandomState instance, default=None
            Determines random number generation for weights and bias
            initialization, train-test split if early stopping is used, and batch
            sampling when solver='sgd' or 'adam'.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        tol : float, default=1e-4
            Tolerance for the optimization. When the loss or score is not improving
            by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
            unless ``learning_rate`` is set to 'adaptive', convergence is
            considered to be reached and training stops.

        verbose : bool, default=False
            Whether to print progress messages to stdout.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous
            call to fit as initialization, otherwise, just erase the
            previous solution. See :term:`the Glossary <warm_start>`.

        momentum : float, default=0.9
            Momentum for gradient descent update. Should be between 0 and 1. Only
            used when solver='sgd'.

        nesterovs_momentum : bool, default=True
            Whether to use Nesterov's momentum. Only used when solver='sgd' and
            momentum > 0.

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation
            score is not improving. If set to true, it will automatically set
            aside 10% of training data as validation and terminate training when
            validation score is not improving by at least tol for
            ``n_iter_no_change`` consecutive epochs. The split is stratified,
            except in a multilabel setting.
            If early stopping is False, then the training stops when the training
            loss does not improve by more than tol for n_iter_no_change consecutive
            passes over the training set.
            Only effective when solver='sgd' or 'adam'.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
            Only used if early_stopping is True.

        beta_1 : float, default=0.9
            Exponential decay rate for estimates of first moment vector in adam,
            should be in [0, 1). Only used when solver='adam'.

        beta_2 : float, default=0.999
            Exponential decay rate for estimates of second moment vector in adam,
            should be in [0, 1). Only used when solver='adam'.

        epsilon : float, default=1e-8
            Value for numerical stability in adam. Only used when solver='adam'.

        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet ``tol`` improvement.
            Only effective when solver='sgd' or 'adam'.
            .. versionadded:: 0.20

        max_fun : int, default=15000
            Only used when solver='lbfgs'. Maximum number of loss function calls.
            The solver iterates until convergence (determined by 'tol'), number
            of iterations reaches max_iter, or this number of loss function calls.
            Note that number of loss function calls will be greater than or equal
            to the number of iterations for the `MLPClassifier`.
            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.neural_network.MLPClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """

        super().__init__()
        self.hidden_layer_sizes = (hidden_layer_sizes,)
        self.activation = (activation,)
        self.solver = (solver,)
        self.alpha = (alpha,)
        self.batch_size = (batch_size,)
        self.learning_rate = (learning_rate,)
        self.learning_rate_init = (learning_rate_init,)
        self.power_t = (power_t,)
        self.max_iter = (max_iter,)
        self.shuffle = (shuffle,)
        self.tol = (tol,)
        self.verbose = (verbose,)
        self.warm_start = (warm_start,)
        self.momentum = (momentum,)
        self.nesterovs_momentum = (nesterovs_momentum,)
        self.early_stopping = (early_stopping,)
        self.validation_fraction = (validation_fraction,)
        self.beta_1 = (beta_1,)
        self.beta_2 = (beta_2,)
        self.epsilon = (epsilon,)
        self.n_iter_no_change = (n_iter_no_change,)
        self.max_fun = (max_fun,)

        if random_state is not None:
            self.random_state = (random_state,)
        else:
            self.random_state = (self.random_state,)

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes[0],
            activation=self.activation[0],
            alpha=self.alpha[0],
            batch_size=self.batch_size[0],
            learning_rate=self.learning_rate[0],
            learning_rate_init=self.learning_rate_init[0],
            power_t=self.power_t[0],
            max_iter=self.max_iter[0],
            shuffle=self.shuffle[0],
            random_state=self.random_state[0],
            tol=self.tol[0],
            verbose=self.verbose[0],
            warm_start=self.warm_start[0],
            momentum=self.momentum[0],
            solver=self.solver[0],
            nesterovs_momentum=self.nesterovs_momentum[0],
            early_stopping=self.early_stopping[0],
            validation_fraction=self.validation_fraction[0],
            beta_1=self.beta_1[0],
            beta_2=self.beta_2[0],
            epsilon=self.epsilon[0],
            n_iter_no_change=self.n_iter_no_change[0],
            max_fun=self.max_fun[0],
        )

        self.naming = MLPClassification.name

    def ray_tune(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """Select a deterministic MLP configuration in the current process."""

        from sklearn.metrics import accuracy_score

        random_state = self._automl_random_seed()

        def customized_model(l1: int, l2: int, l3: int, batch: int) -> object:
            """The customized model by Scikit-learn framework."""
            return MLPClassifier(hidden_layer_sizes=(l1, l2, l3), batch_size=batch, random_state=random_state)

        scored_models = []
        for config in self._automl_mlp_configurations():
            clfr = customized_model(**config)
            clfr.fit(X_train, y_train)
            y_pred = clfr.predict(X_test)
            scored_models.append((float(accuracy_score(y_test, y_pred)), config))

        _, best_config = max(scored_models, key=lambda item: item[0])
        self.ray_best_model = customized_model(**best_config)

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = multi_layer_perceptron_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @staticmethod
    def _plot_loss_curve(trained_model: object, algorithm_name: str, func_name: str, local_path: str, mlflow_path: str) -> None:
        """Plot the learning curve of the trained model."""
        print(f"-----* {func_name} *-----")
        data = pd.DataFrame(trained_model.loss_curve_, columns=["Loss"])
        data.plot(title="Loss")
        save_fig(f"{func_name} - {algorithm_name}", local_path, mlflow_path)
        save_data_without_data_identifier(data, f"{func_name} - {algorithm_name}", local_path, mlflow_path)

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        if self.model.get_params()["solver"] in ["sgd", "adam"]:
            self._plot_loss_curve(
                trained_model=self.model,
                algorithm_name=self.naming,
                func_name=MLPSpecialFunction.LOSS_CURVE_DIAGRAM.value,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        if self.model.get_params()["solver"] in ["sgd", "adam"]:
            self._plot_loss_curve(
                trained_model=self.auto_model,
                algorithm_name=self.naming,
                func_name=MLPSpecialFunction.LOSS_CURVE_DIAGRAM.value,
                local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
                mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
            )


class ExtraTreesClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Extra-Trees algorithm to make insightful products."""

    name = "Extra-Trees"
    special_function = [func.value for func in ExtraTreesSpecialFunction]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[float, int] = 2,
        min_samples_leaf: Union[float, int] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[str, float, int] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: str = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
            .. versionchanged:: 0.22
            The default value of ``n_estimators`` changed from 10 to 100
            in 0.22.

        criterion : {"gini", "entropy", "log_loss"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "log_loss" and "entropy" both for the
            Shannon information gain, see :ref:`tree_mathematical_formulation`.
            Note: This parameter is tree-specific.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.
            .. versionchanged:: 0.18
            Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.
            .. versionchanged:: 0.18
            Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `max(1, int(max_features * n_features_in_))` features are considered at each
            split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            .. versionchanged:: 1.1
                The default of `max_features` changed from `"auto"` to `"sqrt"`.
            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.
            .. versionadded:: 0.19

        bootstrap : bool, default=False
            Whether bootstrap samples are used when building trees. If False, the
            whole dataset is used to build each tree.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score.
            Only available if bootstrap=True.

        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls 3 sources of randomness:
            - the bootstrapping of the samples used when building trees
            (if ``bootstrap=True``)
            - the sampling of the features to consider when looking for the best
            split at each node (if ``max_features < n_features``)
            - the draw of the splits for each of the `max_features`
            See :term:`Glossary <random_state>` for details.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`Glossary <warm_start>` and
            :ref:`gradient_boosting_warm_start` for details.

        class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
                default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            The "balanced_subsample" mode is the same as "balanced" except that
            weights are computed based on the bootstrap sample for every tree
            grown.
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.
            .. versionadded:: 0.22

        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
            `max_samples` should be in the interval `(0.0, 1.0]`.
            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.ensemble.ExtraTreesClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn-ensemble-extratreesclassifier
        """

        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )

        self.naming = ExtraTreesClassification.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": ["extra_tree"],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = extra_trees_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=ExtraTreesClassification.X_train,
            name_column=ExtraTreesClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=ExtraTreesSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=ExtraTreesSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=ExtraTreesClassification.X_train,
            name_column=ExtraTreesClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=ExtraTreesSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=ExtraTreesSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class GradientBoostingClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Gradient Boosting algorithm to make insightful products."""

    name = "Gradient Boosting"
    special_function = [func.value for func in GradientBoostingSpecialFunction]

    def __init__(
        self,
        *,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "squared_error",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: float = 3,
        min_impurity_decrease: float = 0.0,
        init: Optional[object] = None,
        random_state: Optional[int] = None,
        max_features: Union[str, int, float] = None,
        verbose: int = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        loss : {'log_loss', 'exponential'}, default='log_loss'
            The loss function to be optimized. 'log_loss' refers to binomial and
            multinomial deviance, the same as used in logistic regression.
            It is a good choice for classification with probabilistic outputs.
            For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by `learning_rate`.
            There is a trade-off between learning_rate and n_estimators.
            Values must be in the range `[0.0, inf)`.

        n_estimators : int, default=100
            The number of boosting stages to perform. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.
            Values must be in the range `[1, inf)`.

        subsample : float, default=1.0
            The fraction of samples to be used for fitting the individual base
            learners. If smaller than 1.0 this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.
            Values must be in the range `(0.0, 1.0]`.

        criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
            The function to measure the quality of a split. Supported criteria are
            'friedman_mse' for the mean squared error with improvement score by
            Friedman, 'squared_error' for mean squared error. The default value of
            'friedman_mse' is generally the best as it can provide a better
            approximation in some cases.

            .. versionadded:: 0.18

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, values must be in the range `[2, inf)`.
            - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
              will be `ceil(min_samples_split * n_samples)`.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, values must be in the range `[1, inf)`.
            - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
              will be `ceil(min_samples_leaf * n_samples)`.

            .. versionchanged:: 0.18
               Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.
            Values must be in the range `[0.0, 0.5]`.

        max_depth : int or None, default=3
            Maximum depth of the individual regression estimators. The maximum
            depth limits the number of nodes in the tree. Tune this parameter
            for best performance; the best value depends on the interaction
            of the input variables. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
            If int, values must be in the range `[1, inf)`.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            Values must be in the range `[0.0, inf)`.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        init : estimator or 'zero', default=None
            An estimator object that is used to compute the initial predictions.
            ``init`` has to provide :term:`fit` and :term:`predict_proba`. If
            'zero', the initial raw predictions are set to zero. By default, a
            ``DummyEstimator`` predicting the classes priors is used.

        random_state : int, RandomState instance or None, default=None
            Controls the random seed given to each Tree estimator at each
            boosting iteration.
            In addition, it controls the random permutation of the features at
            each split (see Notes for more details).
            It also controls the random splitting of the training data to obtain a
            validation set if `n_iter_no_change` is not None.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        max_features : {'sqrt', 'log2'}, int or float, default=None
            The number of features to consider when looking for the best split:

            - If int, values must be in the range `[1, inf)`.
            - If float, values must be in the range `(0.0, 1.0]` and the features
              considered at each split will be `max(1, int(max_features * n_features_in_))`.
            - If 'sqrt', then `max_features=sqrt(n_features)`.
            - If 'log2', then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Choosing `max_features < n_features` leads to a reduction of variance
            and an increase in bias.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        verbose : int, default=0
            Enable verbose output. If 1 then it prints progress and performance
            once in a while (the more trees the lower the frequency). If greater
            than 1 then it prints progress and performance for every tree.
            Values must be in the range `[0, inf)`.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            Values must be in the range `[2, inf)`.
            If `None`, then unlimited number of leaf nodes.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just erase the
            previous solution. See :term:`the Glossary <warm_start>`.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Values must be in the range `(0.0, 1.0)`.
            Only used if ``n_iter_no_change`` is set to an integer.

            .. versionadded:: 0.20

        n_iter_no_change : int, default=None
            ``n_iter_no_change`` is used to decide if early stopping will be used
            to terminate training when validation score is not improving. By
            default it is set to None to disable early stopping. If set to a
            number, it will set aside ``validation_fraction`` size of the training
            data as validation and terminate training when validation score is not
            improving in all of the previous ``n_iter_no_change`` numbers of
            iterations. The split is stratified.
            Values must be in the range `[1, inf)`.

            .. versionadded:: 0.20

        tol : float, default=1e-4
            Tolerance for the early stopping. When the loss is not improving
            by at least tol for ``n_iter_no_change`` iterations (if set to a
            number), the training stops.
            Values must be in the range `[0.0, inf)`.

            .. versionadded:: 0.20

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed.
            Values must be in the range `[0.0, inf)`.
            See :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        References
        ----------
        Scikit-learn API: sklearn.ensemble.HistGradientBoostingClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        """

        super().__init__()
        self.loss = (loss,)
        self.learning_rate = (learning_rate,)
        self.n_estimators = (n_estimators,)
        self.criterion = (criterion,)
        self.min_samples_split = (min_samples_split,)
        self.min_samples_leaf = (min_samples_leaf,)
        self.min_weight_fraction_leaf = (min_weight_fraction_leaf,)
        self.max_depth = (max_depth,)
        self.init = (init,)
        self.subsample = (subsample,)
        self.max_features = (max_features,)
        self.verbose = (verbose,)
        self.max_leaf_nodes = (max_leaf_nodes,)
        self.min_impurity_decrease = (min_impurity_decrease,)
        self.warm_start = (warm_start,)
        self.validation_fraction = (validation_fraction,)
        self.n_iter_no_change = (n_iter_no_change,)
        self.tol = (tol,)
        self.ccp_alpha = (ccp_alpha,)

        if random_state is not None:
            self.random_state = (random_state,)
        else:
            self.random_state = (self.random_state,)

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = GradientBoostingClassifier(
            loss=self.loss[0],
            learning_rate=self.learning_rate[0],
            n_estimators=self.n_estimators[0],
            criterion=self.criterion[0],
            min_samples_split=self.min_samples_split[0],
            min_samples_leaf=self.min_samples_leaf[0],
            min_weight_fraction_leaf=self.min_weight_fraction_leaf[0],
            max_depth=self.max_depth[0],
            init=self.init[0],
            subsample=self.subsample[0],
            max_features=self.max_features[0],
            random_state=self.random_state[0],
            verbose=self.verbose[0],
            max_leaf_nodes=self.max_leaf_nodes[0],
            min_impurity_decrease=self.min_impurity_decrease[0],
            warm_start=self.warm_start[0],
            validation_fraction=self.validation_fraction[0],
            n_iter_no_change=self.n_iter_no_change[0],
            tol=self.tol[0],
            ccp_alpha=self.ccp_alpha[0],
        )

        self.naming = GradientBoostingClassification.name
        self.customized = True
        self.customized_name = "Gradient Boosting"

    @property
    def settings(self) -> Dict:
        """The configuration of Gradient Boosting to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized Gradient Boosting of FLAML framework."""
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from flaml.model import SKLearnEstimator
        from sklearn.ensemble import GradientBoostingClassifier

        class MyGradientBoostingClassification(SKLearnEstimator):
            def __init__(self, task="classification", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = GradientBoostingClassifier

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    # "criterion":{"domain": tune.choice(["friedman_mse", "squared_error"])},
                    "n_estimators": {"domain": tune.lograndint(lower=4, upper=512), "init_value": 100},
                    "max_depth": {"domain": tune.randint(lower=1, upper=10), "init_value": 3},
                    "learning_rate": {"domain": tune.loguniform(lower=0.001, upper=1.0), "init_value": 0.1},
                    "subsample": {"domain": tune.uniform(lower=0.1, upper=1.0), "init_value": 1.0},
                    "min_samples_split": {"domain": tune.randint(lower=2, upper=20), "init_value": 2},
                    "min_samples_leaf": {"domain": tune.randint(lower=1, upper=20), "init_value": 1},
                    "min_impurity_decrease": {"domain": tune.loguniform(lower=1e-10, upper=1e-2), "init_value": 0.0},
                }
                return space

        return MyGradientBoostingClassification

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = gradient_boosting_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=GradientBoostingClassification.X_train,
            name_column=GradientBoostingClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=GradientBoostingSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0][0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=GradientBoostingSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=GradientBoostingClassification.X_train,
            name_column=GradientBoostingClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=GradientBoostingSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0][0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=GradientBoostingSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class AdaBoostClassification(TreeWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using AdaBoosting algorithm to make insightful products."""

    name = "AdaBoost"
    special_function = [func.value for func in AdaBoostSpecialFunction]

    def __init__(
        self,
        estimator: object = None,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Union[int] = None,
        max_depth: int = 3,
        # algorithm: str = "SAMME",  # may deprecated in new sklearn version 1.6
    ) -> None:
        """
        Parameters
        ----------
        estimator : object, default=None
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            ``classes_`` and ``n_classes_`` attributes. If ``None``, then
            the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
            initialized with `max_depth=1`.
            .. versionadded:: 1.2
               `base_estimator` was renamed to `estimator`.
        n_estimators : int, default=50
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.
            Values must be in the range `[1, inf)`.
        learning_rate : float, default=1.0
            Weight applied to each classifier at each boosting iteration. A higher
            learning rate increases the contribution of each classifier. There is
            a trade-off between the `learning_rate` and `n_estimators` parameters.
            Values must be in the range `(0.0, inf)`.
        algorithm : {'SAMME'}, default='SAMME'
            Use the SAMME discrete boosting algorithm.
            .. deprecated:: 1.6
                `algorithm` is deprecated and will be removed in version 1.8. This
                estimator only implements the 'SAMME' algorithm.
        random_state : int, RandomState instance or None, default=None
            Controls the random seed given at each `estimator` at each
            boosting iteration.
            Thus, it is only used when `estimator` exposes a `random_state`.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        References
        ----------
        Scikit-learn API: sklearn.ensemble.AdaBoostClassifier
        https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        """
        super().__init__()
        self.estimator = DecisionTreeClassifier(
            max_depth=max_depth,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        # if 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        if random_state is not None:
            self.random_state = (random_state,)
        else:
            self.random_state = (self.random_state,)

        self.model = AdaBoostClassifier(
            base_estimator=self.estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            # algorithm=self.algorithm,  # deprecated in version 1.6 of sklearn
            random_state=self.random_state[0],
        )

        self.naming = AdaBoostClassification.name
        self.customized = True
        self.customized_name = "AdaBoost"

    @property
    def settings(self) -> Dict:
        """The configuration of AdaBoosting to implenment AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "classification",  # task type
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized AdaBoosting of FLAML framework"""
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from flaml.model import SKLearnEstimator
        from sklearn.ensemble import AdaBoostClassifier

        class MyAdaBoostClassification(SKLearnEstimator):
            def __init__(self, task="classification", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = AdaBoostClassifier

            @classmethod
            def search_space(cls, data_size, task):
                space = {  # FLAML can only change these two hyperparameter
                    "n_estimators": {
                        "domain": tune.lograndint(lower=4, upper=512),
                        "init_value": 50,
                    },
                    "learning_rate": {
                        "domain": tune.loguniform(lower=0.001, upper=1.0),
                        "init_value": 0.1,
                    },
                }
                return space

        return MyAdaBoostClassification

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"-*-* {cls.name} - Hyper-parameters Specification -*-*")
        hyper_parameters = adaboost_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=AdaBoostClassification.X_train,
            name_column=AdaBoostClassification.name_train,
            trained_model=self.model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=AdaBoostSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=AdaBoostSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML frameworks"""
        GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        self._plot_feature_importance(
            X_train=AdaBoostClassification.X_train,
            name_column=AdaBoostClassification.name_train,
            trained_model=self.auto_model,
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=AdaBoostSpecialFunction.FEATURE_IMPORTANCE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )
        self._plot_tree(
            trained_model=self.auto_model.estimators_[0],
            image_config=self.image_config,
            algorithm_name=self.naming,
            func_name=AdaBoostSpecialFunction.SINGLE_TREE_DIAGRAM.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH,
            mlflow_path=MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH,
        )


class KNNClassification(ClassificationWorkflowBase):
    """The automation workflow of using KNN algorithm to make insightful products."""

    name = "K-Nearest Neighbors"
    special_function = []

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Optional[Dict] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use by default for :meth:`kneighbors` queries.

        weights : {'uniform', 'distance'}, callable or None, default='uniform'
            Weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights.  All points in each neighborhood
            are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
            array of distances, and returns an array of the same shape
            containing the weights.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.

        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.  This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree.  The optimal value depends on the
            nature of the problem.

        p : int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric : str or callable, default='minkowski'
            Metric to use for distance computation. Default is "minkowski", which
            results in the standard Euclidean distance when p = 2. See the
            documentation of `scipy.spatial.distance
            <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
            the metrics listed in
            :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
            values.

            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square during fit. X may be a :term:`sparse graph`, in which
            case only "nonzero" elements may be considered neighbors.

            If metric is a callable function, it takes two arrays representing 1D
            vectors as inputs and must return one value indicating the distance
            between those vectors. This works for Scipy's metrics, but is less
            efficient than passing the metric name as a string.

        metric_params : dict, default=None
            Additional keyword arguments for the metric function.

        n_jobs : int, default=None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
            Doesn't affect :meth:`fit` method.


        References
        ----------
        Scikit-learn API: sklearn.neighbors.KNeighborsClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )

        self.naming = KNNClassification.name

    @property
    def settings(self) -> Dict:
        """The configuration to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": ["kneighbor"],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f"{self.naming} - automl.log",  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = knn_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        pass

    @dispatch(bool)
    def special_components(self, is_automl: bool, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        pass


class SGDClassification(LinearWorkflowMixin, ClassificationWorkflowBase):
    """The automation workflow of using Stochastic Gradient Descent - SGD algorithm to make insightful products."""

    name = "Stochastic Gradient Descent"

    special_function = [func.value for func in SGDSpecialFunction]

    def __init__(
        self,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: Union[float, None] = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        n_jobs: int = None,
        random_state: Optional[int] = None,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight: Union[dict, str] = None,
        warm_start: bool = False,
        average: Union[bool, int] = False,
    ) -> None:
        """
        Parameters
        ----------
        loss : {'hinge', 'log_loss', 'modified_huber', 'squared_hinge',\
            'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',\
            'squared_epsilon_insensitive'}, default='hinge'
            The loss function to be used.

            - 'hinge' gives a linear SVM.
            - 'log_loss' gives logistic regression, a probabilistic classifier.
            - 'modified_huber' is another smooth loss that brings tolerance to
            outliers as well as probability estimates.
            - 'squared_hinge' is like hinge but is quadratically penalized.
            - 'perceptron' is the linear loss used by the perceptron algorithm.
            - The other losses, 'squared_error', 'huber', 'epsilon_insensitive' and
            'squared_epsilon_insensitive' are designed for regression but can be useful
            in classification as well; see
            :class:`~sklearn.linear_model.SGDRegressor` for a description.

            More details about the losses formulas can be found in the
            :ref:`User Guide <sgd_mathematical_formulation>`.

        penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
            The penalty (aka regularization term) to be used. Defaults to 'l2'
            which is the standard regularizer for linear SVM models. 'l1' and
            'elasticnet' might bring sparsity to the model (feature selection)
            not achievable with 'l2'. No penalty is added when set to `None`.

        alpha : float, default=0.0001
            Constant that multiplies the regularization term. The higher the
            value, the stronger the regularization. Also used to compute the
            learning rate when `learning_rate` is set to 'optimal'.
            Values must be in the range `[0.0, inf)`.

        l1_ratio : float, default=0.15
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
            Only used if `penalty` is 'elasticnet'.
            Values must be in the range `[0.0, 1.0]`.

        fit_intercept : bool, default=True
            Whether the intercept should be estimated or not. If False, the
            data is assumed to be already centered.

        max_iter : int, default=1000
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the ``fit`` method, and not the
            :meth:`partial_fit` method.
            Values must be in the range `[1, inf)`.

            .. versionadded:: 0.19

        tol : float or None, default=1e-3
            The stopping criterion. If it is not None, training will stop
            when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
            epochs.
            Convergence is checked against the training loss or the
            validation loss depending on the `early_stopping` parameter.
            Values must be in the range `[0.0, inf)`.

            .. versionadded:: 0.19

        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch.

        verbose : int, default=0
            The verbosity level.
            Values must be in the range `[0, inf)`.

        epsilon : float, default=0.1
            Epsilon in the epsilon-insensitive loss functions; only if `loss` is
            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
            For 'huber', determines the threshold at which it becomes less
            important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current prediction
            and the correct label are ignored if they are less than this threshold.
            Values must be in the range `[0.0, inf)`.

        n_jobs : int, default=None
            The number of CPUs to use to do the OVA (One Versus All, for
            multi-class problems) computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        random_state : int, RandomState instance, default=None
            Used for shuffling the data, when ``shuffle`` is set to ``True``.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
            Integer values must be in the range `[0, 2**32 - 1]`.

        learning_rate : str, default='optimal'
            The learning rate schedule:

            - 'constant': `eta = eta0`
            - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
            where `t0` is chosen by a heuristic proposed by Leon Bottou.
            - 'invscaling': `eta = eta0 / pow(t, power_t)`
            - 'adaptive': `eta = eta0`, as long as the training keeps decreasing.
            Each time n_iter_no_change consecutive epochs fail to decrease the
            training loss by tol or fail to increase validation score by tol if
            `early_stopping` is `True`, the current learning rate is divided by 5.

                .. versionadded:: 0.20
                    Added 'adaptive' option

        eta0 : float, default=0.0
            The initial learning rate for the 'constant', 'invscaling' or
            'adaptive' schedules. The default value is 0.0 as eta0 is not used by
            the default schedule 'optimal'.
            Values must be in the range `(0.0, inf)`.

        power_t : float, default=0.5
            The exponent for inverse scaling learning rate [default 0.5].
            Values must be in the range `(-inf, inf)`.

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation
            score is not improving. If set to `True`, it will automatically set aside
            a stratified fraction of training data as validation and terminate
            training when validation score returned by the `score` method is not
            improving by at least tol for n_iter_no_change consecutive epochs.

            .. versionadded:: 0.20
                Added 'early_stopping' option

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
            Only used if `early_stopping` is True.
            Values must be in the range `(0.0, 1.0)`.

            .. versionadded:: 0.20
                Added 'validation_fraction' option

        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before stopping
            fitting.
            Convergence is checked against the training loss or the
            validation loss depending on the `early_stopping` parameter.
            Integer values must be in the range `[1, max_iter)`.

            .. versionadded:: 0.20
                Added 'n_iter_no_change' option

        class_weight : dict, {class_label: weight} or "balanced", default=None
            Preset for the class_weight fit parameter.

            Weights associated with classes. If not given, all classes
            are supposed to have weight one.

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            See :term:`the Glossary <warm_start>`.

            Repeatedly calling fit or partial_fit when warm_start is True can
            result in a different solution than when calling fit a single time
            because of the way the data is shuffled.
            If a dynamic learning rate is used, the learning rate is adapted
            depending on the number of samples already seen. Calling ``fit`` resets
            this counter, while ``partial_fit`` will result in increasing the
            existing counter.

        average : bool or int, default=False
            When set to `True`, computes the averaged SGD weights across all
            updates and stores the result in the ``coef_`` attribute. If set to
            an int greater than 1, averaging will begin once the total number of
            samples seen reaches `average`. So ``average=10`` will begin
            averaging after seeing 10 samples.
            Integer values must be in the range `[1, n_samples]`.


        References
        ----------
        Scikit-learn API: sklearn.linear_model.SGDClassifier
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        """

        super().__init__()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

        if random_state is not None:
            self.random_state = random_state

        # If 'random_state' is None, 'self.random_state' comes from the parent class 'WorkflowBase'
        self.model = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            warm_start=self.warm_start,
            average=self.average,
        )

        self.naming = SGDClassification.name
        self.customized = True
        self.customized_name = "Stochastic Gradient Descent"

    @property
    def settings(self) -> Dict:
        """The configuration of SGD to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": 10,  # total running time in seconds
            "metric": "accuracy",
            "estimator_list": [self.customized_name],  # list of ML learners
            "task": "classification",  # task type
            # "log_file_name": f'{self.naming} - automl.log',  # flaml log file
            # "log_training_metric": True,  # whether to log training metric
        }
        return configuration

    @property
    def customization(self) -> object:
        """The customized SGD of FLAML framework."""
        from flaml import tune
        from flaml.data import CLASSIFICATION
        from flaml.model import SKLearnEstimator
        from sklearn.linear_model import SGDClassifier

        class MySGDClassification(SKLearnEstimator):
            def __init__(self, task="classification", n_jobs=None, **config):
                super().__init__(task, **config)
                if task in CLASSIFICATION:
                    self.estimator_class = SGDClassifier

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "loss": {"domain": tune.choice(["log_loss", "modified_huber"]), "init_value": "log_loss"},
                    "penalty": {"domain": tune.choice(["l2", "l1", "elasticnet", None]), "init_value": "l2"},
                    "alpha": {"domain": tune.loguniform(lower=0.0001, upper=1), "init_value": 0.0001},
                    "l1_ratio": {"domain": tune.uniform(lower=0, upper=1), "init_value": 0.15},
                    "fit_intercept": {"domain": tune.choice([True, False]), "init_value": True},
                    "max_iter": {"domain": tune.randint(lower=50, upper=1000), "init_value": 1000},
                    "tol": {"domain": tune.loguniform(lower=0.000001, upper=0.001), "init_value": 0.001},
                    "shuffle": {"domain": tune.choice([True, False]), "init_value": True},
                    "learning_rate": {"domain": tune.choice(["constant", "optimal", "invscaling", "adaptive"]), "init_value": "optimal"},
                    "eta0": {"domain": tune.loguniform(lower=0.000001, upper=0.1), "init_value": 0.000001},
                    "power_t": {"domain": tune.uniform(lower=0.1, upper=0.9), "init_value": 0.5},
                    "early_stopping": {"domain": tune.choice([True, False]), "init_value": False},
                    # Keep enough samples on both sides of the early-stopping
                    # split. FLAML may sample the upper bound exactly, so 1.0
                    # is not a valid bound for sklearn's open interval (0, 1).
                    "validation_fraction": {"domain": tune.uniform(lower=0.05, upper=0.5), "init_value": 0.1},
                    "warm_start": {"domain": tune.choice([True, False]), "init_value": False},
                }
                return space

        return MySGDClassification

    @classmethod
    def manual_hyper_parameters(cls) -> Dict:
        """Manual hyper-parameters specification."""
        print(f"[bold green]-*-*- {cls.name} - Hyper-parameters Specification -*-*-[/bold green]")
        hyper_parameters = sgd_classificaiton_manual_hyper_parameters()
        clear_output()
        return hyper_parameters

    @dispatch()
    def special_components(self, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by Scikit-learn framework."""
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            features_name=SGDClassification.X_train.columns,
            regression_classification="Classification",
            y_train=SGDClassification.y,
            algorithm_name=self.naming,
            func_name=SGDSpecialFunction.SGD_CLASSIFICATION_FORMULA.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )

    @dispatch(bool)
    def special_components(self, is_automl: bool = False, **kwargs) -> None:
        """Invoke all special application functions for this algorithms by FLAML framework."""
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        self._show_formula(
            coef=self.auto_model.coef_,
            intercept=self.auto_model.intercept_,
            features_name=SGDClassification.X.columns,
            regression_classification="Classification",
            y_train=SGDClassification.y,
            algorithm_name=self.naming,
            func_name=SGDSpecialFunction.SGD_CLASSIFICATION_FORMULA.value,
            local_path=GEOPI_OUTPUT_ARTIFACTS_PATH,
            mlflow_path="root",
        )
