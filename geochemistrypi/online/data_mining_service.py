"""Lightweight Data Mining workflows for the Online application."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_mining_models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    extract_linear_parameters,
    get_classification_model,
    get_regression_model,
)
from .schemas import (
    ArtifactResponse,
    ClassificationConfusionItem,
    ClassificationMetrics,
    ClassificationResponse,
    ClassificationSummary,
    ClusterCenterItem,
    ClusteringMetrics,
    ClusteringResponse,
    ClusteringSummary,
    ClusterSizeItem,
    ColumnProfileItem,
    DataMiningCatalogResponse,
    DataMiningFeatureItem,
    DataMiningMethodItem,
    DataPreprocessingResponse,
    DataPreprocessingSummary,
    DatasetProfileResponse,
    DatasetProfileSummary,
    RegressionCoefficientItem,
    RegressionMetrics,
    RegressionResponse,
    RegressionSummary,
)
from .service import InvalidDatasetError, UploadTooLargeError


class DataMiningService:
    """Run small, non-interactive Data Mining jobs without the legacy web stack."""

    supported_suffixes = {".xlsx", ".csv"}
    preprocessing_strategies = {
        "keep",
        "drop_rows",
        "fill_mean",
        "fill_median",
        "fill_mode",
    }

    def __init__(
        self,
        runtime_dir: Path,
        max_upload_bytes: int = 10 * 1024 * 1024,
        max_rows: int = 100_000,
        max_columns: int = 500,
    ):
        self.runtime_dir = runtime_dir.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.jobs_dir = self.runtime_dir / "data-mining-jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_catalog() -> DataMiningCatalogResponse:
        return DataMiningCatalogResponse(
            features=[
                DataMiningFeatureItem(
                    name="dataset_profile",
                    description="Dataset overview and quality check",
                    status="verified",
                    status_message=(
                        "已完成 Excel/CSV 上传、数据类型识别、缺失值、重复行、唯一值、"
                        "数值统计、预览和 JSON 报告下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["页面质量概览", "逐列统计", "数据预览", "JSON 报告"],
                ),
                DataMiningFeatureItem(
                    name="data_preprocessing",
                    description="Data preprocessing",
                    status="verified",
                    status_message=(
                        "已完成列选择、缺失值处理、结果预览、CSV 数据下载"
                        "和 JSON 处理记录验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["处理结果预览", "CSV 处理数据", "JSON 处理记录"],
                ),
                DataMiningFeatureItem(
                    name="regression",
                    description="Regression",
                    status="verified",
                    status_message=(
                        "已接入 v0.8 线性、二阶多项式、Lasso、Elastic Net、"
                        "Bayesian Ridge 和 Ridge 回归，并完成固定随机种子训练测试划分、"
                        "R²/MAE/RMSE、系数、预测结果和报告下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["回归指标", "模型系数", "预测结果 CSV", "JSON 模型报告"],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                        )
                        for definition in REGRESSION_MODELS.values()
                    ],
                ),
                DataMiningFeatureItem(
                    name="classification",
                    description="Classification",
                    status="verified",
                    status_message=(
                        "已接入 v0.8 Logistic、SVM、Decision Tree、Random Forest、"
                        "Extra-Trees、MLP、Gradient Boosting、KNN、SGD 和 AdaBoost，"
                        "并完成分层训练测试划分、"
                        "Accuracy/Precision/Recall/F1、混淆矩阵和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["分类指标", "混淆矩阵", "预测结果 CSV", "JSON 模型报告"],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                        )
                        for definition in CLASSIFICATION_MODELS.values()
                    ],
                ),
                DataMiningFeatureItem(
                    name="clustering",
                    description="Clustering",
                    status="verified",
                    status_message=(
                        "已完成数值特征标准化、K-means 聚类、簇数设置、"
                        "三项聚类评价指标、聚类中心和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["聚类指标", "簇大小与中心", "聚类结果 CSV", "JSON 模型报告"],
                ),
            ]
        )

    def profile_dataset(
        self,
        *,
        filename: str | None,
        content: bytes,
    ) -> DatasetProfileResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        summary = self._build_summary(dataframe)
        column_profiles = [
            self._profile_column(str(column), dataframe[column])
            for column in dataframe.columns
        ]
        warnings = self._build_warnings(dataframe, summary, column_profiles)
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in dataframe.head(10).to_dict(orient="records")
        ]

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        report_path = output_dir / "dataset_profile.json"
        report_payload = {
            "report_version": "dataset-profile-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "summary": summary.model_dump(),
            "columns": [profile.model_dump() for profile in column_profiles],
            "preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return DatasetProfileResponse(
            job_id=job_id,
            status="success",
            message="Dataset profile completed",
            source_filename=Path(filename or "dataset").name,
            summary=summary,
            columns=column_profiles,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=report_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{report_path.name}"
                    ),
                    size_bytes=report_path.stat().st_size,
                )
            ],
        )

    def preprocess_dataset(
        self,
        *,
        filename: str | None,
        content: bytes,
        selected_columns: list[str],
        missing_strategy: str,
    ) -> DataPreprocessingResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        columns = self._validate_selected_columns(
            dataframe,
            selected_columns,
        )
        if missing_strategy not in self.preprocessing_strategies:
            allowed = ", ".join(sorted(self.preprocessing_strategies))
            raise InvalidDatasetError(
                f"Unknown missing-value strategy. Choose one of: {allowed}"
            )

        processed = dataframe.loc[:, columns].copy()
        original_missing_cells = int(processed.isna().sum().sum())
        warnings: list[str] = []

        if missing_strategy == "drop_rows":
            processed = processed.dropna(axis=0, how="any")
            if processed.empty:
                raise InvalidDatasetError(
                    "The selected missing-value strategy removes all data rows"
                )
        elif missing_strategy in {"fill_mean", "fill_median"}:
            numeric_columns = [
                column
                for column in processed.columns
                if pd.api.types.is_numeric_dtype(processed[column])
                and not pd.api.types.is_bool_dtype(processed[column])
            ]
            for column in numeric_columns:
                finite_values = pd.to_numeric(
                    processed[column],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan)
                fill_value = (
                    finite_values.mean()
                    if missing_strategy == "fill_mean"
                    else finite_values.median()
                )
                if pd.notna(fill_value):
                    processed[column] = processed[column].fillna(fill_value)
            remaining_non_numeric = int(
                processed.drop(columns=numeric_columns, errors="ignore")
                .isna()
                .sum()
                .sum()
            )
            if remaining_non_numeric:
                warnings.append(
                    f"{remaining_non_numeric} 个非数值缺失单元格未被"
                    "均值/中位数规则修改。"
                )
        elif missing_strategy == "fill_mode":
            for column in processed.columns:
                mode = processed[column].mode(dropna=True)
                if not mode.empty:
                    processed[column] = processed[column].fillna(mode.iloc[0])

        processed_missing_cells = int(processed.isna().sum().sum())
        filled_cells = (
            max(0, original_missing_cells - processed_missing_cells)
            if missing_strategy.startswith("fill_")
            else 0
        )
        if processed_missing_cells:
            warnings.append(
                f"处理结果仍有 {processed_missing_cells} 个缺失单元格。"
            )
        else:
            warnings.append("处理结果中没有缺失单元格。")

        summary = DataPreprocessingSummary(
            original_rows=int(dataframe.shape[0]),
            original_columns=int(dataframe.shape[1]),
            processed_rows=int(processed.shape[0]),
            processed_columns=int(processed.shape[1]),
            removed_rows=int(dataframe.shape[0] - processed.shape[0]),
            removed_columns=int(dataframe.shape[1] - processed.shape[1]),
            original_missing_cells=original_missing_cells,
            processed_missing_cells=processed_missing_cells,
            filled_cells=filled_cells,
        )
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in processed.head(10).to_dict(orient="records")
        ]

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        data_path = output_dir / "processed_data.csv"
        report_path = output_dir / "preprocessing_report.json"
        processed.to_csv(data_path, index=False, encoding="utf-8-sig")
        report_payload = {
            "report_version": "data-preprocessing-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "selected_columns": columns,
            "missing_strategy": missing_strategy,
            "summary": summary.model_dump(),
            "preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return DataPreprocessingResponse(
            job_id=job_id,
            status="success",
            message="Data preprocessing completed",
            source_filename=Path(filename or "dataset").name,
            selected_columns=columns,
            missing_strategy=missing_strategy,
            summary=summary,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=data_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{data_path.name}"
                    ),
                    size_bytes=data_path.stat().st_size,
                ),
                ArtifactResponse(
                    name=report_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{report_path.name}"
                    ),
                    size_bytes=report_path.stat().st_size,
                ),
            ],
        )

    def run_regression(
        self,
        *,
        filename: str | None,
        content: bytes,
        target_column: str,
        feature_columns: list[str],
        test_size: float = 0.2,
        model_name: str = "linear_regression",
    ) -> RegressionResponse:
        try:
            model_definition = get_regression_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        if target_column not in dataframe.columns:
            raise InvalidDatasetError(
                f"Unknown target column: {target_column}"
            )
        features = self._validate_selected_columns(
            dataframe,
            feature_columns,
        )
        if target_column in features:
            raise InvalidDatasetError(
                "The target column cannot also be a feature column"
            )
        if not 0.1 <= test_size <= 0.5:
            raise InvalidDatasetError(
                "Test size must be between 0.1 and 0.5"
            )

        model_columns = [*features, target_column]
        non_numeric = [
            column
            for column in model_columns
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric:
            raise InvalidDatasetError(
                "Regression requires numeric columns: "
                + ", ".join(non_numeric)
            )

        model_data = (
            dataframe.loc[:, model_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        if usable_rows < 10:
            raise InvalidDatasetError(
                "Regression requires at least 10 complete numeric rows"
            )

        test_rows = int(math.ceil(usable_rows * test_size))
        train_rows = usable_rows - test_rows
        if test_rows < 2:
            raise InvalidDatasetError(
                "The test split must contain at least 2 rows"
            )
        if train_rows <= len(features):
            raise InvalidDatasetError(
                "The training split must contain more rows than feature columns"
            )

        target = model_data[target_column].astype(float)
        if target.nunique(dropna=True) < 2:
            raise InvalidDatasetError(
                "The target column must contain at least two distinct values"
            )
        feature_data = model_data.loc[:, features].astype(float)
        (
            features_train,
            features_test,
            target_train,
            target_test,
        ) = train_test_split(
            feature_data,
            target,
            test_size=test_size,
            random_state=42,
        )

        model = model_definition.factory()
        model.fit(features_train, target_train)
        predicted = model.predict(features_test)
        mae = float(mean_absolute_error(target_test, predicted))
        rmse = float(math.sqrt(mean_squared_error(target_test, predicted)))
        r2_value = float(
            r2_score(target_test, predicted, force_finite=False)
        )
        warnings: list[str] = []
        if not math.isfinite(r2_value):
            r2: float | None = None
            warnings.append(
                "测试集目标值缺少变化，无法计算有效的 R²。"
            )
        else:
            r2 = r2_value
        if dropped_rows:
            warnings.append(
                f"训练前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            )
        else:
            warnings.append("所有数据行均可用于回归。")

        intercept, coefficient_names, coefficient_values = extract_linear_parameters(
            model,
            features,
        )
        coefficients = [
            RegressionCoefficientItem(
                feature=feature,
                coefficient=float(coefficient),
            )
            for feature, coefficient in zip(
                coefficient_names,
                coefficient_values,
                strict=True,
            )
        ]
        equation = self._build_regression_equation(
            target_column=target_column,
            intercept=intercept,
            coefficients=coefficients,
        )
        prediction_frame = pd.DataFrame(
            {
                "source_row": [
                    int(index) + 2
                    if isinstance(index, (int, np.integer))
                    else str(index)
                    for index in target_test.index
                ],
                "actual": target_test.to_numpy(dtype=float),
                "predicted": predicted,
                "residual": target_test.to_numpy(dtype=float) - predicted,
            }
        ).sort_values("source_row")
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in prediction_frame.head(20).to_dict(orient="records")
        ]
        summary = RegressionSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            train_rows=int(features_train.shape[0]),
            test_rows=int(features_test.shape[0]),
            feature_count=len(features),
        )
        metrics = RegressionMetrics(
            r2=r2,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
        )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        predictions_path = output_dir / "regression_predictions.csv"
        report_path = output_dir / "regression_report.json"
        prediction_frame.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        report_payload = {
            "report_version": (
                "linear-regression-v1"
                if model_name == "linear_regression"
                else "v080-regression-v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "target_column": target_column,
            "feature_columns": features,
            "test_size": test_size,
            "random_state": 42,
            "summary": summary.model_dump(),
            "metrics": metrics.model_dump(),
            "intercept": intercept,
            "coefficients": [
                coefficient.model_dump()
                for coefficient in coefficients
            ],
            "equation": equation,
            "prediction_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return RegressionResponse(
            job_id=job_id,
            status="success",
            message=f"{model_definition.display_name} completed",
            source_filename=Path(filename or "dataset").name,
            model=model_name,
            model_display_name=model_definition.display_name,
            target_column=target_column,
            feature_columns=features,
            test_size=test_size,
            random_state=42,
            summary=summary,
            metrics=metrics,
            intercept=intercept,
            coefficients=coefficients,
            equation=equation,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=predictions_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/"
                        f"{predictions_path.name}"
                    ),
                    size_bytes=predictions_path.stat().st_size,
                ),
                ArtifactResponse(
                    name=report_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{report_path.name}"
                    ),
                    size_bytes=report_path.stat().st_size,
                ),
            ],
        )

    def run_classification(
        self,
        *,
        filename: str | None,
        content: bytes,
        target_column: str,
        feature_columns: list[str],
        test_size: float = 0.2,
        model_name: str = "logistic_regression",
    ) -> ClassificationResponse:
        try:
            model_definition = get_classification_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        if target_column not in dataframe.columns:
            raise InvalidDatasetError(
                f"Unknown target column: {target_column}"
            )
        features = self._validate_selected_columns(
            dataframe,
            feature_columns,
        )
        if target_column in features:
            raise InvalidDatasetError(
                "The target column cannot also be a feature column"
            )
        if not 0.1 <= test_size <= 0.5:
            raise InvalidDatasetError(
                "Test size must be between 0.1 and 0.5"
            )

        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric:
            raise InvalidDatasetError(
                "Classification requires numeric feature columns: "
                + ", ".join(non_numeric)
            )

        model_columns = [*features, target_column]
        model_data = (
            dataframe.loc[:, model_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        if usable_rows < 12:
            raise InvalidDatasetError(
                "Classification requires at least 12 complete rows"
            )

        target = model_data[target_column].astype(str)
        class_counts = target.value_counts()
        class_count = int(class_counts.shape[0])
        if class_count < 2:
            raise InvalidDatasetError(
                "The target column must contain at least two classes"
            )
        if int(class_counts.min()) < 2:
            raise InvalidDatasetError(
                "Each target class must contain at least two complete rows"
            )

        test_rows = int(math.ceil(usable_rows * test_size))
        train_rows = usable_rows - test_rows
        if test_rows < class_count or train_rows < class_count:
            raise InvalidDatasetError(
                "The train and test splits must each have room for every class"
            )

        feature_data = model_data.loc[:, features].astype(float)
        (
            features_train,
            features_test,
            target_train,
            target_test,
        ) = train_test_split(
            feature_data,
            target,
            test_size=test_size,
            random_state=42,
            stratify=target,
        )

        model = model_definition.factory()
        model.fit(features_train, target_train)
        predicted = model.predict(features_test)
        classes = [str(value) for value in model.classes_]
        metrics = ClassificationMetrics(
            accuracy=float(accuracy_score(target_test, predicted)),
            precision_macro=float(
                precision_score(
                    target_test,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            recall_macro=float(
                recall_score(
                    target_test,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            f1_macro=float(
                f1_score(
                    target_test,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
        )
        matrix = confusion_matrix(
            target_test,
            predicted,
            labels=classes,
        )
        confusion_items = [
            ClassificationConfusionItem(
                actual_class=actual_class,
                predicted_class=predicted_class,
                count=int(matrix[actual_index, predicted_index]),
            )
            for actual_index, actual_class in enumerate(classes)
            for predicted_index, predicted_class in enumerate(classes)
        ]
        prediction_frame = pd.DataFrame(
            {
                "source_row": [
                    int(index) + 2
                    if isinstance(index, (int, np.integer))
                    else str(index)
                    for index in target_test.index
                ],
                "actual": target_test.to_numpy(dtype=str),
                "predicted": np.asarray(predicted, dtype=str),
                "correct": target_test.to_numpy(dtype=str)
                == np.asarray(predicted, dtype=str),
            }
        ).sort_values("source_row")
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in prediction_frame.head(20).to_dict(orient="records")
        ]
        warnings = [
            f"训练前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            if dropped_rows
            else "所有数据行均可用于分类。"
        ]
        summary = ClassificationSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            train_rows=int(features_train.shape[0]),
            test_rows=int(features_test.shape[0]),
            feature_count=len(features),
            class_count=class_count,
        )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        predictions_path = output_dir / "classification_predictions.csv"
        report_path = output_dir / "classification_report.json"
        prediction_frame.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        report_payload = {
            "report_version": (
                "logistic-classification-v1"
                if model_name == "logistic_regression"
                else "v080-classification-v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "target_column": target_column,
            "feature_columns": features,
            "test_size": test_size,
            "random_state": 42,
            "classes": classes,
            "summary": summary.model_dump(),
            "metrics": metrics.model_dump(),
            "confusion_matrix": [item.model_dump() for item in confusion_items],
            "prediction_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ClassificationResponse(
            job_id=job_id,
            status="success",
            message=f"{model_definition.display_name} completed",
            source_filename=Path(filename or "dataset").name,
            model=model_name,
            model_display_name=model_definition.display_name,
            target_column=target_column,
            feature_columns=features,
            test_size=test_size,
            random_state=42,
            classes=classes,
            summary=summary,
            metrics=metrics,
            confusion_matrix=confusion_items,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=predictions_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/"
                        f"{predictions_path.name}"
                    ),
                    size_bytes=predictions_path.stat().st_size,
                ),
                ArtifactResponse(
                    name=report_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{report_path.name}"
                    ),
                    size_bytes=report_path.stat().st_size,
                ),
            ],
        )

    def run_clustering(
        self,
        *,
        filename: str | None,
        content: bytes,
        feature_columns: list[str],
        cluster_count: int = 3,
    ) -> ClusteringResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        features = self._validate_selected_columns(
            dataframe,
            feature_columns,
        )
        if not 2 <= cluster_count <= 10:
            raise InvalidDatasetError(
                "Cluster count must be between 2 and 10"
            )
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric:
            raise InvalidDatasetError(
                "Clustering requires numeric feature columns: "
                + ", ".join(non_numeric)
            )

        model_data = (
            dataframe.loc[:, features]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        minimum_rows = max(10, cluster_count * 2)
        if usable_rows < minimum_rows:
            raise InvalidDatasetError(
                "Clustering requires at least "
                f"{minimum_rows} complete numeric rows for {cluster_count} clusters"
            )

        feature_data = model_data.astype(float)
        distinct_rows = int(
            np.unique(feature_data.to_numpy(dtype=float), axis=0).shape[0]
        )
        if distinct_rows < cluster_count:
            raise InvalidDatasetError(
                "The dataset must contain at least as many distinct feature rows "
                "as clusters"
            )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)
        if int(np.unique(labels).shape[0]) != cluster_count:
            raise InvalidDatasetError(
                "K-means could not produce the requested number of distinct clusters"
            )

        silhouette_sample_size = min(10_000, usable_rows)
        metrics = ClusteringMetrics(
            silhouette_score=float(
                silhouette_score(
                    scaled_features,
                    labels,
                    sample_size=silhouette_sample_size,
                    random_state=42,
                )
            ),
            davies_bouldin_score=float(
                davies_bouldin_score(scaled_features, labels)
            ),
            calinski_harabasz_score=float(
                calinski_harabasz_score(scaled_features, labels)
            ),
        )
        cluster_sizes = [
            ClusterSizeItem(
                cluster=cluster,
                rows=int(np.sum(labels == cluster)),
            )
            for cluster in range(cluster_count)
        ]
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        cluster_centers = [
            ClusterCenterItem(
                cluster=cluster,
                values={
                    feature: float(value)
                    for feature, value in zip(
                        features,
                        centers_original[cluster],
                        strict=True,
                    )
                },
            )
            for cluster in range(cluster_count)
        ]

        assignment_frame = feature_data.copy()
        assignment_frame.insert(
            0,
            "source_row",
            [
                int(index) + 2
                if isinstance(index, (int, np.integer))
                else str(index)
                for index in assignment_frame.index
            ],
        )
        assignment_frame["cluster"] = labels
        assignment_frame = assignment_frame.sort_values("source_row")
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in assignment_frame.head(20).to_dict(orient="records")
        ]
        warnings = [
            f"聚类前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            if dropped_rows
            else "所有数据行均可用于聚类。"
        ]
        if usable_rows > silhouette_sample_size:
            warnings.append(
                "Silhouette 指标使用固定随机种子抽样 10,000 行计算。"
            )
        summary = ClusteringSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            feature_count=len(features),
            cluster_count=cluster_count,
        )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        assignments_path = output_dir / "clustering_assignments.csv"
        report_path = output_dir / "clustering_report.json"
        assignment_frame.to_csv(
            assignments_path,
            index=False,
            encoding="utf-8-sig",
        )
        report_payload = {
            "report_version": "kmeans-clustering-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": "kmeans",
            "feature_columns": features,
            "cluster_count": cluster_count,
            "random_state": 42,
            "summary": summary.model_dump(),
            "metrics": metrics.model_dump(),
            "cluster_sizes": [item.model_dump() for item in cluster_sizes],
            "cluster_centers": [item.model_dump() for item in cluster_centers],
            "assignment_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ClusteringResponse(
            job_id=job_id,
            status="success",
            message="K-means clustering completed",
            source_filename=Path(filename or "dataset").name,
            model="kmeans",
            feature_columns=features,
            cluster_count=cluster_count,
            random_state=42,
            summary=summary,
            metrics=metrics,
            cluster_sizes=cluster_sizes,
            cluster_centers=cluster_centers,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=assignments_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/"
                        f"{assignments_path.name}"
                    ),
                    size_bytes=assignments_path.stat().st_size,
                ),
                ArtifactResponse(
                    name=report_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{report_path.name}"
                    ),
                    size_bytes=report_path.stat().st_size,
                ),
            ],
        )

    def resolve_artifact(self, job_id: str, file_path: str) -> Path:
        output_dir = (self.jobs_dir / job_id / "output").resolve()
        candidate = (output_dir / file_path).resolve()
        try:
            candidate.relative_to(output_dir)
        except ValueError as exc:
            raise FileNotFoundError(file_path) from exc
        if not candidate.is_file():
            raise FileNotFoundError(file_path)
        return candidate

    def _validate_upload(self, filename: str | None, content: bytes) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in self.supported_suffixes:
            raise InvalidDatasetError(
                "Data Mining supports .xlsx and .csv files"
            )
        if not content:
            raise InvalidDatasetError("The uploaded file is empty")
        if len(content) > self.max_upload_bytes:
            raise UploadTooLargeError(
                f"The uploaded file exceeds {self.max_upload_bytes} bytes"
            )
        return suffix

    @staticmethod
    def _read_dataframe(suffix: str, content: bytes) -> pd.DataFrame:
        try:
            if suffix == ".xlsx":
                return pd.read_excel(BytesIO(content))
            return pd.read_csv(BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDatasetError("CSV files must use UTF-8 encoding") from exc
        except Exception as exc:
            raise InvalidDatasetError(
                f"The uploaded {suffix} file cannot be read"
            ) from exc

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        if dataframe.shape[1] == 0:
            raise InvalidDatasetError("The dataset contains no columns")
        if dataframe.shape[0] == 0:
            raise InvalidDatasetError("The dataset contains no data rows")
        if dataframe.shape[0] > self.max_rows:
            raise InvalidDatasetError(
                f"The dataset exceeds the {self.max_rows} row limit"
            )
        if dataframe.shape[1] > self.max_columns:
            raise InvalidDatasetError(
                f"The dataset exceeds the {self.max_columns} column limit"
            )

    @staticmethod
    def _validate_selected_columns(
        dataframe: pd.DataFrame,
        selected_columns: list[str],
    ) -> list[str]:
        if not isinstance(selected_columns, list):
            raise InvalidDatasetError("Selected columns must be a JSON list")
        columns = [
            str(column)
            for column in selected_columns
            if isinstance(column, str) and column
        ]
        columns = list(dict.fromkeys(columns))
        if not columns:
            raise InvalidDatasetError("Select at least one column")
        unknown = [column for column in columns if column not in dataframe.columns]
        if unknown:
            raise InvalidDatasetError(
                "Unknown selected columns: " + ", ".join(unknown)
            )
        return columns

    @staticmethod
    def _build_regression_equation(
        *,
        target_column: str,
        intercept: float,
        coefficients: list[RegressionCoefficientItem],
    ) -> str:
        terms = [f"{intercept:.8g}"]
        for coefficient in coefficients:
            sign = "+" if coefficient.coefficient >= 0 else "-"
            terms.append(
                f"{sign} {abs(coefficient.coefficient):.8g}"
                f" × {coefficient.feature}"
            )
        return f"{target_column} = " + " ".join(terms)

    @staticmethod
    def _build_summary(dataframe: pd.DataFrame) -> DatasetProfileSummary:
        missing_cells = int(dataframe.isna().sum().sum())
        total_cells = int(dataframe.shape[0] * dataframe.shape[1])
        numeric_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column])
            and not pd.api.types.is_bool_dtype(dataframe[column])
        ]
        boolean_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_bool_dtype(dataframe[column])
        ]
        datetime_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_datetime64_any_dtype(dataframe[column])
        ]
        text_columns = (
            dataframe.shape[1]
            - len(numeric_columns)
            - len(boolean_columns)
            - len(datetime_columns)
        )
        infinite_cells = sum(
            int(np.isinf(dataframe[column].to_numpy(dtype=float)).sum())
            for column in numeric_columns
        )
        return DatasetProfileSummary(
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_rate=missing_cells / total_cells,
            duplicate_rows=int(dataframe.duplicated().sum()),
            numeric_columns=len(numeric_columns),
            text_columns=text_columns,
            datetime_columns=len(datetime_columns),
            boolean_columns=len(boolean_columns),
            infinite_cells=infinite_cells,
            memory_bytes=int(dataframe.memory_usage(index=True, deep=True).sum()),
        )

    @classmethod
    def _profile_column(
        cls,
        name: str,
        series: pd.Series,
    ) -> ColumnProfileItem:
        non_null_series = series.dropna()
        missing = int(series.isna().sum())
        if pd.api.types.is_bool_dtype(series):
            data_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            data_type = "number"
        elif pd.api.types.is_datetime64_any_dtype(series):
            data_type = "datetime"
        else:
            data_type = "text"

        minimum: float | str | None = None
        maximum: float | str | None = None
        mean: float | None = None
        standard_deviation: float | None = None
        if not non_null_series.empty and data_type == "number":
            finite = pd.to_numeric(non_null_series, errors="coerce")
            finite = finite[np.isfinite(finite)]
            if not finite.empty:
                minimum = float(finite.min())
                maximum = float(finite.max())
                mean = float(finite.mean())
                standard_deviation = (
                    float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
                )
        elif not non_null_series.empty and data_type == "datetime":
            minimum = cls._json_value(non_null_series.min())
            maximum = cls._json_value(non_null_series.max())

        sample_values: list[Any] = []
        for value in non_null_series.drop_duplicates().head(3):
            sample_values.append(cls._json_value(value))

        return ColumnProfileItem(
            name=name,
            data_type=data_type,
            pandas_dtype=str(series.dtype),
            non_null=int(non_null_series.shape[0]),
            missing=missing,
            missing_rate=missing / len(series),
            unique=int(series.nunique(dropna=True)),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            sample_values=sample_values,
        )

    @staticmethod
    def _build_warnings(
        dataframe: pd.DataFrame,
        summary: DatasetProfileSummary,
        profiles: list[ColumnProfileItem],
    ) -> list[str]:
        warnings: list[str] = []
        all_missing = [profile.name for profile in profiles if profile.non_null == 0]
        high_missing = [
            profile.name
            for profile in profiles
            if 0.3 <= profile.missing_rate < 1
        ]
        constant = [
            profile.name
            for profile in profiles
            if profile.non_null > 0 and profile.unique <= 1
        ]
        if all_missing:
            warnings.append(
                "全部为空的列：" + "、".join(all_missing)
            )
        if high_missing:
            warnings.append(
                "缺失率不低于 30% 的列：" + "、".join(high_missing)
            )
        if summary.duplicate_rows:
            warnings.append(f"发现 {summary.duplicate_rows} 行完全重复记录。")
        if constant:
            warnings.append(
                "仅含单一非空值的列：" + "、".join(constant)
            )
        if summary.infinite_cells:
            warnings.append(
                f"数值列中发现 {summary.infinite_cells} 个无穷大值。"
            )
        if not warnings:
            warnings.append("未发现高缺失率、完全重复、常量列或无穷大值问题。")
        return warnings

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
