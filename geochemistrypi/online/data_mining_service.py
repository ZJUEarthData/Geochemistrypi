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

from geochemistrypi.data_mining.process.time_series import (
    compute_subaerial_proportion,
)

from .data_mining_models import (
    ANOMALY_DETECTION_MODELS,
    CLASSIFICATION_MODELS,
    CLUSTERING_MODELS,
    DIMENSIONALITY_REDUCTION_MODELS,
    REGRESSION_MODELS,
    extract_linear_parameters,
    get_anomaly_detection_model,
    get_classification_model,
    get_clustering_model,
    get_dimensionality_reduction_model,
    get_regression_model,
)
from .schemas import (
    AnomalyDetectionResponse,
    AnomalyDetectionSummary,
    AnomalyScoreSummary,
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
    DimensionalityReductionMetrics,
    DimensionalityReductionResponse,
    DimensionalityReductionSummary,
    RegressionCoefficientItem,
    RegressionMetrics,
    RegressionResponse,
    RegressionSummary,
    ProbabilityModelInfo,
    ProbabilityModelMetrics,
    ProbabilityPredictionSummary,
    TimeSeriesBinItem,
    TimeSeriesResponse,
    TimeSeriesSummary,
)
from .service import InvalidDatasetError, UploadTooLargeError
from .subaerial_probability import (
    MIN_FEATURES_PER_ROW,
    MODEL_DISPLAY_NAME,
    MODEL_VERSION,
    predict_subaerial_probability,
)


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
        max_upload_bytes: int = 25 * 1024 * 1024,
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
                        "已接入 v0.8 K-Means、DBSCAN、Agglomerative、"
                        "Affinity Propagation、Mean Shift 和 OPTICS，完成标准化、"
                        "噪声识别、三项聚类评价指标、聚类中心和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=["聚类指标", "簇大小与中心", "聚类结果 CSV", "JSON 模型报告"],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                            uses_cluster_count=definition.uses_cluster_count,
                        )
                        for definition in CLUSTERING_MODELS.values()
                    ],
                ),
                DataMiningFeatureItem(
                    name="dimensionality_reduction",
                    description="Dimensionality reduction",
                    status="verified",
                    status_message=(
                        "已接入 v0.8 PCA、T-SNE 和 MDS，完成数值特征标准化、"
                        "二维/三维低维坐标、PCA 解释方差、T-SNE KL 散度、"
                        "MDS stress 以及结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "低维坐标预览",
                        "模型诊断指标",
                        "降维结果 CSV",
                        "JSON 模型报告",
                    ],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                        )
                        for definition in DIMENSIONALITY_REDUCTION_MODELS.values()
                    ],
                ),
                DataMiningFeatureItem(
                    name="anomaly_detection",
                    description="Anomaly detection",
                    status="verified",
                    status_message=(
                        "已接入 v0.8 Isolation Forest 和 Local Outlier Factor，"
                        "完成数值特征标准化、逐行异常标签、统一方向异常分数、"
                        "异常样品预览和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "正常/异常样品统计",
                        "异常分数与标签",
                        "异常检测结果 CSV",
                        "JSON 模型报告",
                    ],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                        )
                        for definition in ANOMALY_DETECTION_MODELS.values()
                    ],
                ),
                DataMiningFeatureItem(
                    name="time_series",
                    description="Time series",
                    status="verified",
                    status_message=(
                        "已接入 v0.8 陆上玄武岩比例时间序列工作流，完成字段映射、"
                        "年龄分箱、固定随机种子 Bootstrap、2σ 不确定度、曲线图和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "陆上玄武岩比例曲线",
                        "年龄分箱结果表",
                        "时间序列结果 CSV",
                        "SVG 矢量图",
                        "JSON 分析报告",
                    ],
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
        model_name: str = "kmeans",
    ) -> ClusteringResponse:
        try:
            model_definition = get_clustering_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
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
        minimum_rows = (
            max(10, cluster_count * 2)
            if model_definition.uses_cluster_count
            else 10
        )
        if usable_rows < minimum_rows:
            raise InvalidDatasetError(
                "Clustering requires at least "
                f"{minimum_rows} complete numeric rows"
            )

        feature_data = model_data.astype(float)
        distinct_rows = int(
            np.unique(feature_data.to_numpy(dtype=float), axis=0).shape[0]
        )
        required_distinct_rows = (
            cluster_count if model_definition.uses_cluster_count else 2
        )
        if distinct_rows < required_distinct_rows:
            if model_definition.uses_cluster_count:
                raise InvalidDatasetError(
                    "The dataset must contain at least as many distinct feature "
                    f"rows as clusters for {model_definition.display_name}"
                )
            raise InvalidDatasetError(
                "The dataset must contain at least two distinct feature rows for "
                f"{model_definition.display_name}"
            )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        model = model_definition.factory(cluster_count)
        labels = model.fit_predict(scaled_features)
        unique_labels = sorted(int(value) for value in np.unique(labels))
        valid_cluster_labels = [label for label in unique_labels if label != -1]
        actual_cluster_count = len(valid_cluster_labels)
        noise_rows = int(np.sum(labels == -1))
        if model_definition.uses_cluster_count and actual_cluster_count != cluster_count:
            raise InvalidDatasetError(
                f"{model_definition.display_name} could not produce the requested "
                "number of distinct clusters"
            )
        metric_mask = labels != -1
        metric_rows = int(np.sum(metric_mask))
        if actual_cluster_count < 2 or metric_rows <= actual_cluster_count:
            raise InvalidDatasetError(
                f"{model_definition.display_name} produced fewer than two usable "
                "clusters. Try different data or model parameters."
            )

        metric_features = scaled_features[metric_mask]
        metric_labels = labels[metric_mask]
        silhouette_sample_size = min(10_000, metric_rows)
        metrics = ClusteringMetrics(
            silhouette_score=float(
                silhouette_score(
                    metric_features,
                    metric_labels,
                    sample_size=silhouette_sample_size,
                    random_state=42,
                )
            ),
            davies_bouldin_score=float(
                davies_bouldin_score(metric_features, metric_labels)
            ),
            calinski_harabasz_score=float(
                calinski_harabasz_score(metric_features, metric_labels)
            ),
        )
        cluster_sizes = [
            ClusterSizeItem(
                cluster=cluster,
                rows=int(np.sum(labels == cluster)),
            )
            for cluster in unique_labels
        ]
        original_values = feature_data.to_numpy(dtype=float)
        cluster_centers = [
            ClusterCenterItem(
                cluster=cluster,
                values={
                    feature: float(value)
                    for feature, value in zip(
                        features,
                        original_values[labels == cluster].mean(axis=0),
                        strict=True,
                    )
                },
            )
            for cluster in valid_cluster_labels
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
        if metric_rows > 10_000:
            warnings.append(
                "Silhouette 指标使用固定随机种子抽样 10,000 行计算。"
            )
        if noise_rows:
            warnings.append(
                f"{model_definition.display_name} 将 {noise_rows} 行识别为噪声；"
                "聚类指标和中心不包含这些噪声行。"
            )
        summary = ClusteringSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            feature_count=len(features),
            cluster_count=actual_cluster_count,
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
            "report_version": (
                "kmeans-clustering-v1"
                if model_name == "kmeans"
                else "v080-clustering-v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "feature_columns": features,
            "cluster_count": actual_cluster_count,
            "requested_cluster_count": (
                cluster_count if model_definition.uses_cluster_count else None
            ),
            "noise_rows": noise_rows,
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
            message=f"{model_definition.display_name} completed",
            source_filename=Path(filename or "dataset").name,
            model=model_name,
            model_display_name=model_definition.display_name,
            feature_columns=features,
            cluster_count=actual_cluster_count,
            requested_cluster_count=(
                cluster_count if model_definition.uses_cluster_count else None
            ),
            noise_rows=noise_rows,
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

    def run_dimensionality_reduction(
        self,
        *,
        filename: str | None,
        content: bytes,
        feature_columns: list[str],
        component_count: int = 2,
        model_name: str = "pca",
    ) -> DimensionalityReductionResponse:
        try:
            model_definition = get_dimensionality_reduction_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        features = self._validate_selected_columns(dataframe, feature_columns)
        if component_count not in {2, 3}:
            raise InvalidDatasetError("Component count must be 2 or 3")
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric:
            raise InvalidDatasetError(
                "Dimensionality reduction requires numeric feature columns: "
                + ", ".join(non_numeric)
            )

        model_data = (
            dataframe.loc[:, features]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        minimum_rows = max(5, component_count + 1)
        if usable_rows < minimum_rows:
            raise InvalidDatasetError(
                "Dimensionality reduction requires at least "
                f"{minimum_rows} complete numeric rows"
            )
        if (
            model_definition.max_rows is not None
            and usable_rows > model_definition.max_rows
        ):
            raise InvalidDatasetError(
                f"{model_definition.display_name} supports at most "
                f"{model_definition.max_rows:,} complete rows in Online mode"
            )
        if component_count > len(features):
            raise InvalidDatasetError(
                "Component count cannot exceed the number of selected features"
            )

        feature_data = model_data.astype(float)
        distinct_rows = int(
            np.unique(feature_data.to_numpy(dtype=float), axis=0).shape[0]
        )
        if distinct_rows < 2:
            raise InvalidDatasetError(
                "Dimensionality reduction requires at least two distinct "
                "feature rows"
            )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        model = model_definition.factory(component_count, usable_rows)
        reduced_values = np.asarray(
            model.fit_transform(scaled_features),
            dtype=float,
        )
        if reduced_values.shape != (usable_rows, component_count):
            raise InvalidDatasetError(
                f"{model_definition.display_name} returned an unexpected output shape"
            )
        if not np.isfinite(reduced_values).all():
            raise InvalidDatasetError(
                f"{model_definition.display_name} returned non-finite coordinates"
            )

        explained_values = getattr(model, "explained_variance_ratio_", None)
        explained_variance_ratio = (
            [float(value) for value in explained_values]
            if explained_values is not None
            else []
        )
        cumulative_explained_variance_ratio = (
            [float(value) for value in np.cumsum(explained_variance_ratio)]
            if explained_variance_ratio
            else []
        )
        kl_divergence_value = getattr(model, "kl_divergence_", None)
        stress_value = getattr(model, "stress_", None)
        metrics = DimensionalityReductionMetrics(
            explained_variance_ratio=explained_variance_ratio,
            cumulative_explained_variance_ratio=(
                cumulative_explained_variance_ratio
            ),
            total_explained_variance_ratio=(
                float(sum(explained_variance_ratio))
                if explained_variance_ratio
                else None
            ),
            kl_divergence=(
                float(kl_divergence_value)
                if kl_divergence_value is not None
                and math.isfinite(float(kl_divergence_value))
                else None
            ),
            stress=(
                float(stress_value)
                if stress_value is not None
                and math.isfinite(float(stress_value))
                else None
            ),
        )

        reduced_frame = feature_data.copy()
        reduced_frame.insert(
            0,
            "source_row",
            [
                int(index) + 2
                if isinstance(index, (int, np.integer))
                else str(index)
                for index in reduced_frame.index
            ],
        )
        for component_index in range(component_count):
            reduced_frame[f"component_{component_index + 1}"] = reduced_values[
                :, component_index
            ]
        reduced_frame = reduced_frame.sort_values("source_row")
        preview_columns = ["source_row"] + [
            f"component_{index + 1}" for index in range(component_count)
        ]
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in reduced_frame.loc[:, preview_columns]
            .head(20)
            .to_dict(orient="records")
        ]
        warnings = [
            f"降维前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            if dropped_rows
            else "所有数据行均可用于降维。"
        ]
        if model_name in {"tsne", "mds"}:
            warnings.append(
                f"{model_definition.display_name} 坐标用于相对结构解释，"
                "坐标轴本身没有原始地球化学单位。"
            )
        summary = DimensionalityReductionSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            feature_count=len(features),
            component_count=component_count,
        )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        coordinates_path = output_dir / "dimensionality_reduction_coordinates.csv"
        report_path = output_dir / "dimensionality_reduction_report.json"
        reduced_frame.to_csv(coordinates_path, index=False, encoding="utf-8-sig")
        report_payload = {
            "report_version": "v080-dimensionality-reduction-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "feature_columns": features,
            "component_count": component_count,
            "random_state": 42,
            "summary": summary.model_dump(),
            "metrics": metrics.model_dump(),
            "coordinate_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return DimensionalityReductionResponse(
            job_id=job_id,
            status="success",
            message=f"{model_definition.display_name} completed",
            source_filename=Path(filename or "dataset").name,
            model=model_name,
            model_display_name=model_definition.display_name,
            feature_columns=features,
            component_count=component_count,
            random_state=42,
            summary=summary,
            metrics=metrics,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=coordinates_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/"
                        f"{coordinates_path.name}"
                    ),
                    size_bytes=coordinates_path.stat().st_size,
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

    def run_anomaly_detection(
        self,
        *,
        filename: str | None,
        content: bytes,
        feature_columns: list[str],
        model_name: str = "isolation_forest",
    ) -> AnomalyDetectionResponse:
        try:
            model_definition = get_anomaly_detection_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        features = self._validate_selected_columns(dataframe, feature_columns)
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric:
            raise InvalidDatasetError(
                "Anomaly detection requires numeric feature columns: "
                + ", ".join(non_numeric)
            )

        model_data = (
            dataframe.loc[:, features]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        if usable_rows < 10:
            raise InvalidDatasetError(
                "Anomaly detection requires at least 10 complete numeric rows"
            )
        feature_data = model_data.astype(float)
        distinct_rows = int(
            np.unique(feature_data.to_numpy(dtype=float), axis=0).shape[0]
        )
        if distinct_rows < 2:
            raise InvalidDatasetError(
                "Anomaly detection requires at least two distinct feature rows"
            )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        model = model_definition.factory(usable_rows)
        labels = np.asarray(model.fit_predict(scaled_features), dtype=int)
        if model_name == "local_outlier_factor":
            anomaly_scores = -np.asarray(
                model.negative_outlier_factor_,
                dtype=float,
            )
        else:
            anomaly_scores = -np.asarray(
                model.decision_function(scaled_features),
                dtype=float,
            )
        if not np.isfinite(anomaly_scores).all():
            raise InvalidDatasetError(
                f"{model_definition.display_name} returned non-finite anomaly scores"
            )

        anomaly_mask = labels == -1
        anomaly_rows = int(np.sum(anomaly_mask))
        normal_rows = int(usable_rows - anomaly_rows)
        score_summary = AnomalyScoreSummary(
            minimum=float(np.min(anomaly_scores)),
            maximum=float(np.max(anomaly_scores)),
            mean=float(np.mean(anomaly_scores)),
        )
        detection_frame = feature_data.copy()
        detection_frame.insert(
            0,
            "source_row",
            [
                int(index) + 2
                if isinstance(index, (int, np.integer))
                else str(index)
                for index in detection_frame.index
            ],
        )
        detection_frame["anomaly_label"] = np.where(
            anomaly_mask,
            "anomaly",
            "normal",
        )
        detection_frame["is_anomaly"] = anomaly_mask
        detection_frame["anomaly_score"] = anomaly_scores
        detection_frame = detection_frame.sort_values("source_row")
        preview_columns = [
            "source_row",
            "anomaly_label",
            "is_anomaly",
            "anomaly_score",
        ]
        preview_frame = detection_frame.sort_values(
            "anomaly_score",
            ascending=False,
        ).loc[:, preview_columns]
        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in preview_frame.head(20).to_dict(orient="records")
        ]
        warnings = [
            f"异常检测前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            if dropped_rows
            else "所有数据行均可用于异常检测。"
        ]
        warnings.append(
            "异常分数已统一为数值越大越异常；不同算法之间的绝对分数不可直接比较。"
        )
        summary = AnomalyDetectionSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            feature_count=len(features),
            normal_rows=normal_rows,
            anomaly_rows=anomaly_rows,
        )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        results_path = output_dir / "anomaly_detection_results.csv"
        report_path = output_dir / "anomaly_detection_report.json"
        detection_frame.to_csv(results_path, index=False, encoding="utf-8-sig")
        report_payload = {
            "report_version": "v080-anomaly-detection-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "feature_columns": features,
            "random_state": 42 if model_name == "isolation_forest" else None,
            "summary": summary.model_dump(),
            "score_summary": score_summary.model_dump(),
            "anomaly_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return AnomalyDetectionResponse(
            job_id=job_id,
            status="success",
            message=f"{model_definition.display_name} completed",
            source_filename=Path(filename or "dataset").name,
            model=model_name,
            model_display_name=model_definition.display_name,
            feature_columns=features,
            random_state=42 if model_name == "isolation_forest" else None,
            summary=summary,
            score_summary=score_summary,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=results_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/"
                        f"{results_path.name}"
                    ),
                    size_bytes=results_path.stat().st_size,
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

    def run_predicted_time_series(
        self,
        *,
        filename: str | None,
        content: bytes,
        age_column: str,
        age_max_column: str,
        latitude_column: str,
        longitude_column: str,
        age_unit: str = "Ma",
        bin_width: float = 10.0,
        bootstrap_iterations: int = 100,
    ) -> TimeSeriesResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        mapped_columns = self._validate_selected_columns(
            dataframe,
            [age_column, age_max_column, latitude_column, longitude_column],
        )
        if len(mapped_columns) != 4:
            raise InvalidDatasetError(
                "Predicted time series requires four different mapped columns"
            )
        if age_unit not in {"Ma", "Ga"}:
            raise InvalidDatasetError("Age unit must be Ma or Ga")
        if not math.isfinite(bin_width) or bin_width <= 0:
            raise InvalidDatasetError("Bin width must be a positive finite number")
        if bootstrap_iterations < 10 or bootstrap_iterations > 1000:
            raise InvalidDatasetError(
                "Bootstrap iterations must be between 10 and 1,000"
            )

        try:
            prediction = predict_subaerial_probability(dataframe)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc

        context = dataframe.loc[:, mapped_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        context = context.replace([np.inf, -np.inf], np.nan)
        predicted_mask = prediction.probabilities.notna()
        complete_context = context.notna().all(axis=1)
        eligible_mask = predicted_mask & complete_context
        eligible_rows = int(eligible_mask.sum())
        if eligible_rows < 10:
            raise InvalidDatasetError(
                "Model-predicted time series requires at least 10 rows with "
                "sufficient geochemistry, age, and coordinates"
            )

        eligible_context = context.loc[eligible_mask]
        if (eligible_context[age_column] < 0).any() or (
            eligible_context[age_max_column] < 0
        ).any():
            raise InvalidDatasetError("Age values must be zero or greater")
        if (
            (eligible_context[latitude_column] < -90)
            | (eligible_context[latitude_column] > 90)
        ).any():
            raise InvalidDatasetError("Latitude values must be between -90 and 90")
        if (
            (eligible_context[longitude_column] < -180)
            | (eligible_context[longitude_column] > 180)
        ).any():
            raise InvalidDatasetError("Longitude values must be between -180 and 180")

        analysis = eligible_context.copy()
        probability_column = "Predicted subaerial probability"
        analysis[probability_column] = prediction.probabilities.loc[eligible_mask]
        maximum_analysis_rows = 25_000
        if analysis.shape[0] > maximum_analysis_rows:
            sampled = self._stratified_age_sample(
                analysis,
                age_column=age_column,
                bin_width=bin_width,
                max_rows=maximum_analysis_rows,
            )
        else:
            sampled = analysis
        sampled_rows = int(sampled.shape[0])

        internal_columns = {
            age_column: "__age__",
            age_max_column: "__age_max__",
            latitude_column: "__latitude__",
            longitude_column: "__longitude__",
            probability_column: "__predicted_probability__",
        }
        derived = sampled.rename(columns=internal_columns).loc[
            :,
            [
                "__age__",
                "__age_max__",
                "__predicted_probability__",
                "__latitude__",
                "__longitude__",
            ],
        ]
        derived_name = f"{Path(filename or 'dataset').stem}-model-predicted.csv"
        response = self.run_time_series(
            filename=derived_name,
            content=derived.to_csv(index=False).encode("utf-8"),
            age_column="__age__",
            age_max_column="__age_max__",
            probability_column="__predicted_probability__",
            latitude_column="__latitude__",
            longitude_column="__longitude__",
            age_unit=age_unit,
            bin_width=bin_width,
            bootstrap_iterations=bootstrap_iterations,
        )

        bundle = prediction.bundle
        metrics = bundle.metrics
        model_info = ProbabilityModelInfo(
            version=MODEL_VERSION,
            display_name=MODEL_DISPLAY_NAME,
            training_rows=bundle.training_rows,
            training_sha256=bundle.training_sha256,
            recognized_features=list(prediction.recognized_features),
            metrics=ProbabilityModelMetrics(
                validation_rows=metrics.validation_rows,
                mean_absolute_error=metrics.mean_absolute_error,
                root_mean_squared_error=metrics.root_mean_squared_error,
                r2=metrics.r2,
            ),
            target_description=(
                "Surrogate of the published Liu et al. (2024) estimated "
                "subaerial-basalt probability; not the authors' original model"
            ),
        )
        prediction_summary = ProbabilityPredictionSummary(
            predicted_rows=int(predicted_mask.sum()),
            insufficient_feature_rows=int((~predicted_mask).sum()),
            eligible_time_series_rows=eligible_rows,
            sampled_time_series_rows=sampled_rows,
            minimum_features_per_row=MIN_FEATURES_PER_ROW,
        )
        sampled_out_rows = eligible_rows - sampled_rows
        warnings = [
            (
                "Probability source: model-predicted by Liu-2024 surrogate v1; "
                "this is not the authors' original trained model."
            ),
            (
                f"Recognized {len(prediction.recognized_features)} geochemical "
                f"features; predicted {int(predicted_mask.sum()):,} rows."
            ),
        ]
        dropped_rows = int(dataframe.shape[0] - eligible_rows)
        if dropped_rows:
            warnings.append(
                f"Excluded {dropped_rows:,} rows lacking sufficient features, "
                "age, or coordinates."
            )
        if sampled_out_rows:
            warnings.append(
                f"Used a deterministic age-stratified sample of {sampled_rows:,} "
                f"from {eligible_rows:,} eligible rows for the O(n²) spatial-age "
                "weighting step."
            )
        warnings.append(
            "The shaded range is Bootstrap ±2σ; the result is a binned relation, "
            "not a forecast."
        )

        response.message = "Model-predicted time series analysis completed"
        response.source_filename = Path(filename or "dataset").name
        response.age_column = age_column
        response.age_max_column = age_max_column
        response.probability_column = probability_column
        response.latitude_column = latitude_column
        response.longitude_column = longitude_column
        response.probability_source = "model_predicted"
        response.probability_model = model_info
        response.prediction_summary = prediction_summary
        response.summary.original_rows = int(dataframe.shape[0])
        response.summary.usable_rows = sampled_rows
        response.summary.dropped_rows = dropped_rows
        response.summary.sampled_out_rows = sampled_out_rows
        response.warnings = warnings

        selected_index = set(sampled.index)
        sample_id_column = next(
            (
                column
                for column in ("SAMPLE_ID", "SAMPLE ID", "ID")
                if column in dataframe.columns
            ),
            None,
        )
        audit = pd.DataFrame(
            {
                "source_row": np.arange(2, dataframe.shape[0] + 2),
                "predicted_subaerial_probability": prediction.probabilities,
                "available_geochemical_features": prediction.available_feature_count,
                "eligible_for_prediction": predicted_mask,
                "eligible_for_time_series": eligible_mask,
                "selected_for_time_series": dataframe.index.isin(selected_index),
                "model_version": MODEL_VERSION,
            }
        )
        if sample_id_column:
            audit.insert(1, "sample_id", dataframe[sample_id_column].values)
        output_dir = self.jobs_dir / response.job_id / "output"
        prediction_path = output_dir / "predicted_subaerial_probabilities.csv"
        audit.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        response.artifacts.append(
            ArtifactResponse(
                name=prediction_path.name,
                download_url=(
                    f"/api/data-mining/jobs/{response.job_id}/files/"
                    f"{prediction_path.name}"
                ),
                size_bytes=prediction_path.stat().st_size,
            )
        )

        report_path = output_dir / "time_series_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(
            {
                "source_filename": response.source_filename,
                "column_mapping": {
                    "age": age_column,
                    "age_max": age_max_column,
                    "subaerial_probability": probability_column,
                    "latitude": latitude_column,
                    "longitude": longitude_column,
                },
                "probability_source": response.probability_source,
                "probability_model": model_info.model_dump(),
                "prediction_summary": prediction_summary.model_dump(),
                "summary": response.summary.model_dump(),
                "warnings": warnings,
            }
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for artifact in response.artifacts:
            if artifact.name == report_path.name:
                artifact.size_bytes = report_path.stat().st_size
                break
        return response

    @staticmethod
    def _stratified_age_sample(
        dataframe: pd.DataFrame,
        *,
        age_column: str,
        bin_width: float,
        max_rows: int,
    ) -> pd.DataFrame:
        if dataframe.shape[0] <= max_rows:
            return dataframe.copy()
        strata = np.floor(dataframe[age_column].to_numpy(dtype=float) / bin_width)
        counts = pd.Series(strata, index=dataframe.index).value_counts().sort_index()
        ideal = counts * (max_rows / dataframe.shape[0])
        quotas = np.floor(ideal).astype(int).clip(lower=1, upper=counts)
        while int(quotas.sum()) > max_rows:
            candidates = quotas[quotas > 1].sort_values(ascending=False)
            quotas.loc[candidates.index[0]] -= 1
        fractions = (ideal - np.floor(ideal)).sort_values(ascending=False)
        while int(quotas.sum()) < max_rows:
            changed = False
            for key in fractions.index:
                if quotas.loc[key] < counts.loc[key]:
                    quotas.loc[key] += 1
                    changed = True
                    if int(quotas.sum()) == max_rows:
                        break
            if not changed:
                break

        rng = np.random.RandomState(2025)
        selected: list[Any] = []
        stratum_series = pd.Series(strata, index=dataframe.index)
        for key, quota in quotas.items():
            index = stratum_series.index[stratum_series == key].to_numpy()
            selected.extend(rng.choice(index, size=int(quota), replace=False).tolist())
        return dataframe.loc[sorted(selected)].copy()

    def run_time_series(
        self,
        *,
        filename: str | None,
        content: bytes,
        age_column: str,
        age_max_column: str,
        probability_column: str,
        latitude_column: str,
        longitude_column: str,
        age_unit: str = "Ma",
        bin_width: float = 10.0,
        bootstrap_iterations: int = 100,
    ) -> TimeSeriesResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        mapped_columns = self._validate_selected_columns(
            dataframe,
            [
                age_column,
                age_max_column,
                probability_column,
                latitude_column,
                longitude_column,
            ],
        )
        if len(mapped_columns) != 5:
            raise InvalidDatasetError(
                "Time series requires five different mapped columns"
            )
        if age_unit not in {"Ma", "Ga"}:
            raise InvalidDatasetError("Age unit must be Ma or Ga")
        if not math.isfinite(bin_width) or bin_width <= 0:
            raise InvalidDatasetError("Bin width must be a positive finite number")
        if bootstrap_iterations < 10 or bootstrap_iterations > 1000:
            raise InvalidDatasetError(
                "Bootstrap iterations must be between 10 and 1,000"
            )

        numeric_data = dataframe.loc[:, mapped_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        valid_mask = numeric_data.notna().all(axis=1)
        analysis_data = numeric_data.loc[valid_mask].copy()
        usable_rows = int(analysis_data.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        if usable_rows < 10:
            raise InvalidDatasetError(
                "Time series requires at least 10 complete numeric rows"
            )
        if usable_rows > 25_000:
            raise InvalidDatasetError(
                "Time series supports at most 25,000 complete rows in Online mode"
            )
        if (analysis_data[age_column] < 0).any() or (
            analysis_data[age_max_column] < 0
        ).any():
            raise InvalidDatasetError("Age values must be zero or greater")
        probabilities = analysis_data[probability_column]
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise InvalidDatasetError(
                "Subaerial probabilities must be between 0 and 1"
            )
        latitudes = analysis_data[latitude_column]
        if ((latitudes < -90) | (latitudes > 90)).any():
            raise InvalidDatasetError("Latitude values must be between -90 and 90")
        longitudes = analysis_data[longitude_column]
        if ((longitudes < -180) | (longitudes > 180)).any():
            raise InvalidDatasetError(
                "Longitude values must be between -180 and 180"
            )
        maximum_age = float(analysis_data[age_column].max())
        if maximum_age <= 0:
            raise InvalidDatasetError(
                "Time series requires at least one age greater than zero"
            )
        estimated_bin_count = int(math.ceil(maximum_age / bin_width))
        if estimated_bin_count > 5_000:
            raise InvalidDatasetError(
                "The selected bin width would create more than 5,000 age bins"
            )

        internal_bin_width = float(bin_width)
        if age_unit == "Ga":
            analysis_data[age_column] *= 1000.0
            analysis_data[age_max_column] *= 1000.0
            internal_bin_width *= 1000.0

        age_x, mean_values, uncertainty_values = compute_subaerial_proportion(
            analysis_data,
            bin_width=internal_bin_width,
            n_iter=bootstrap_iterations,
            age_col=age_column,
            age_max_col=age_max_column,
            prob_col=probability_column,
            lat_col=latitude_column,
            lon_col=longitude_column,
        )
        display_ages = age_x / 1000.0 if age_unit == "Ga" else age_x
        bins = [
            TimeSeriesBinItem(
                age=float(age),
                mean_proportion=(
                    float(mean) if math.isfinite(float(mean)) else None
                ),
                uncertainty_2sigma=(
                    float(uncertainty)
                    if math.isfinite(float(uncertainty))
                    else None
                ),
            )
            for age, mean, uncertainty in zip(
                display_ages,
                mean_values,
                uncertainty_values,
            )
        ]
        populated_bins = sum(item.mean_proportion is not None for item in bins)
        if populated_bins == 0:
            raise InvalidDatasetError(
                "Time series produced no populated age bins; increase bin width"
            )
        summary = TimeSeriesSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            bin_count=len(bins),
            populated_bins=populated_bins,
        )
        warnings = [
            (
                f"时间序列分析前删除了 {dropped_rows} 行缺少必需数值的记录。"
                if dropped_rows
                else "所有数据行均可用于时间序列分析。"
            ),
            "阴影范围表示 Bootstrap 结果的 ±2σ；结果反映分箱统计关系，不代表时间序列预测。",
        ]

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        results_path = output_dir / "subaerial_proportion.csv"
        figure_path = output_dir / "subaerial_proportion.svg"
        report_path = output_dir / "time_series_report.json"
        pd.DataFrame(
            [item.model_dump() for item in bins]
        ).to_csv(results_path, index=False, encoding="utf-8-sig")
        self._write_time_series_svg(
            figure_path,
            bins=bins,
            age_unit=age_unit,
            bin_width=bin_width,
            bootstrap_iterations=bootstrap_iterations,
        )
        report_payload = {
            "report_version": "v080-time-series-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "column_mapping": {
                "age": age_column,
                "age_max": age_max_column,
                "subaerial_probability": probability_column,
                "latitude": latitude_column,
                "longitude": longitude_column,
            },
            "age_unit": age_unit,
            "bin_width": bin_width,
            "bootstrap_iterations": bootstrap_iterations,
            "random_state": 2025,
            "probability_source": "uploaded",
            "summary": summary.model_dump(),
            "bins": [item.model_dump() for item in bins],
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        artifacts = [
            ArtifactResponse(
                name=path.name,
                download_url=f"/api/data-mining/jobs/{job_id}/files/{path.name}",
                size_bytes=path.stat().st_size,
            )
            for path in (results_path, figure_path, report_path)
        ]
        return TimeSeriesResponse(
            job_id=job_id,
            status="success",
            message="Time series analysis completed",
            source_filename=Path(filename or "dataset").name,
            age_column=age_column,
            age_max_column=age_max_column,
            probability_column=probability_column,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            age_unit=age_unit,
            bin_width=bin_width,
            bootstrap_iterations=bootstrap_iterations,
            random_state=2025,
            probability_source="uploaded",
            summary=summary,
            bins=bins,
            warnings=warnings,
            artifacts=artifacts,
        )

    @staticmethod
    def _write_time_series_svg(
        path: Path,
        *,
        bins: list[TimeSeriesBinItem],
        age_unit: str,
        bin_width: float,
        bootstrap_iterations: int,
    ) -> None:
        valid = [
            item
            for item in bins
            if item.mean_proportion is not None
            and item.uncertainty_2sigma is not None
        ]
        ages = [item.age for item in valid]
        minimum_age = min(ages)
        maximum_age = max(ages)
        age_span = maximum_age - minimum_age or 1.0
        width, height = 960, 520
        left, right, top, bottom = 92, 36, 58, 76
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_position(age: float) -> float:
            return left + (maximum_age - age) / age_span * plot_width

        def y_position(value: float) -> float:
            clipped = min(100.0, max(0.0, value))
            return top + (100.0 - clipped) / 100.0 * plot_height

        line_points = " ".join(
            f"{x_position(item.age):.2f},{y_position(item.mean_proportion or 0):.2f}"
            for item in valid
        )
        upper = [
            (
                x_position(item.age),
                y_position(
                    (item.mean_proportion or 0)
                    + (item.uncertainty_2sigma or 0)
                ),
            )
            for item in valid
        ]
        lower = [
            (
                x_position(item.age),
                y_position(
                    (item.mean_proportion or 0)
                    - (item.uncertainty_2sigma or 0)
                ),
            )
            for item in reversed(valid)
        ]
        band_points = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in [*upper, *lower]
        )
        grid_lines = []
        for value in range(0, 101, 20):
            y = y_position(float(value))
            grid_lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                f'y2="{y:.2f}" class="grid"/>'
            )
            grid_lines.append(
                f'<text x="{left-14}" y="{y+5:.2f}" text-anchor="end" '
                f'class="tick">{value}</text>'
            )
        x_ticks = []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            age = maximum_age - age_span * fraction
            x = left + plot_width * fraction
            x_ticks.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{height-bottom}" class="grid"/>'
            )
            x_ticks.append(
                f'<text x="{x:.2f}" y="{height-bottom+26}" '
                f'text-anchor="middle" class="tick">{age:g}</text>'
            )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#18323a}} .title{{font-size:22px;font-weight:700}}
.subtitle{{font-size:13px;fill:#5c7078}} .axis{{stroke:#18323a;stroke-width:1.4}}
.grid{{stroke:#dbe5e7;stroke-width:1;stroke-dasharray:4 5}} .tick{{font-size:12px;fill:#526970}}
.curve{{fill:none;stroke:#e5674f;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}}
.band{{fill:#e5674f;fill-opacity:.18;stroke:none}} .label{{font-size:14px;font-weight:600}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">Estimated proportion of subaerial basalts</text>
<text x="{left}" y="49" class="subtitle">Bin width {bin_width:g} {age_unit} · {bootstrap_iterations} bootstrap iterations · shaded area ±2σ</text>
{''.join(grid_lines)}{''.join(x_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
<polygon points="{band_points}" class="band"/>
<polyline points="{line_points}" class="curve"/>
<text x="{left + plot_width/2:.2f}" y="{height-22}" text-anchor="middle" class="label">Age ({age_unit})</text>
<text x="24" y="{top + plot_height/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top + plot_height/2:.2f})">Estimated proportion (%)</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

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
