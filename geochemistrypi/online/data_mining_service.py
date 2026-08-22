"""Lightweight Data Mining workflows for the Online application."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geochemistrypi._version import __version__

from geochemistrypi.data_mining.process.time_series import (
    compute_subaerial_proportion,
)

from .data_mining_models import (
    ANOMALY_DETECTION_MODELS,
    CLASSIFICATION_MODELS,
    CLUSTERING_MODELS,
    DIMENSIONALITY_REDUCTION_MODELS,
    REGRESSION_MODELS,
    configure_model,
    extract_linear_parameters,
    get_anomaly_detection_model,
    get_classification_model,
    get_clustering_model,
    get_dimensionality_reduction_model,
    get_regression_model,
    get_hyperparameters,
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
    HyperparameterItem,
    DataPreprocessingResponse,
    DataPreprocessingSummary,
    DatasetProfileResponse,
    DatasetProfileSummary,
    DimensionalityReductionMetrics,
    DimensionalityReductionResponse,
    DimensionalityReductionSummary,
    ModelInferenceResponse,
    ModelInferenceSummary,
    ModelComparisonItem,
    ModelComparisonResponse,
    CrossValidationMetricItem,
    CrossValidationResult,
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
from .limits import MAX_UPLOAD_BYTES
from .subaerial_probability import (
    MIN_FEATURES_PER_ROW,
    MODEL_DISPLAY_NAME,
    MODEL_VERSION,
    predict_subaerial_probability,
)


class DataMiningService:
    """Run small, non-interactive Data Mining jobs without the legacy web stack."""

    supported_suffixes = {".xlsx", ".csv"}
    supervised_pipeline_filename = "trained_pipeline.joblib"
    supervised_pipeline_schema = "geochemistrypi-online-supervised-pipeline-v1"
    preprocessing_strategies = {
        "keep",
        "drop_rows",
        "fill_mean",
        "fill_median",
        "fill_mode",
    }
    anomaly_reproduction_profiles = {
        "general",
        "sharapatov_2025_figure_3a",
        "zhu_2024_figure_8a",
    }
    sharapatov_figure3a_columns = (
        "source_row_id",
        "Name",
        "PC1_full_svd_reference",
        "PC2_full_svd_reference",
        "PC1_notebook_auto_solver_rs42",
        "PC2_notebook_auto_solver_rs42",
        "if_prediction_notebook_raw_features",
        "if_anomaly_notebook_raw_features",
        "if_score_samples_notebook_raw_features",
        "if_decision_notebook_raw_features",
        "if_anomaly_scaled_features_contamination_0_05",
        "if_anomaly_pca5_contamination_0_05",
        "if_anomaly_online_scaled_contamination_auto",
        "if_score_samples_online_scaled_contamination_auto",
        "if_decision_online_scaled_contamination_auto",
    )
    zhu_figure8a_ratio_columns = (
        "Na_Cl_ratio",
        "Na_F_ratio",
        "Na_SO4_ratio",
        "F_Cl_ratio",
        "SO4_Cl_ratio",
    )
    zhu_figure8a_series_columns = (
        "Date",
        *zhu_figure8a_ratio_columns,
        "Published_LOF_Outlier_P0_08",
    )
    zhu_figure8a_earthquake_columns = (
        "Event_ID",
        "Event_DateTime",
        "Longitude_deg_E",
        "Latitude_deg_N",
        "Epicentral_Depth_km",
        "Magnitude",
        "GA_Epicentral_Distance_km",
        "Use_in_Figure8a",
        "Marker_Criterion",
        "Table_S1_Source_Row",
    )

    def __init__(
        self,
        runtime_dir: Path,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
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
                        "已接入 v0.8 的 15 个回归模型，包括线性/正则化模型、"
                        "树集成、KNN、MLP、SGD、SVR 和 XGBoost，并完成固定随机"
                        "种子划分、R²/MAE/RMSE、预测、Pipeline 与推理验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "回归指标",
                        "模型系数（线性模型）",
                        "预测结果 CSV",
                        "JSON 模型报告",
                        "已训练 Pipeline",
                        "Application Data 推理",
                    ],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                            hyperparameters=[
                                HyperparameterItem(
                                    name=parameter.name,
                                    display_name=parameter.display_name,
                                    description=parameter.description,
                                    value_type=parameter.value_type,
                                    default=parameter.default,
                                    minimum=parameter.minimum,
                                    maximum=parameter.maximum,
                                    step=parameter.step,
                                    options=list(parameter.options),
                                )
                                for parameter in get_hyperparameters(
                                    "regression", definition.name
                                )
                            ],
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
                        "Extra-Trees、MLP、Gradient Boosting、KNN、SGD、AdaBoost "
                        "和 XGBoost，并完成文本标签编码、分层训练测试划分、"
                        "Accuracy/Precision/Recall/F1、混淆矩阵和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "分类指标",
                        "混淆矩阵",
                        "预测结果 CSV",
                        "JSON 模型报告",
                        "已训练 Pipeline",
                        "Application Data 推理",
                    ],
                    methods=[
                        DataMiningMethodItem(
                            name=definition.name,
                            display_name=definition.display_name,
                            description=definition.description,
                            hyperparameters=[
                                HyperparameterItem(
                                    name=parameter.name,
                                    display_name=parameter.display_name,
                                    description=parameter.description,
                                    value_type=parameter.value_type,
                                    default=parameter.default,
                                    minimum=parameter.minimum,
                                    maximum=parameter.maximum,
                                    step=parameter.step,
                                    options=list(parameter.options),
                                )
                                for parameter in get_hyperparameters(
                                    "classification", definition.name
                                )
                            ],
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
                        "PCA/异常分数诊断图、异常样品预览和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "正常/异常样品统计",
                        "异常分数与标签",
                        "异常检测诊断图 SVG/PNG",
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
                        "年龄分箱、固定随机种子 Bootstrap、2σ 不确定度、散点误差图和结果下载验证。"
                    ),
                    input_formats=[".xlsx", ".csv"],
                    outputs=[
                        "陆上玄武岩比例散点误差图",
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

    @staticmethod
    def _build_supervised_pipeline(estimator: Any) -> Pipeline:
        """Keep training and later application inference in one fitted object."""
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    @staticmethod
    def _validate_cross_validation_folds(
        folds: int,
        row_count: int,
        *,
        minimum_class_rows: int | None = None,
    ) -> None:
        if folds == 0:
            return
        if not 2 <= folds <= 10:
            raise InvalidDatasetError(
                "Cross-validation folds must be 0 (disabled) or between 2 and 10"
            )
        if folds > row_count:
            raise InvalidDatasetError(
                "Cross-validation folds cannot exceed the number of usable rows"
            )
        if minimum_class_rows is not None and folds > minimum_class_rows:
            raise InvalidDatasetError(
                "For stratified cross-validation, every class must contain at "
                f"least {folds} rows"
            )
        if row_count < folds * 2:
            raise InvalidDatasetError(
                "Cross-validation requires at least two validation rows per fold"
            )

    def _cross_validate_supervised(
        self,
        *,
        task_type: str,
        model: Pipeline,
        features: pd.DataFrame,
        target: pd.Series,
        folds: int,
    ) -> CrossValidationResult | None:
        if folds == 0:
            return None
        if task_type == "regression":
            splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
            scoring: dict[str, Any] = {
                "r2": "r2",
                "mean_absolute_error": "neg_mean_absolute_error",
                "root_mean_squared_error": "neg_root_mean_squared_error",
            }
            labels = {
                "r2": ("R²", True, 1.0),
                "mean_absolute_error": ("Mean absolute error", False, -1.0),
                "root_mean_squared_error": ("Root mean squared error", False, -1.0),
            }
            strategy = "Shuffled K-Fold"
        else:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            scoring = {
                "accuracy": "accuracy",
                "precision_macro": make_scorer(
                    precision_score, average="macro", zero_division=0
                ),
                "recall_macro": make_scorer(
                    recall_score, average="macro", zero_division=0
                ),
                "f1_macro": make_scorer(
                    f1_score, average="macro", zero_division=0
                ),
            }
            labels = {
                "accuracy": ("Accuracy", True, 1.0),
                "precision_macro": ("Macro precision", True, 1.0),
                "recall_macro": ("Macro recall", True, 1.0),
                "f1_macro": ("Macro F1", True, 1.0),
            }
            strategy = "Shuffled Stratified K-Fold"

        scores = cross_validate(
            model,
            features,
            target,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        metrics = []
        for name, (display_name, higher_is_better, multiplier) in labels.items():
            values = np.asarray(scores[f"test_{name}"], dtype=float) * multiplier
            metrics.append(
                CrossValidationMetricItem(
                    name=name,
                    display_name=display_name,
                    mean=float(np.mean(values)),
                    standard_deviation=float(np.std(values, ddof=0)),
                    higher_is_better=higher_is_better,
                )
            )
        return CrossValidationResult(
            folds=folds,
            strategy=strategy,
            random_state=42,
            metrics=metrics,
        )

    def _save_supervised_pipeline(
        self,
        path: Path,
        *,
        pipeline: Pipeline,
        task_type: str,
        model_name: str,
        model_display_name: str,
        target_column: str,
        feature_columns: list[str],
        source_filename: str,
    ) -> None:
        joblib.dump(
            {
                "schema_version": self.supervised_pipeline_schema,
                "software_version": __version__,
                "scikit_learn_version": sklearn_version,
                "pandas_version": pd.__version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "task_type": task_type,
                "model": model_name,
                "model_display_name": model_display_name,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "source_filename": source_filename,
                "pipeline": pipeline,
            },
            path,
            compress=3,
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
        hyperparameters: dict[str, Any] | None = None,
        cross_validation_folds: int = 0,
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
        self._validate_cross_validation_folds(
            cross_validation_folds,
            usable_rows,
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

        try:
            estimator = configure_model(
                "regression", model_definition, hyperparameters
            )
        except (TypeError, ValueError) as exc:
            raise InvalidDatasetError(str(exc)) from exc
        model = self._build_supervised_pipeline(estimator)
        cross_validation = self._cross_validate_supervised(
            task_type="regression",
            model=model,
            features=feature_data,
            target=target,
            folds=cross_validation_folds,
        )
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

        linear_parameters = extract_linear_parameters(model, features)
        if linear_parameters is None:
            intercept = None
            coefficients: list[RegressionCoefficientItem] = []
            equation = None
        else:
            intercept, coefficient_names, coefficient_values = linear_parameters
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
        pipeline_path = output_dir / self.supervised_pipeline_filename
        prediction_frame.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        self._save_supervised_pipeline(
            pipeline_path,
            pipeline=model,
            task_type="regression",
            model_name=model_name,
            model_display_name=model_definition.display_name,
            target_column=target_column,
            feature_columns=features,
            source_filename=Path(filename or "dataset").name,
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
            "hyperparameters": hyperparameters or {},
            "cross_validation": (
                cross_validation.model_dump() if cross_validation else None
            ),
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
            "pipeline_artifact": pipeline_path.name,
            "pipeline_schema_version": self.supervised_pipeline_schema,
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
            hyperparameters=hyperparameters or {},
            cross_validation=cross_validation,
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
                ArtifactResponse(
                    name=pipeline_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{pipeline_path.name}"
                    ),
                    size_bytes=pipeline_path.stat().st_size,
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
        hyperparameters: dict[str, Any] | None = None,
        cross_validation_folds: int = 0,
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
        self._validate_cross_validation_folds(
            cross_validation_folds,
            usable_rows,
            minimum_class_rows=int(class_counts.min()),
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

        try:
            estimator = configure_model(
                "classification", model_definition, hyperparameters
            )
        except (TypeError, ValueError) as exc:
            raise InvalidDatasetError(str(exc)) from exc
        model = self._build_supervised_pipeline(estimator)
        cross_validation = self._cross_validate_supervised(
            task_type="classification",
            model=model,
            features=feature_data,
            target=target,
            folds=cross_validation_folds,
        )
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
        pipeline_path = output_dir / self.supervised_pipeline_filename
        prediction_frame.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        self._save_supervised_pipeline(
            pipeline_path,
            pipeline=model,
            task_type="classification",
            model_name=model_name,
            model_display_name=model_definition.display_name,
            target_column=target_column,
            feature_columns=features,
            source_filename=Path(filename or "dataset").name,
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
            "hyperparameters": hyperparameters or {},
            "cross_validation": (
                cross_validation.model_dump() if cross_validation else None
            ),
            "classes": classes,
            "summary": summary.model_dump(),
            "metrics": metrics.model_dump(),
            "confusion_matrix": [item.model_dump() for item in confusion_items],
            "prediction_preview": preview,
            "warnings": warnings,
            "pipeline_artifact": pipeline_path.name,
            "pipeline_schema_version": self.supervised_pipeline_schema,
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
            hyperparameters=hyperparameters or {},
            cross_validation=cross_validation,
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
                ArtifactResponse(
                    name=pipeline_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{pipeline_path.name}"
                    ),
                    size_bytes=pipeline_path.stat().st_size,
                ),
            ],
        )

    def run_model_comparison(
        self,
        *,
        filename: str | None,
        content: bytes,
        task_type: str,
        target_column: str,
        feature_columns: list[str],
        model_names: list[str],
        cross_validation_folds: int = 5,
        hyperparameters: dict[str, dict[str, Any]] | None = None,
    ) -> ModelComparisonResponse:
        if task_type not in {"regression", "classification"}:
            raise InvalidDatasetError(
                "Task type must be regression or classification"
            )
        if not isinstance(model_names, list) or len(model_names) < 2:
            raise InvalidDatasetError("Select at least two models to compare")
        if len(model_names) != len(set(model_names)):
            raise InvalidDatasetError("Comparison models must be unique")
        registry = (
            REGRESSION_MODELS
            if task_type == "regression"
            else CLASSIFICATION_MODELS
        )
        unknown_models = [name for name in model_names if name not in registry]
        if unknown_models:
            raise InvalidDatasetError(
                "Unknown comparison model(s): " + ", ".join(unknown_models)
            )
        if len(model_names) > len(registry):
            raise InvalidDatasetError("Too many comparison models selected")
        hyperparameters = hyperparameters or {}
        if not isinstance(hyperparameters, dict):
            raise InvalidDatasetError("Hyperparameters must be a JSON object")
        unknown_parameter_models = sorted(set(hyperparameters) - set(model_names))
        if unknown_parameter_models:
            raise InvalidDatasetError(
                "Hyperparameters were provided for unselected model(s): "
                + ", ".join(unknown_parameter_models)
            )

        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]
        if target_column not in dataframe.columns:
            raise InvalidDatasetError(f"Unknown target column: {target_column}")
        features = self._validate_selected_columns(dataframe, feature_columns)
        if target_column in features:
            raise InvalidDatasetError(
                "The target column cannot also be a feature column"
            )
        non_numeric_features = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_bool_dtype(dataframe[column])
        ]
        if non_numeric_features:
            raise InvalidDatasetError(
                "Model comparison requires numeric feature columns: "
                + ", ".join(non_numeric_features)
            )
        if task_type == "regression" and (
            not pd.api.types.is_numeric_dtype(dataframe[target_column])
            or pd.api.types.is_bool_dtype(dataframe[target_column])
        ):
            raise InvalidDatasetError(
                "Regression comparison requires a numeric target column"
            )

        model_data = (
            dataframe.loc[:, [*features, target_column]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        usable_rows = int(model_data.shape[0])
        minimum_rows = 10 if task_type == "regression" else 12
        if usable_rows < minimum_rows:
            raise InvalidDatasetError(
                f"{task_type.title()} comparison requires at least "
                f"{minimum_rows} complete rows"
            )
        feature_data = model_data.loc[:, features].astype(float)
        if task_type == "regression":
            target = model_data[target_column].astype(float)
            if target.nunique(dropna=True) < 2:
                raise InvalidDatasetError(
                    "The target column must contain at least two distinct values"
                )
            minimum_class_rows = None
        else:
            target = model_data[target_column].astype(str)
            class_counts = target.value_counts()
            if int(class_counts.shape[0]) < 2:
                raise InvalidDatasetError(
                    "The target column must contain at least two classes"
                )
            minimum_class_rows = int(class_counts.min())
        self._validate_cross_validation_folds(
            cross_validation_folds,
            usable_rows,
            minimum_class_rows=minimum_class_rows,
        )
        if cross_validation_folds == 0:
            raise InvalidDatasetError(
                "Model comparison requires cross-validation with 2 to 10 folds"
            )

        comparison_results: list[ModelComparisonItem] = []
        successful_results: list[ModelComparisonItem] = []
        failed_results: list[ModelComparisonItem] = []
        primary_metric = "r2" if task_type == "regression" else "f1_macro"
        for model_name in model_names:
            definition = registry[model_name]
            configured_parameters = hyperparameters.get(model_name, {})
            try:
                estimator = configure_model(
                    task_type,
                    definition,
                    configured_parameters,
                )
                model = self._build_supervised_pipeline(estimator)
                cross_validation = self._cross_validate_supervised(
                    task_type=task_type,
                    model=model,
                    features=feature_data,
                    target=target,
                    folds=cross_validation_folds,
                )
                assert cross_validation is not None
                primary_score = next(
                    metric.mean
                    for metric in cross_validation.metrics
                    if metric.name == primary_metric
                )
                successful_results.append(
                    ModelComparisonItem(
                        rank=0,
                        model=model_name,
                        model_display_name=definition.display_name,
                        status="success",
                        primary_score=primary_score,
                        metrics=cross_validation.metrics,
                        hyperparameters=configured_parameters,
                    )
                )
            except Exception as exc:
                failed_results.append(
                    ModelComparisonItem(
                        rank=0,
                        model=model_name,
                        model_display_name=definition.display_name,
                        status="failed",
                        hyperparameters=configured_parameters,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        if not successful_results:
            errors = "; ".join(
                f"{item.model_display_name}: {item.error}"
                for item in failed_results
            )
            raise InvalidDatasetError("All comparison models failed. " + errors)

        successful_results.sort(
            key=lambda item: item.primary_score
            if item.primary_score is not None
            else -math.inf,
            reverse=True,
        )
        for rank, item in enumerate(successful_results, start=1):
            item.rank = rank
        for rank, item in enumerate(
            failed_results,
            start=len(successful_results) + 1,
        ):
            item.rank = rank
        comparison_results = [*successful_results, *failed_results]
        warnings = []
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        if dropped_rows:
            warnings.append(
                f"训练前删除了 {dropped_rows} 行含缺失值或无穷值的记录。"
            )
        if failed_results:
            warnings.append(
                f"{len(failed_results)} 个模型未能完成；排名仅依据成功模型。"
            )

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        csv_path = output_dir / "model_comparison.csv"
        report_path = output_dir / "model_comparison_report.json"
        csv_rows = []
        for item in comparison_results:
            row: dict[str, Any] = {
                "rank": item.rank,
                "model": item.model,
                "model_display_name": item.model_display_name,
                "status": item.status,
                "error": item.error,
            }
            for metric in item.metrics:
                row[f"{metric.name}_mean"] = metric.mean
                row[f"{metric.name}_std"] = metric.standard_deviation
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_payload = {
            "report_version": "supervised-model-comparison-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": features,
            "usable_rows": usable_rows,
            "cross_validation_folds": cross_validation_folds,
            "cross_validation_strategy": (
                "Shuffled K-Fold"
                if task_type == "regression"
                else "Shuffled Stratified K-Fold"
            ),
            "random_state": 42,
            "comparison_metric": primary_metric,
            "best_model": successful_results[0].model,
            "results": [item.model_dump() for item in comparison_results],
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ModelComparisonResponse(
            job_id=job_id,
            status="success",
            message=(
                f"Compared {len(successful_results)} models with "
                f"{cross_validation_folds}-fold cross-validation"
            ),
            source_filename=Path(filename or "dataset").name,
            task_type=task_type,
            target_column=target_column,
            feature_columns=features,
            cross_validation_folds=cross_validation_folds,
            comparison_metric=primary_metric,
            best_model=successful_results[0].model,
            results=comparison_results,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=csv_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{csv_path.name}"
                    ),
                    size_bytes=csv_path.stat().st_size,
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

    def run_model_inference(
        self,
        *,
        training_job_id: str,
        filename: str | None,
        content: bytes,
    ) -> ModelInferenceResponse:
        if (
            len(training_job_id) != 32
            or any(character not in "0123456789abcdef" for character in training_job_id)
        ):
            raise InvalidDatasetError("Training Job ID is invalid")

        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        try:
            pipeline_path = self.resolve_artifact(
                training_job_id,
                self.supervised_pipeline_filename,
            )
        except FileNotFoundError as exc:
            raise InvalidDatasetError(
                "The training job does not contain a saved supervised Pipeline"
            ) from exc

        try:
            bundle = joblib.load(pipeline_path)
        except Exception as exc:
            raise InvalidDatasetError("The saved Pipeline cannot be loaded") from exc
        if (
            not isinstance(bundle, dict)
            or bundle.get("schema_version") != self.supervised_pipeline_schema
        ):
            raise InvalidDatasetError("The saved Pipeline format is not supported")

        task_type = bundle.get("task_type")
        if task_type not in {"regression", "classification"}:
            raise InvalidDatasetError("The saved Pipeline is not a supervised model")
        pipeline = bundle.get("pipeline")
        if not isinstance(pipeline, Pipeline):
            raise InvalidDatasetError("The saved model does not contain a valid Pipeline")
        feature_columns = [str(column) for column in bundle.get("feature_columns", [])]
        if not feature_columns:
            raise InvalidDatasetError("The saved Pipeline has no feature definition")
        missing_columns = [column for column in feature_columns if column not in dataframe.columns]
        if missing_columns:
            raise InvalidDatasetError(
                "Application Data is missing required feature columns: "
                + ", ".join(missing_columns)
            )

        feature_data = dataframe.loc[:, feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        predictable_mask = feature_data.notna().any(axis=1)
        predicted_rows = int(predictable_mask.sum())
        excluded_rows = int((~predictable_mask).sum())
        if predicted_rows == 0:
            raise InvalidDatasetError(
                "Application Data contains no row with a usable numeric feature"
            )
        imputed_rows = int(
            feature_data.loc[predictable_mask].isna().any(axis=1).sum()
        )
        predicted = pipeline.predict(feature_data.loc[predictable_mask])

        output_frame = dataframe.copy()
        source_row_column = self._unique_output_column(output_frame, "source_row")
        prediction_column = self._unique_output_column(
            output_frame,
            f"predicted_{bundle['target_column']}",
        )
        status_column = self._unique_output_column(output_frame, "inference_status")
        output_frame.insert(
            0,
            source_row_column,
            np.arange(2, dataframe.shape[0] + 2),
        )
        output_frame[prediction_column] = None
        output_frame[status_column] = "excluded_no_numeric_features"
        output_frame.loc[predictable_mask, prediction_column] = np.asarray(predicted)
        output_frame.loc[predictable_mask, status_column] = "predicted"
        if imputed_rows:
            imputed_mask = predictable_mask & feature_data.isna().any(axis=1)
            output_frame.loc[imputed_mask, status_column] = "predicted_with_imputation"

        preview = [
            {
                str(column): self._json_value(value)
                for column, value in row.items()
            }
            for row in output_frame.head(20).to_dict(orient="records")
        ]
        warnings: list[str] = []
        if imputed_rows:
            warnings.append(
                f"{imputed_rows} application rows contained missing or non-numeric feature "
                "values and were imputed with training-set medians."
            )
        if excluded_rows:
            warnings.append(
                f"{excluded_rows} application rows were excluded because all required "
                "features were missing or non-numeric."
            )
        if not warnings:
            warnings.append("All application rows were predicted without imputation.")

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        predictions_path = output_dir / "application_predictions.csv"
        report_path = output_dir / "application_inference_report.json"
        output_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        summary = ModelInferenceSummary(
            original_rows=int(dataframe.shape[0]),
            predicted_rows=predicted_rows,
            excluded_rows=excluded_rows,
            imputed_rows=imputed_rows,
            feature_count=len(feature_columns),
        )
        report_payload = {
            "report_version": "application-inference-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_job_id": training_job_id,
            "source_filename": Path(filename or "application-data").name,
            "task_type": task_type,
            "model": bundle["model"],
            "model_display_name": bundle["model_display_name"],
            "target_column": bundle["target_column"],
            "feature_columns": feature_columns,
            "prediction_column": prediction_column,
            "pipeline_schema_version": bundle["schema_version"],
            "training_software_version": bundle["software_version"],
            "training_scikit_learn_version": bundle.get("scikit_learn_version"),
            "training_pandas_version": bundle.get("pandas_version"),
            "inference_software_version": __version__,
            "summary": summary.model_dump(),
            "prediction_preview": preview,
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ModelInferenceResponse(
            job_id=job_id,
            training_job_id=training_job_id,
            status="success",
            message=f"{bundle['model_display_name']} application inference completed",
            source_filename=Path(filename or "application-data").name,
            task_type=task_type,
            model=bundle["model"],
            model_display_name=bundle["model_display_name"],
            target_column=bundle["target_column"],
            feature_columns=feature_columns,
            prediction_column=prediction_column,
            pipeline_schema_version=bundle["schema_version"],
            software_version=bundle["software_version"],
            summary=summary,
            preview=preview,
            warnings=warnings,
            artifacts=[
                ArtifactResponse(
                    name=predictions_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{predictions_path.name}"
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

    @staticmethod
    def _unique_output_column(dataframe: pd.DataFrame, base_name: str) -> str:
        candidate = base_name
        suffix = 2
        while candidate in dataframe.columns:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

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
        contamination: str | float = "auto",
        reproduction_profile: str = "general",
    ) -> AnomalyDetectionResponse:
        normalized_profile = str(reproduction_profile).strip().lower()
        if normalized_profile not in self.anomaly_reproduction_profiles:
            choices = ", ".join(sorted(self.anomaly_reproduction_profiles))
            raise InvalidDatasetError(
                "Unknown anomaly reproduction profile "
                f"'{reproduction_profile}'. Choose one of: {choices}"
            )
        normalized_contamination = self._validate_anomaly_contamination(
            contamination
        )
        if normalized_profile == "sharapatov_2025_figure_3a":
            if model_name != "isolation_forest":
                raise InvalidDatasetError(
                    "Sharapatov et al. (2025) Figure 3a requires "
                    "the isolation_forest model"
                )
            if normalized_contamination == "auto":
                normalized_contamination = 0.05
            elif not math.isclose(normalized_contamination, 0.05, abs_tol=1e-12):
                raise InvalidDatasetError(
                    "Sharapatov et al. (2025) Figure 3a requires "
                    "contamination = 0.05"
                )
        elif normalized_profile == "zhu_2024_figure_8a":
            if model_name != "local_outlier_factor":
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a requires "
                    "the local_outlier_factor model"
                )
            if normalized_contamination == "auto":
                normalized_contamination = 0.08
            elif not math.isclose(normalized_contamination, 0.08, abs_tol=1e-12):
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a requires contamination = 0.08"
                )
        try:
            model_definition = get_anomaly_detection_model(model_name)
        except ValueError as exc:
            raise InvalidDatasetError(str(exc)) from exc
        suffix = self._validate_upload(filename, content)
        if (
            normalized_profile == "sharapatov_2025_figure_3a"
            and suffix != ".xlsx"
        ):
            raise InvalidDatasetError(
                "Sharapatov et al. (2025) Figure 3a reproduction requires an "
                "XLSX workbook containing the Figure3a_Data audit sheet"
            )
        if normalized_profile == "zhu_2024_figure_8a" and suffix != ".xlsx":
            raise InvalidDatasetError(
                "Zhu et al. (2024) Figure 8a reproduction requires an XLSX "
                "workbook containing Figure8a_Series and Earthquakes sheets"
            )
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        sharapatov_reference: pd.DataFrame | None = None
        zhu_reference: tuple[pd.DataFrame, pd.DataFrame] | None = None
        if normalized_profile == "sharapatov_2025_figure_3a":
            sharapatov_reference = self._read_sharapatov_figure3a_reference(
                content
            )
        elif normalized_profile == "zhu_2024_figure_8a":
            zhu_reference = self._read_zhu_figure8a_reference(content)

        features = self._validate_selected_columns(dataframe, feature_columns)
        reserved_output_columns = {
            "source_row",
            "visualization_x",
            "visualization_y",
            "visualization_observation",
            "anomaly_label",
            "is_anomaly",
            "anomaly_score",
        }
        conflicting_columns = [
            column for column in features if column in reserved_output_columns
        ]
        if conflicting_columns:
            raise InvalidDatasetError(
                "Anomaly detection feature columns conflict with reserved output "
                "columns: " + ", ".join(conflicting_columns)
            )
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

        if normalized_profile == "sharapatov_2025_figure_3a":
            if dataframe.shape[0] != 3_112 or len(features) != 138:
                raise InvalidDatasetError(
                    "Sharapatov et al. (2025) Figure 3a requires the audited "
                    "3,112-row dataset with exactly 138 selected features"
                )
            if usable_rows != 3_112 or dropped_rows != 0:
                raise InvalidDatasetError(
                    "Sharapatov et al. (2025) Figure 3a requires 3,112 complete "
                    "finite feature rows"
                )
            assert sharapatov_reference is not None
            self._validate_sharapatov_figure3a_input_alignment(
                dataframe=dataframe,
                model_index=model_data.index,
                reference=sharapatov_reference,
            )
        elif normalized_profile == "zhu_2024_figure_8a":
            if set(features) != set(self.zhu_figure8a_ratio_columns) or len(
                features
            ) != len(self.zhu_figure8a_ratio_columns):
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a requires exactly these five "
                    "selected features: "
                    + ", ".join(self.zhu_figure8a_ratio_columns)
                )
            if dataframe.shape[0] != 302 or usable_rows != 302 or dropped_rows != 0:
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a requires 302 complete finite "
                    "GA ratio observations"
                )
            if "Date" not in dataframe.columns:
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a requires a Date column in "
                    "the Online_Input sheet"
                )
            assert zhu_reference is not None
            self._validate_zhu_figure8a_input_alignment(
                dataframe=dataframe,
                model_index=model_data.index,
                series=zhu_reference[0],
            )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        model = model_definition.factory(usable_rows, normalized_contamination)
        labels = np.asarray(model.fit_predict(scaled_features), dtype=int)
        if model_name == "local_outlier_factor":
            anomaly_scores = -np.asarray(
                model.negative_outlier_factor_,
                dtype=float,
            )
            decision_threshold = -float(model.offset_)
        else:
            anomaly_scores = -np.asarray(
                model.decision_function(scaled_features),
                dtype=float,
            )
            decision_threshold = 0.0
        if not np.isfinite(anomaly_scores).all():
            raise InvalidDatasetError(
                f"{model_definition.display_name} returned non-finite anomaly scores"
            )

        if len(features) >= 2:
            projection_solver = (
                "full"
                if normalized_profile == "sharapatov_2025_figure_3a"
                else "randomized"
            )
            projection_model = PCA(
                n_components=2,
                svd_solver=projection_solver,
                random_state=(42 if projection_solver == "randomized" else None),
            )
            visualization_coordinates = projection_model.fit_transform(
                scaled_features
            )
            explained_variance_ratio = [
                float(value)
                for value in projection_model.explained_variance_ratio_
            ]
            visualization_kind = "pca"
            visualization_x_label = (
                f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)"
            )
            visualization_y_label = (
                f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)"
            )
        else:
            projection_solver = None
            visualization_coordinates = np.column_stack(
                (scaled_features[:, 0], anomaly_scores)
            )
            explained_variance_ratio = []
            visualization_kind = "single_feature"
            visualization_x_label = f"{features[0]} (standardized)"
            visualization_y_label = "Anomaly score"
        if not np.isfinite(visualization_coordinates).all():
            raise InvalidDatasetError(
                "Anomaly visualization returned non-finite coordinates"
            )

        anomaly_mask = labels == -1
        anomaly_rows = int(np.sum(anomaly_mask))
        normal_rows = int(usable_rows - anomaly_rows)
        paper_reproduction_payload: dict[str, Any] | None = None
        if normalized_profile == "sharapatov_2025_figure_3a":
            if anomaly_rows != 156:
                raise InvalidDatasetError(
                    "Sharapatov et al. (2025) Figure 3a audited reproduction "
                    f"expects 156 Online anomalies, but this run produced {anomaly_rows}"
                )
            assert sharapatov_reference is not None
            sharapatov_agreement = self._sharapatov_figure3a_label_agreement(
                model_index=model_data.index,
                fresh_anomaly_mask=anomaly_mask,
                reference=sharapatov_reference,
            )
            paper_reproduction_payload = {
                "profile": normalized_profile,
                "artifact": "paper_reproduction_figure.svg",
                "reference": (
                    "Sharapatov et al. (2025), Applied Computing and "
                    "Geosciences 26, 100250, Figure 3a"
                ),
                "doi": "10.1016/j.acags.2025.100250",
                "figure_contract": "single_panel_pca_isolation_forest",
                "published_figure_label_source": (
                    "Figure3a_Data.if_anomaly_notebook_raw_features "
                    "(audited author-notebook raw-feature labels)"
                ),
                "published_figure_coordinate_source": (
                    "Figure3a_Data.PC1_full_svd_reference and "
                    "PC2_full_svd_reference"
                ),
                "fresh_online_label_source": (
                    "IsolationForest recalculated from 138 standardized "
                    "uploaded features"
                ),
                "label_agreement": sharapatov_agreement,
                "figure_uses_fresh_online_labels": False,
                "figure_uses_archived_reference_labels": True,
                "displayed_rows": usable_rows,
                "normal_rows": normal_rows,
                "anomaly_rows": anomaly_rows,
                "contamination": normalized_contamination,
                "random_state": 42,
                "pca_solver": projection_solver,
                "published_axis_variance_percent": [63.39, 36.61],
                "uploaded_data_variance_percent": [
                    float(value * 100) for value in explained_variance_ratio
                ],
                "variance_discrepancy": (
                    "The paper prints 63.39% and 36.61% on the axes, whereas "
                    "PCA of the audited uploaded 138-feature matrix yields the "
                    "uploaded_data_variance_percent values. Printed axis text is "
                    "retained only for visual fidelity."
                ),
                "model_input_distinction": (
                    "The archived notebook labels were audited from 138 raw "
                    "features; the fresh Online Isolation Forest was fit to the "
                    "same 138 features after StandardScaler transformation."
                ),
                "audited_input": {
                    "rows": 3_112,
                    "features": 138,
                    "expected_anomalies": 156,
                },
            }
        elif normalized_profile == "zhu_2024_figure_8a":
            if anomaly_rows != 25:
                raise InvalidDatasetError(
                    "Zhu et al. (2024) Figure 8a audited Online run expects 25 "
                    f"fresh LOF anomalies at contamination 0.08, but produced {anomaly_rows}"
                )
            assert zhu_reference is not None
            label_agreement = self._zhu_figure8a_label_agreement(
                dataframe=dataframe,
                model_index=model_data.index,
                fresh_anomaly_mask=anomaly_mask,
                series=zhu_reference[0],
            )
            paper_reproduction_payload = {
                "profile": normalized_profile,
                "artifact": "paper_reproduction_figure.svg",
                "reference": (
                    "Zhu et al. (2024), Water Resources Research 60, "
                    "e2023WR034748, Figure 8a"
                ),
                "doi": "10.1029/2023WR034748",
                "figure_contract": "five_ratio_series_with_reference_events",
                "ratio_columns": list(self.zhu_figure8a_ratio_columns),
                "ratio_units": "dimensionless",
                "series_rows": 302,
                "published_reference_outliers": 25,
                "earthquake_catalog_rows": 60,
                "retained_earthquake_markers": 56,
                "contamination": normalized_contamination,
                "published_label_source": (
                    "Figure8a_Series.Published_LOF_Outlier_P0_08 "
                    "(archived Data Set S3 reference)"
                ),
                "fresh_online_label_source": (
                    "LocalOutlierFactor recalculated from the five selected "
                    "standardized ratio columns"
                ),
                "label_agreement": label_agreement,
                "P": 0.08,
                "M_days": 30,
                "M_role": (
                    "earthquake-response evaluation window; not an LOF "
                    "fitting parameter"
                ),
                "figure_uses_fresh_online_labels": False,
                "figure_uses_archived_reference_labels": True,
            }
        score_summary = AnomalyScoreSummary(
            minimum=float(np.min(anomaly_scores)),
            maximum=float(np.max(anomaly_scores)),
            mean=float(np.mean(anomaly_scores)),
        )
        source_rows = [
            int(index) + 2
            if isinstance(index, (int, np.integer))
            else str(index)
            for index in feature_data.index
        ]
        observation_axis = self._anomaly_observation_axis(
            dataframe=dataframe,
            model_index=model_data.index,
            feature_columns=features,
            source_rows=source_rows,
        )
        detection_frame = feature_data.copy()
        detection_frame.insert(
            0,
            "source_row",
            source_rows,
        )
        detection_frame["visualization_x"] = visualization_coordinates[:, 0]
        detection_frame["visualization_y"] = visualization_coordinates[:, 1]
        detection_frame["visualization_observation"] = observation_axis[4]
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
        if normalized_profile == "sharapatov_2025_figure_3a":
            actual_pc1 = explained_variance_ratio[0] * 100
            actual_pc2 = explained_variance_ratio[1] * 100
            warnings.append(
                "Sharapatov et al. (2025) Figure 3a prints PC1 = 63.39% and "
                "PC2 = 36.61%, but PCA of the audited uploaded data yields "
                f"PC1 = {actual_pc1:.4f}% and PC2 = {actual_pc2:.4f}%. The "
                "paper-reproduction SVG retains the published axis text for "
                "visual fidelity and records the computed variance separately."
            )
            assert paper_reproduction_payload is not None
            agreement = paper_reproduction_payload["label_agreement"]
            warnings.append(
                "The Sharapatov paper-reproduction SVG uses archived full-SVD "
                "coordinates and audited author-notebook labels from 138 raw "
                "features. The generic Online diagnostic uses fresh PCA and "
                "fresh Isolation Forest labels from 138 standardized features. "
                f"The archived and fresh anomaly sets contain "
                f"{agreement['archived_notebook_anomalies']} and "
                f"{agreement['fresh_online_anomalies']} minerals, with "
                f"{agreement['intersection']} shared (Jaccard = "
                f"{agreement['jaccard']:.6f})."
            )
        elif normalized_profile == "zhu_2024_figure_8a":
            assert paper_reproduction_payload is not None
            agreement = paper_reproduction_payload["label_agreement"]
            warnings.append(
                "Zhu et al. (2024) Figure 8a uses 25 archived S3 reference "
                "outlier dates in the paper-reproduction SVG, not the freshly "
                "computed Online LOF labels. The fresh and archived sets contain "
                f"{agreement['fresh_online_anomalies']} and "
                f"{agreement['published_reference_anomalies']} dates, with "
                f"{agreement['intersection']} shared (Jaccard = "
                f"{agreement['jaccard']:.6f}); equal counts do not imply label "
                "equivalence."
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
        figure_path = output_dir / "anomaly_detection_figure.svg"
        report_path = output_dir / "anomaly_detection_report.json"
        detection_frame.to_csv(results_path, index=False, encoding="utf-8-sig")
        display_indices = self._select_anomaly_visualization_indices(
            anomaly_mask,
            maximum_points=10_000,
        )
        self._write_anomaly_detection_svg(
            figure_path,
            model_display_name=model_definition.display_name,
            source_filename=Path(filename or "dataset").name,
            feature_columns=features,
            visualization_kind=visualization_kind,
            visualization_coordinates=visualization_coordinates,
            visualization_x_label=visualization_x_label,
            visualization_y_label=visualization_y_label,
            anomaly_scores=anomaly_scores,
            anomaly_mask=anomaly_mask,
            decision_threshold=decision_threshold,
            source_rows=source_rows,
            display_indices=display_indices,
            observation_axis=observation_axis,
        )
        paper_figure_path: Path | None = None
        if normalized_profile == "sharapatov_2025_figure_3a":
            assert sharapatov_reference is not None
            paper_figure_path = output_dir / "paper_reproduction_figure.svg"
            archived_coordinates = sharapatov_reference.loc[
                :, ["PC1_full_svd_reference", "PC2_full_svd_reference"]
            ].to_numpy(dtype=float)
            archived_anomaly_mask = sharapatov_reference[
                "if_anomaly_notebook_raw_features"
            ].to_numpy(dtype=int).astype(bool)
            self._write_sharapatov_figure3a_svg(
                paper_figure_path,
                coordinates=archived_coordinates,
                anomaly_mask=archived_anomaly_mask,
                source_row_ids=sharapatov_reference["source_row_id"].tolist(),
            )
        elif normalized_profile == "zhu_2024_figure_8a":
            assert zhu_reference is not None
            paper_figure_path = output_dir / "paper_reproduction_figure.svg"
            self._write_zhu_figure8a_svg(
                paper_figure_path,
                series=zhu_reference[0],
                earthquakes=zhu_reference[1],
            )
        visualization_payload = {
            "kind": "two_panel_anomaly_diagnostics",
            "artifact": figure_path.name,
            "format": "svg",
            "displayed_rows": int(display_indices.size),
            "sampled_out_rows": int(usable_rows - display_indices.size),
            "total_rows": usable_rows,
            "sampling": (
                "all_rows"
                if display_indices.size == usable_rows
                else "deterministic_evenly_spaced_within_label"
            ),
            "scientific_scope": (
                "The projection is visualization only; anomaly labels and scores "
                "were computed from all selected standardized features."
            ),
            "panel_a": {
                "kind": visualization_kind,
                "x_column": "visualization_x",
                "y_column": "visualization_y",
                "x_axis": visualization_x_label,
                "y_axis": visualization_y_label,
                "explained_variance_ratio": explained_variance_ratio,
                "pca_solver": projection_solver,
                "projection_random_state": (
                    42 if projection_solver == "randomized" else None
                ),
            },
            "panel_b": {
                "kind": "anomaly_score_profile",
                "x_axis": observation_axis[0],
                "x_axis_kind": observation_axis[3],
                "x_column": "visualization_observation",
                "y_axis": "Anomaly score (higher = more anomalous)",
                "decision_threshold": decision_threshold,
            },
            "markers": {
                "normal": "filled_circle",
                "anomaly": "diamond_with_outline",
            },
        }
        report_payload = {
            "report_version": "v080-anomaly-detection-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "model": model_name,
            "model_display_name": model_definition.display_name,
            "feature_columns": features,
            "contamination": normalized_contamination,
            "reproduction_profile": normalized_profile,
            "random_state": 42 if model_name == "isolation_forest" else None,
            "summary": summary.model_dump(),
            "score_summary": score_summary.model_dump(),
            "visualization": visualization_payload,
            "paper_reproduction": paper_reproduction_payload,
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
            contamination=normalized_contamination,
            reproduction_profile=normalized_profile,
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
                    name=figure_path.name,
                    download_url=(
                        f"/api/data-mining/jobs/{job_id}/files/{figure_path.name}"
                    ),
                    size_bytes=figure_path.stat().st_size,
                ),
                *(
                    [
                        ArtifactResponse(
                            name=paper_figure_path.name,
                            download_url=(
                                f"/api/data-mining/jobs/{job_id}/files/"
                                f"{paper_figure_path.name}"
                            ),
                            size_bytes=paper_figure_path.stat().st_size,
                        )
                    ]
                    if paper_figure_path is not None
                    else []
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
            [item.model_dump(exclude={"sample_count"}) for item in bins]
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

    def run_element_time_series(
        self,
        *,
        filename: str | None,
        content: bytes,
        age_column: str,
        value_column: str,
        age_unit: str = "Ma",
        bin_width: float = 100.0,
        value_unit: str = "wt%",
        filter_column: str | None = None,
        filter_min: float | None = None,
        filter_max: float | None = None,
    ) -> TimeSeriesResponse:
        suffix = self._validate_upload(filename, content)
        dataframe = self._read_dataframe(suffix, content)
        self._validate_dataframe(dataframe)
        dataframe.columns = [str(column) for column in dataframe.columns]

        selected = [age_column, value_column]
        if filter_column:
            selected.append(filter_column)
        mapped_columns = self._validate_selected_columns(dataframe, selected)
        if len(mapped_columns) != len(selected):
            raise InvalidDatasetError(
                "Element time series requires different age, value, and filter columns"
            )
        if age_unit not in {"Ma", "Ga"}:
            raise InvalidDatasetError("Age unit must be Ma or Ga")
        if not math.isfinite(bin_width) or bin_width <= 0:
            raise InvalidDatasetError("Bin width must be a positive finite number")
        if filter_column:
            if filter_min is None or filter_max is None:
                raise InvalidDatasetError(
                    "Both filter minimum and maximum are required"
                )
            if not math.isfinite(filter_min) or not math.isfinite(filter_max):
                raise InvalidDatasetError("Filter bounds must be finite numbers")
            if filter_min > filter_max:
                raise InvalidDatasetError(
                    "Filter minimum must not exceed filter maximum"
                )

        numeric = dataframe.loc[:, selected].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        valid_mask = numeric[[age_column, value_column]].notna().all(axis=1)
        if filter_column:
            valid_mask &= numeric[filter_column].between(
                float(filter_min), float(filter_max), inclusive="both"
            )
        analysis = numeric.loc[valid_mask, [age_column, value_column]].copy()
        if (analysis[age_column] < 0).any():
            raise InvalidDatasetError("Age values must be zero or greater")
        if analysis.empty:
            raise InvalidDatasetError(
                "Element time series requires at least one usable numeric row"
            )

        maximum_age = float(analysis[age_column].max())
        if maximum_age <= 0:
            raise InvalidDatasetError(
                "Element time series requires at least one age greater than zero"
            )
        bin_count = max(1, int(math.floor(maximum_age / bin_width)) + 1)
        if bin_count > 5_000:
            raise InvalidDatasetError(
                "The selected bin width would create more than 5,000 age bins"
            )
        analysis["__bin_index__"] = np.floor(
            analysis[age_column] / bin_width
        ).astype(int)
        grouped = analysis.groupby("__bin_index__")[value_column]
        statistics = grouped.agg(["count", "mean", "std"])

        bins: list[TimeSeriesBinItem] = []
        for index in range(bin_count):
            age = (index + 0.5) * bin_width
            if index not in statistics.index:
                bins.append(TimeSeriesBinItem(age=age))
                continue
            count = int(statistics.loc[index, "count"])
            mean = float(statistics.loc[index, "mean"])
            std = float(statistics.loc[index, "std"])
            uncertainty = (
                2.0 * std / math.sqrt(count)
                if count >= 2 and math.isfinite(std)
                else None
            )
            bins.append(
                TimeSeriesBinItem(
                    age=age,
                    mean_proportion=mean,
                    uncertainty_2sigma=uncertainty,
                    sample_count=count,
                )
            )

        usable_rows = int(analysis.shape[0])
        dropped_rows = int(dataframe.shape[0] - usable_rows)
        populated_bins = sum(item.mean_proportion is not None for item in bins)
        summary = TimeSeriesSummary(
            original_rows=int(dataframe.shape[0]),
            usable_rows=usable_rows,
            dropped_rows=dropped_rows,
            bin_count=len(bins),
            populated_bins=populated_bins,
        )
        filter_description = (
            f" after filtering {filter_column} to {filter_min:g}-{filter_max:g}"
            if filter_column and filter_min is not None and filter_max is not None
            else ""
        )
        warnings = [
            (
                f"Used {usable_rows:,} rows{filter_description}; excluded "
                f"{dropped_rows:,} rows with missing values or outside the filter."
            ),
            (
                "Each bin is an unweighted arithmetic mean; uncertainty is +/-2 SEM. "
                "This basic validation does not reproduce Keller et al.'s spatial-"
                "temporal weighting or Monte Carlo age resampling."
            ),
        ]

        job_id = uuid4().hex
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True)
        results_path = output_dir / "element_mean_time_series.csv"
        figure_path = output_dir / "element_mean_time_series.svg"
        report_path = output_dir / "element_mean_time_series_report.json"
        pd.DataFrame(
            {
                "age": [item.age for item in bins],
                "mean_value": [item.mean_proportion for item in bins],
                "uncertainty_2sem": [item.uncertainty_2sigma for item in bins],
                "sample_count": [item.sample_count for item in bins],
            }
        ).to_csv(results_path, index=False, encoding="utf-8-sig")
        self._write_element_time_series_svg(
            figure_path,
            bins=bins,
            age_unit=age_unit,
            value_column=value_column,
            value_unit=value_unit,
            bin_width=bin_width,
        )
        report_payload = {
            "report_version": "element-mean-time-series-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(filename or "dataset").name,
            "analysis_type": "element_mean",
            "column_mapping": {"age": age_column, "value": value_column},
            "value_unit": value_unit,
            "age_unit": age_unit,
            "bin_width": bin_width,
            "uncertainty_method": "2_sem",
            "filter": (
                {
                    "column": filter_column,
                    "minimum": filter_min,
                    "maximum": filter_max,
                }
                if filter_column
                else None
            ),
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
            message="Element mean time series completed",
            source_filename=Path(filename or "dataset").name,
            age_column=age_column,
            age_unit=age_unit,
            bin_width=bin_width,
            analysis_type="element_mean",
            value_column=value_column,
            value_unit=value_unit,
            uncertainty_method="2_sem",
            filter_column=filter_column,
            filter_min=filter_min,
            filter_max=filter_max,
            summary=summary,
            bins=bins,
            warnings=warnings,
            artifacts=artifacts,
        )

    @staticmethod
    def _validate_anomaly_contamination(value: str | float) -> str | float:
        if isinstance(value, bool):
            raise InvalidDatasetError(
                "Anomaly contamination must be 'auto' or a number from "
                "0.001 through 0.5"
            )
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "auto":
                return "auto"
            try:
                numeric_value = float(normalized)
            except ValueError as exc:
                raise InvalidDatasetError(
                    "Anomaly contamination must be 'auto' or a number from "
                    "0.001 through 0.5"
                ) from exc
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise InvalidDatasetError(
                    "Anomaly contamination must be 'auto' or a number from "
                    "0.001 through 0.5"
                ) from exc
        if (
            not math.isfinite(numeric_value)
            or numeric_value < 0.001
            or numeric_value > 0.5
        ):
            raise InvalidDatasetError(
                "Anomaly contamination must be from 0.001 through 0.5"
            )
        return numeric_value

    @classmethod
    def _read_sharapatov_figure3a_reference(
        cls,
        content: bytes,
    ) -> pd.DataFrame:
        try:
            reference = pd.read_excel(
                BytesIO(content),
                sheet_name="Figure3a_Data",
            )
        except Exception as exc:
            raise InvalidDatasetError(
                "Sharapatov et al. (2025) Figure 3a requires a readable "
                "Figure3a_Data worksheet"
            ) from exc
        reference.columns = [str(column) for column in reference.columns]
        if tuple(reference.columns) != cls.sharapatov_figure3a_columns:
            raise InvalidDatasetError(
                "Figure3a_Data must contain the exact audited 15-column schema"
            )
        if reference.shape[0] != 3_112:
            raise InvalidDatasetError(
                "Figure3a_Data must contain exactly 3,112 mineral records"
            )
        normalized = reference.copy()
        source_ids = pd.to_numeric(
            normalized["source_row_id"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(source_ids).all()
            or not np.array_equal(source_ids, np.arange(3_112, dtype=float))
        ):
            raise InvalidDatasetError(
                "Figure3a_Data.source_row_id must be the complete zero-based "
                "sequence 0 through 3,111"
            )
        if normalized["Name"].isna().any() or normalized["Name"].duplicated().any():
            raise InvalidDatasetError(
                "Figure3a_Data.Name must contain 3,112 unique mineral names"
            )
        coordinate_columns = [
            "PC1_full_svd_reference",
            "PC2_full_svd_reference",
        ]
        coordinates = normalized.loc[:, coordinate_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
            raise InvalidDatasetError(
                "Figure3a_Data archived full-SVD coordinates must be finite"
            )
        archived_flags = pd.to_numeric(
            normalized["if_anomaly_notebook_raw_features"],
            errors="coerce",
        )
        if (
            archived_flags.isna().any()
            or not archived_flags.isin([0, 1]).all()
            or int(archived_flags.sum()) != 156
        ):
            raise InvalidDatasetError(
                "Figure3a_Data must contain exactly 156 audited notebook "
                "anomaly flags encoded as 0 or 1"
            )
        normalized["source_row_id"] = source_ids.astype(int)
        normalized.loc[:, coordinate_columns] = coordinates
        normalized["if_anomaly_notebook_raw_features"] = archived_flags.astype(int)
        return normalized

    @staticmethod
    def _validate_sharapatov_figure3a_input_alignment(
        *,
        dataframe: pd.DataFrame,
        model_index: pd.Index,
        reference: pd.DataFrame,
    ) -> None:
        if "Name" not in dataframe.columns:
            raise InvalidDatasetError(
                "Sharapatov et al. (2025) Figure 3a requires the Name column"
            )
        try:
            online_indices = np.asarray(model_index, dtype=int)
        except (TypeError, ValueError) as exc:
            raise InvalidDatasetError(
                "Sharapatov Figure3a_Data cannot be aligned to Online_Input rows"
            ) from exc
        reference_ids = reference["source_row_id"].to_numpy(dtype=int)
        if not np.array_equal(online_indices, reference_ids):
            raise InvalidDatasetError(
                "Figure3a_Data source_row_id values do not align with the "
                "Online_Input row order"
            )
        online_names = dataframe.loc[model_index, "Name"].astype(str).to_numpy()
        archived_names = reference["Name"].astype(str).to_numpy()
        if not np.array_equal(online_names, archived_names):
            raise InvalidDatasetError(
                "Figure3a_Data mineral names do not align with Online_Input"
            )

    @staticmethod
    def _sharapatov_figure3a_label_agreement(
        *,
        model_index: pd.Index,
        fresh_anomaly_mask: np.ndarray,
        reference: pd.DataFrame,
    ) -> dict[str, Any]:
        online_indices = np.asarray(model_index, dtype=int)
        archived = reference.set_index("source_row_id").loc[
            online_indices, "if_anomaly_notebook_raw_features"
        ].to_numpy(dtype=int).astype(bool)
        fresh = np.asarray(fresh_anomaly_mask, dtype=bool)
        intersection = int(np.sum(fresh & archived))
        union = int(np.sum(fresh | archived))
        symmetric_difference = int(np.sum(fresh ^ archived))
        return {
            "alignment_key": "source_row_id_and_Name",
            "fresh_online_anomalies": int(np.sum(fresh)),
            "archived_notebook_anomalies": int(np.sum(archived)),
            "intersection": intersection,
            "symmetric_difference": symmetric_difference,
            "union": union,
            "jaccard": float(intersection / union) if union else 1.0,
            "rowwise_agreement": float(np.mean(fresh == archived)),
            "equivalent": bool(np.array_equal(fresh, archived)),
        }

    @classmethod
    def _read_zhu_figure8a_reference(
        cls,
        content: bytes,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            sheets = pd.read_excel(
                BytesIO(content),
                sheet_name=["Figure8a_Series", "Earthquakes"],
            )
        except Exception as exc:
            raise InvalidDatasetError(
                "Zhu et al. (2024) Figure 8a requires readable "
                "Figure8a_Series and Earthquakes worksheets"
            ) from exc
        series = sheets["Figure8a_Series"].copy()
        earthquakes = sheets["Earthquakes"].copy()
        series.columns = [str(column) for column in series.columns]
        earthquakes.columns = [str(column) for column in earthquakes.columns]
        if tuple(series.columns) != cls.zhu_figure8a_series_columns:
            raise InvalidDatasetError(
                "Figure8a_Series must contain the exact audited seven-column schema"
            )
        if tuple(earthquakes.columns) != cls.zhu_figure8a_earthquake_columns:
            raise InvalidDatasetError(
                "Earthquakes must contain the exact audited ten-column schema"
            )
        if series.shape[0] != 302:
            raise InvalidDatasetError(
                "Figure8a_Series must contain exactly 302 GA observations"
            )
        parsed_dates = pd.to_datetime(series["Date"], errors="coerce", utc=True)
        if parsed_dates.isna().any() or parsed_dates.duplicated().any():
            raise InvalidDatasetError(
                "Figure8a_Series.Date must contain 302 unique valid dates"
            )
        ratios = series.loc[:, cls.zhu_figure8a_ratio_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(ratios.to_numpy(dtype=float)).all():
            raise InvalidDatasetError(
                "Figure8a_Series ratio values must all be finite numbers"
            )
        published_flags = pd.to_numeric(
            series["Published_LOF_Outlier_P0_08"], errors="coerce"
        )
        if (
            published_flags.isna().any()
            or not published_flags.isin([0, 1]).all()
            or int(published_flags.sum()) != 25
        ):
            raise InvalidDatasetError(
                "Figure8a_Series must contain exactly 25 archived P = 0.08 "
                "outlier flags encoded as 0 or 1"
            )
        if earthquakes.shape[0] != 60:
            raise InvalidDatasetError(
                "Earthquakes must contain exactly 60 catalog events"
            )
        parsed_events = pd.to_datetime(
            earthquakes["Event_DateTime"], errors="coerce", utc=True
        )
        if parsed_events.isna().any():
            raise InvalidDatasetError(
                "Earthquakes.Event_DateTime must contain 60 valid timestamps"
            )
        normalized_use = earthquakes["Use_in_Figure8a"].map(
            lambda value: str(value).strip().lower()
        )
        allowed_flags = {"true", "false", "1", "0"}
        if not normalized_use.isin(allowed_flags).all():
            raise InvalidDatasetError(
                "Earthquakes.Use_in_Figure8a must contain only true/false flags"
            )
        retained = normalized_use.isin({"true", "1"})
        if int(retained.sum()) != 56:
            raise InvalidDatasetError(
                "Earthquakes must retain exactly 56 Figure 8a marker events"
            )
        series.loc[:, cls.zhu_figure8a_ratio_columns] = ratios
        series["Published_LOF_Outlier_P0_08"] = published_flags.astype(int)
        series["_parsed_date"] = parsed_dates
        earthquakes["_parsed_event_datetime"] = parsed_events
        earthquakes["_use_in_figure8a"] = retained.to_numpy(dtype=bool)
        return series, earthquakes

    @classmethod
    def _validate_zhu_figure8a_input_alignment(
        cls,
        *,
        dataframe: pd.DataFrame,
        model_index: pd.Index,
        series: pd.DataFrame,
    ) -> None:
        online = dataframe.loc[
            model_index, ["Date", *cls.zhu_figure8a_ratio_columns]
        ].copy()
        online["_parsed_date"] = pd.to_datetime(
            online["Date"], errors="coerce", utc=True
        )
        if online["_parsed_date"].isna().any() or online[
            "_parsed_date"
        ].duplicated().any():
            raise InvalidDatasetError(
                "Online_Input.Date must contain 302 unique valid dates"
            )
        for column in cls.zhu_figure8a_ratio_columns:
            online[column] = pd.to_numeric(online[column], errors="coerce")
        aligned = online.merge(
            series.loc[:, ["_parsed_date", *cls.zhu_figure8a_ratio_columns]],
            on="_parsed_date",
            how="outer",
            suffixes=("_online", "_reference"),
            validate="one_to_one",
            indicator=True,
        )
        if not aligned["_merge"].eq("both").all() or aligned.shape[0] != 302:
            raise InvalidDatasetError(
                "Figure8a_Series dates do not align exactly with Online_Input"
            )
        for column in cls.zhu_figure8a_ratio_columns:
            if not np.allclose(
                aligned[f"{column}_online"].to_numpy(dtype=float),
                aligned[f"{column}_reference"].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise InvalidDatasetError(
                    "Figure8a_Series ratio values do not align exactly with "
                    f"Online_Input for {column}"
                )

    @staticmethod
    def _zhu_figure8a_label_agreement(
        *,
        dataframe: pd.DataFrame,
        model_index: pd.Index,
        fresh_anomaly_mask: np.ndarray,
        series: pd.DataFrame,
    ) -> dict[str, Any]:
        fresh = pd.DataFrame(
            {
                "_parsed_date": pd.to_datetime(
                    dataframe.loc[model_index, "Date"],
                    errors="coerce",
                    utc=True,
                ),
                "fresh": np.asarray(fresh_anomaly_mask, dtype=bool),
            }
        )
        archived = series.loc[
            :, ["_parsed_date", "Published_LOF_Outlier_P0_08"]
        ].rename(columns={"Published_LOF_Outlier_P0_08": "archived"})
        aligned = fresh.merge(
            archived,
            on="_parsed_date",
            how="inner",
            validate="one_to_one",
        ).sort_values("_parsed_date", kind="stable")
        if aligned.shape[0] != 302:
            raise InvalidDatasetError(
                "Fresh Online and archived Zhu labels could not be aligned by Date"
            )
        fresh_mask = aligned["fresh"].to_numpy(dtype=bool)
        archived_mask = aligned["archived"].to_numpy(dtype=int).astype(bool)
        intersection = int(np.sum(fresh_mask & archived_mask))
        union = int(np.sum(fresh_mask | archived_mask))
        return {
            "alignment_key": "Date",
            "aligned_rows": 302,
            "fresh_online_anomalies": int(np.sum(fresh_mask)),
            "published_reference_anomalies": int(np.sum(archived_mask)),
            "intersection": intersection,
            "symmetric_difference": int(np.sum(fresh_mask ^ archived_mask)),
            "union": union,
            "jaccard": float(intersection / union) if union else 1.0,
            "rowwise_agreement": float(np.mean(fresh_mask == archived_mask)),
            "equivalent": bool(np.array_equal(fresh_mask, archived_mask)),
        }

    @staticmethod
    def _write_sharapatov_figure3a_svg(
        path: Path,
        *,
        coordinates: np.ndarray,
        anomaly_mask: np.ndarray,
        source_row_ids: list[int | str],
    ) -> None:
        """Render the audited evidence layer corresponding to paper Figure 3a."""

        coordinates = np.asarray(coordinates, dtype=float)
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or coordinates.shape[0] != anomaly_mask.size
            or coordinates.shape[0] != len(source_row_ids)
            or not np.isfinite(coordinates).all()
        ):
            raise InvalidDatasetError(
                "Sharapatov Figure 3a archived coordinates and labels are invalid"
            )

        width = 960
        height = 700
        left = 104
        right = 48
        top = 126
        bottom = 92
        plot_width = width - left - right
        plot_height = height - top - bottom

        def padded_bounds(values: np.ndarray) -> tuple[float, float]:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            span = maximum - minimum
            if span == 0:
                span = max(abs(maximum), 1.0)
            padding = span * 0.055
            return minimum - padding, maximum + padding

        x_minimum, x_maximum = padded_bounds(coordinates[:, 0])
        y_minimum, y_maximum = padded_bounds(coordinates[:, 1])

        def x_position(value: float) -> float:
            return left + (value - x_minimum) / (x_maximum - x_minimum) * plot_width

        def y_position(value: float) -> float:
            return top + (y_maximum - value) / (y_maximum - y_minimum) * plot_height

        def tick_text(value: float) -> str:
            if abs(value) >= 100:
                return f"{value:.0f}"
            if abs(value) >= 10:
                return f"{value:.1f}"
            return f"{value:.2f}"

        grid: list[str] = []
        for tick_index in range(6):
            x_value = x_minimum + (x_maximum - x_minimum) * tick_index / 5
            x = x_position(x_value)
            grid.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top + plot_height}" class="grid"/>'
            )
            grid.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 25}" '
                f'text-anchor="middle" class="tick">{escape(tick_text(x_value))}</text>'
            )
            y_value = y_minimum + (y_maximum - y_minimum) * tick_index / 5
            y = y_position(y_value)
            grid.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>'
            )
            grid.append(
                f'<text x="{left - 13}" y="{y + 4:.2f}" text-anchor="end" '
                f'class="tick">{escape(tick_text(y_value))}</text>'
            )

        markers: list[str] = []
        for is_anomaly in (False, True):
            for row_index in np.flatnonzero(anomaly_mask == is_anomaly):
                x = x_position(float(coordinates[row_index, 0]))
                y = y_position(float(coordinates[row_index, 1]))
                source_id = escape(str(source_row_ids[int(row_index)]), quote=True)
                label = "anomaly" if is_anomaly else "normal"
                tooltip = escape(
                    f"source_row_id {source_id}; archived notebook label {label}; "
                    f"PC1 {coordinates[row_index, 0]:.6g}; "
                    f"PC2 {coordinates[row_index, 1]:.6g}"
                )
                css_class = (
                    "paper-anomaly-point" if is_anomaly else "paper-normal-point"
                )
                radius = 4.2 if is_anomaly else 2.5
                markers.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.1f}" '
                    f'class="{css_class}" data-source-row-id="{source_id}">'
                    f'<title>{tooltip}</title></circle>'
                )

        normal_count = int(np.sum(~anomaly_mask))
        anomaly_count = int(np.sum(anomaly_mask))
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="sharapatov-title sharapatov-description">
<title id="sharapatov-title">Isolation Forest Anomaly Detection</title>
<desc id="sharapatov-description">Audited reconstruction of Sharapatov et al. (2025) Figure 3a using all {coordinates.shape[0]:,} archived full-SVD PCA coordinates and the archived raw-feature Isolation Forest notebook labels. This reference evidence layer is distinct from the freshly computed Online diagnostic.</desc>
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#111827}}
.title{{font-size:24px;font-weight:700}} .subtitle{{font-size:12px;fill:#53636d}}
.panel-label{{font-size:20px;font-weight:700}} .axis{{stroke:#202b33;stroke-width:1.5}}
.grid{{stroke:#dfe5e8;stroke-width:1}} .tick{{font-size:11px;fill:#4d5c65}}
.axis-label{{font-size:15px;font-weight:600}} .legend{{font-size:12px;fill:#263640}}
.paper-normal-point,.paper-normal-legend{{fill:#2474b5;fill-opacity:.72;stroke:#ffffff;stroke-width:.45}}
.paper-anomaly-point,.paper-anomaly-legend{{fill:#d8342a;fill-opacity:.90;stroke:#551610;stroke-width:.75}}
.frame{{fill:#ffffff;stroke:#b8c4ca;stroke-width:1}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="38" class="title">Isolation Forest Anomaly Detection</text>
<text x="{left}" y="61" class="subtitle">Sharapatov et al. (2025), Figure 3a — archived full-SVD coordinates and raw-feature notebook labels</text>
<g class="legend" transform="translate({left} 86)">
  <circle cx="0" cy="0" r="4" class="paper-normal-legend"/><text x="10" y="4">Normal Minerals ({normal_count:,})</text>
  <circle cx="190" cy="0" r="5" class="paper-anomaly-legend"/><text x="202" y="4">Anomalies (Isolation Forest; {anomaly_count:,})</text>
</g>
<text x="{left - 54}" y="{top - 15}" class="panel-label">(a)</text>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="frame"/>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
{''.join(markers)}
<text x="{left + plot_width / 2:.2f}" y="{height - 31}" text-anchor="middle" class="axis-label">Principal Component 1 (63.39% variance)</text>
<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" class="axis-label" transform="rotate(-90 28 {top + plot_height / 2:.2f})">Principal Component 2 (36.61% variance)</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

    @classmethod
    def _write_zhu_figure8a_svg(
        cls,
        path: Path,
        *,
        series: pd.DataFrame,
        earthquakes: pd.DataFrame,
    ) -> None:
        """Render the archived time-series evidence layer corresponding to Figure 8a."""

        ordered_series = series.sort_values("_parsed_date", kind="stable").copy()
        retained_events = earthquakes.loc[
            earthquakes["_use_in_figure8a"]
        ].sort_values("_parsed_event_datetime", kind="stable")
        if (
            ordered_series.shape[0] != 302
            or int(ordered_series["Published_LOF_Outlier_P0_08"].sum()) != 25
            or retained_events.shape[0] != 56
        ):
            raise InvalidDatasetError(
                "Zhu Figure 8a archived series or event counts are invalid"
            )

        width = 1180
        height = 650
        left = 90
        right = 42
        top = 118
        bottom = 82
        plot_width = width - left - right
        plot_height = height - top - bottom
        x_minimum = pd.Timestamp("2020-01-01", tz="UTC").timestamp()
        x_maximum = pd.Timestamp("2022-08-15", tz="UTC").timestamp()
        y_minimum = -1.0
        y_maximum = 11.1

        def x_position(value: pd.Timestamp) -> float:
            epoch = value.timestamp()
            return left + (epoch - x_minimum) / (x_maximum - x_minimum) * plot_width

        def y_position(value: float) -> float:
            return top + (y_maximum - value) / (y_maximum - y_minimum) * plot_height

        dates = ordered_series["_parsed_date"]
        if (
            dates.min().timestamp() < x_minimum
            or dates.max().timestamp() > x_maximum
            or retained_events["_parsed_event_datetime"].min().timestamp()
            < x_minimum
            or retained_events["_parsed_event_datetime"].max().timestamp()
            > x_maximum
        ):
            raise InvalidDatasetError(
                "Zhu Figure 8a archived dates fall outside the audited plot domain"
            )

        ratio_styles = (
            ("Na_Cl_ratio", "Na⁺/Cl⁻", "#C51B1D"),
            ("Na_F_ratio", "Na⁺/F⁻", "#32CD32"),
            ("Na_SO4_ratio", "Na⁺/SO₄²⁻", "#6A1A14"),
            ("F_Cl_ratio", "F⁻/Cl⁻", "#192A92"),
            ("SO4_Cl_ratio", "SO₄²⁻/Cl⁻", "#25D8D5"),
        )
        ratio_lines: list[str] = []
        for column, label, colour in ratio_styles:
            points = " ".join(
                f"{x_position(moment):.2f},{y_position(float(value)):.2f}"
                for moment, value in zip(
                    ordered_series["_parsed_date"],
                    ordered_series[column],
                    strict=True,
                )
            )
            ratio_lines.append(
                f'<polyline points="{points}" class="ratio-line" '
                f'stroke="{colour}" data-series="{escape(column, quote=True)}">'
                f'<title>{escape(label)} raw ratio series (302 observations)</title>'
                f'</polyline>'
            )

        archived_outliers = ordered_series.loc[
            ordered_series["Published_LOF_Outlier_P0_08"].astype(bool)
        ]
        outlier_markers = "".join(
            f'<circle cx="{x_position(moment):.2f}" '
            f'cy="{y_position(10.38):.2f}" r="4.8" '
            f'class="published-outlier-marker"><title>Archived Data Set S3 '
            f'LOF outlier: {escape(moment.strftime("%Y-%m-%d"))}</title>'
            f'</circle>'
            for moment in archived_outliers["_parsed_date"]
        )
        earthquake_markers: list[str] = []
        event_y = y_position(10.86)
        for moment, raw_event_id in zip(
            retained_events["_parsed_event_datetime"],
            retained_events["Event_ID"],
            strict=True,
        ):
            x = x_position(moment)
            event_id = escape(str(raw_event_id))
            event_date = escape(moment.strftime("%Y-%m-%d"))
            earthquake_markers.append(
                f'<path d="M {x - 5:.2f} {event_y - 4:.2f} '
                f'L {x + 5:.2f} {event_y - 4:.2f} L {x:.2f} {event_y + 5:.2f} Z" '
                f'class="earthquake-marker"><title>Retained earthquake '
                f'{event_id}: {event_date}</title></path>'
            )

        grid: list[str] = []
        for y_value in range(0, 11, 2):
            y = y_position(float(y_value))
            grid.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>'
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'class="tick">{y_value}</text>'
            )
        x_ticks = pd.to_datetime(
            [
                "2020-01-01",
                "2020-07-01",
                "2021-01-01",
                "2021-07-01",
                "2022-01-01",
                "2022-07-01",
            ],
            utc=True,
        )
        for moment in x_ticks:
            x = x_position(moment)
            grid.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top + plot_height}" class="grid"/>'
                f'<text x="{x:.2f}" y="{top + plot_height + 25}" '
                f'text-anchor="middle" class="tick">'
                f'{moment.year}/{moment.month}/{moment.day}</text>'
            )

        legend_items: list[str] = []
        legend_x = left
        for _, label, colour in ratio_styles:
            legend_items.append(
                f'<line x1="{legend_x}" y1="82" x2="{legend_x + 23}" y2="82" '
                f'stroke="{colour}" stroke-width="2"/>'
                f'<text x="{legend_x + 29}" y="86" class="legend">'
                f'{escape(label)}</text>'
            )
            legend_x += 115
        legend_items.append(
            f'<circle cx="{legend_x + 4}" cy="82" r="4.8" '
            f'class="published-outlier-legend"/><text x="{legend_x + 14}" '
            f'y="86" class="legend">Outliers (25)</text>'
        )
        legend_x += 128
        legend_items.append(
            f'<path d="M {legend_x} 77 L {legend_x + 10} 77 '
            f'L {legend_x + 5} 87 Z" class="earthquake-legend"/>'
            f'<text x="{legend_x + 16}" y="86" class="legend">Earthquake (56)</text>'
        )

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="zhu-title zhu-description">
<title id="zhu-title">Zhu et al. (2024), Figure 8a reference reconstruction</title>
<desc id="zhu-description">Five unsmoothed raw ion-ratio series from 302 observations, 25 archived open-circle LOF outlier dates from Data Set S3, and 56 retained earthquake markers. The archived outlier markers are a published reference evidence layer and are not the freshly computed Online LOF labels.</desc>
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#151d22}}
.title{{font-size:21px;font-weight:700}} .subtitle{{font-size:12px;fill:#596a73}}
.panel-label{{font-size:20px;font-weight:700}} .axis{{stroke:#222c32;stroke-width:1.4}}
.grid{{stroke:#e1e6e8;stroke-width:1}} .tick{{font-size:11px;fill:#4c5c64}}
.axis-label{{font-size:14px;font-weight:600}} .legend{{font-size:11px}}
.ratio-line{{fill:none;stroke-width:1.55;stroke-linejoin:round;stroke-linecap:round}}
.published-outlier-marker,.published-outlier-legend{{fill:#ffffff;stroke:#00aeb3;stroke-width:2}}
.earthquake-marker,.earthquake-legend{{fill:#e31a1c;stroke:#7d080a;stroke-width:.55}}
.frame{{fill:#ffffff;stroke:#b8c3c8;stroke-width:1}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="34" class="title">Zhu et al. (2024), Figure 8a reference reconstruction</text>
<text x="{left}" y="55" class="subtitle">Archived published outlier dates and earthquake catalogue are shown separately from the freshly computed Online LOF diagnostic.</text>
{''.join(legend_items)}
<text x="{left - 52}" y="{top - 13}" class="panel-label">(a)</text>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="frame"/>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
{''.join(ratio_lines)}
{outlier_markers}
{''.join(earthquake_markers)}
<text x="{left + plot_width / 2:.2f}" y="{height - 28}" text-anchor="middle" class="axis-label">Date</text>
<text x="27" y="{top + plot_height / 2:.2f}" text-anchor="middle" class="axis-label" transform="rotate(-90 27 {top + plot_height / 2:.2f})">Ion ratio (dimensionless)</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

    @staticmethod
    def _select_anomaly_visualization_indices(
        anomaly_mask: np.ndarray,
        *,
        maximum_points: int,
    ) -> np.ndarray:
        """Return a deterministic, anomaly-preserving subset for SVG rendering."""

        total_rows = int(anomaly_mask.size)
        all_indices = np.arange(total_rows, dtype=int)
        if total_rows <= maximum_points:
            return all_indices

        anomaly_indices = all_indices[np.asarray(anomaly_mask, dtype=bool)]
        normal_indices = all_indices[~np.asarray(anomaly_mask, dtype=bool)]

        def evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
            if count <= 0 or indices.size == 0:
                return np.asarray([], dtype=int)
            if indices.size <= count:
                return indices
            positions = np.linspace(0, indices.size - 1, num=count, dtype=int)
            return indices[positions]

        half_quota = maximum_points // 2
        anomaly_quota = min(anomaly_indices.size, half_quota)
        normal_quota = min(normal_indices.size, half_quota)
        remaining = maximum_points - anomaly_quota - normal_quota
        anomaly_extra = min(anomaly_indices.size - anomaly_quota, remaining)
        anomaly_quota += anomaly_extra
        remaining -= anomaly_extra
        normal_quota += min(normal_indices.size - normal_quota, remaining)
        selected_anomalies = evenly_spaced(anomaly_indices, anomaly_quota)
        selected_normals = evenly_spaced(normal_indices, normal_quota)
        return np.sort(np.concatenate((selected_anomalies, selected_normals)))

    @staticmethod
    def _anomaly_observation_axis(
        *,
        dataframe: pd.DataFrame,
        model_index: pd.Index,
        feature_columns: list[str],
        source_rows: list[int | str],
    ) -> tuple[str, np.ndarray, bool, str, list[Any]]:
        """Use an explicit date/time column when it is complete, else source rows."""

        candidates = []
        for column in dataframe.columns:
            if column in feature_columns:
                continue
            series = dataframe[column]
            normalized_name = str(column).strip().lower()
            is_named_time = bool(
                re.search(
                    r"(?:^|[^a-z0-9])(?:date|time|year|age)(?:$|[^a-z0-9])",
                    normalized_name,
                )
            ) or any(
                token in normalized_name for token in ("日期", "时间", "年代")
            )
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            if is_named_time or is_datetime:
                candidates.append((not is_datetime, not is_named_time, str(column)))
        candidates.sort()

        for _, _, column in candidates:
            values = dataframe.loc[model_index, column]
            if pd.api.types.is_bool_dtype(values):
                continue
            if pd.api.types.is_numeric_dtype(values):
                numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(
                    dtype=float
                )
                if (
                    np.isfinite(numeric_values).all()
                    and np.unique(numeric_values).size >= 2
                ):
                    return (
                        column,
                        numeric_values,
                        False,
                        "numeric_time",
                        numeric_values.tolist(),
                    )
                continue
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            if parsed.notna().all() and parsed.nunique() >= 2:
                seconds = parsed.map(
                    lambda value: value.timestamp()
                ).to_numpy(dtype=float)
                exported = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
                return column, seconds, True, "datetime", exported

        try:
            numeric_source_rows = np.asarray(source_rows, dtype=float)
        except (TypeError, ValueError):
            numeric_source_rows = np.arange(1, len(source_rows) + 1, dtype=float)
        return (
            "Source row",
            numeric_source_rows,
            False,
            "source_row",
            list(source_rows),
        )

    @staticmethod
    def _write_anomaly_detection_svg(
        path: Path,
        *,
        model_display_name: str,
        source_filename: str,
        feature_columns: list[str],
        visualization_kind: str,
        visualization_coordinates: np.ndarray,
        visualization_x_label: str,
        visualization_y_label: str,
        anomaly_scores: np.ndarray,
        anomaly_mask: np.ndarray,
        decision_threshold: float,
        source_rows: list[int | str],
        display_indices: np.ndarray,
        observation_axis: tuple[str, np.ndarray, bool, str, list[Any]],
    ) -> None:
        """Write a dependency-free, publication-oriented anomaly diagnostic SVG."""

        width, height = 1200, 660
        plot_top, plot_height = 158, 350
        panel_width = 470
        left_x, right_x = 76, 668
        bottom_y = plot_top + plot_height
        observation_label = observation_axis[0]
        observation_values = observation_axis[1]
        observation_is_datetime = observation_axis[2]

        def padded_bounds(
            values: np.ndarray,
            *,
            include: float | None = None,
        ) -> tuple[float, float]:
            finite_values = np.asarray(values, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if include is not None and math.isfinite(include):
                finite_values = np.append(finite_values, include)
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            span = maximum - minimum
            if span == 0:
                span = max(abs(maximum), 1.0)
            padding = span * 0.08
            return minimum - padding, maximum + padding

        def axis_tick(value: float) -> str:
            absolute = abs(value)
            if absolute >= 10_000 or (0 < absolute < 0.001):
                return f"{value:.2e}"
            if absolute >= 100:
                return f"{value:.0f}"
            return f"{value:.3g}"

        def date_tick(value: float, span: float) -> str:
            moment = datetime.fromtimestamp(value, tz=timezone.utc)
            if span > 60 * 60 * 24 * 365 * 3:
                return moment.strftime("%Y")
            if span > 60 * 60 * 24 * 90:
                return moment.strftime("%Y-%m")
            return moment.strftime("%Y-%m-%d")

        left_values = visualization_coordinates[display_indices]
        score_values = anomaly_scores[display_indices]
        axis_values = observation_values[display_indices]
        displayed_mask = anomaly_mask[display_indices]
        left_x_min, left_x_max = padded_bounds(left_values[:, 0])
        left_y_min, left_y_max = padded_bounds(left_values[:, 1])
        right_x_min, right_x_max = padded_bounds(axis_values)
        score_min, score_max = padded_bounds(
            score_values,
            include=decision_threshold,
        )

        def map_x(value: float, panel_x: float, minimum: float, maximum: float) -> float:
            return panel_x + (value - minimum) / (maximum - minimum) * panel_width

        def map_y(value: float, minimum: float, maximum: float) -> float:
            return plot_top + (maximum - value) / (maximum - minimum) * plot_height

        def axes(
            *,
            panel_x: int,
            x_minimum: float,
            x_maximum: float,
            y_minimum: float,
            y_maximum: float,
            x_label: str,
            y_label: str,
            datetime_axis: bool = False,
        ) -> str:
            elements: list[str] = []
            x_span = x_maximum - x_minimum
            for index in range(5):
                value = x_minimum + x_span * index / 4
                x = map_x(value, panel_x, x_minimum, x_maximum)
                tick_text = (
                    date_tick(value, x_span)
                    if datetime_axis
                    else axis_tick(value)
                )
                elements.append(
                    f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" '
                    f'y2="{bottom_y}" class="grid"/>'
                )
                elements.append(
                    f'<text x="{x:.2f}" y="{bottom_y + 24}" text-anchor="middle" '
                    f'class="tick">{escape(tick_text)}</text>'
                )
            y_span = y_maximum - y_minimum
            for index in range(5):
                value = y_minimum + y_span * index / 4
                y = map_y(value, y_minimum, y_maximum)
                elements.append(
                    f'<line x1="{panel_x}" y1="{y:.2f}" '
                    f'x2="{panel_x + panel_width}" y2="{y:.2f}" class="grid"/>'
                )
                elements.append(
                    f'<text x="{panel_x - 12}" y="{y + 4:.2f}" text-anchor="end" '
                    f'class="tick">{escape(axis_tick(value))}</text>'
                )
            elements.extend(
                [
                    f'<line x1="{panel_x}" y1="{plot_top}" x2="{panel_x}" '
                    f'y2="{bottom_y}" class="axis"/>',
                    f'<line x1="{panel_x}" y1="{bottom_y}" '
                    f'x2="{panel_x + panel_width}" y2="{bottom_y}" class="axis"/>',
                    f'<text x="{panel_x + panel_width / 2:.2f}" y="{bottom_y + 53}" '
                    f'text-anchor="middle" class="axis-label">{escape(x_label)}</text>',
                    f'<text x="{panel_x - 54}" y="{plot_top + plot_height / 2:.2f}" '
                    f'text-anchor="middle" class="axis-label" '
                    f'transform="rotate(-90 {panel_x - 54} '
                    f'{plot_top + plot_height / 2:.2f})">{escape(y_label)}</text>',
                ]
            )
            return "".join(elements)

        def marker(
            *,
            x: float,
            y: float,
            is_anomaly: bool,
            source_row: int | str,
            score: float,
            panel: str,
        ) -> str:
            tooltip = escape(
                f"Source row {source_row}; anomaly score {score:.6g}; "
                f"label {'anomaly' if is_anomaly else 'normal'}; panel {panel}"
            )
            source_attribute = escape(str(source_row), quote=True)
            if is_anomaly:
                return (
                    f'<path d="M {x:.2f} {y - 5.5:.2f} L {x + 5.5:.2f} {y:.2f} '
                    f'L {x:.2f} {y + 5.5:.2f} L {x - 5.5:.2f} {y:.2f} Z" '
                    f'class="anomaly-point" data-source-row="{source_attribute}">'
                    f'<title>{tooltip}</title></path>'
                )
            return (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" '
                f'class="normal-point" data-source-row="{source_attribute}">'
                f'<title>{tooltip}</title></circle>'
            )

        left_points: list[str] = []
        right_points: list[str] = []
        for anomaly_value in (False, True):
            for local_index, source_index in enumerate(display_indices):
                if bool(displayed_mask[local_index]) != anomaly_value:
                    continue
                score = float(score_values[local_index])
                source_row = source_rows[int(source_index)]
                left_points.append(
                    marker(
                        x=map_x(
                            float(left_values[local_index, 0]),
                            left_x,
                            left_x_min,
                            left_x_max,
                        ),
                        y=map_y(
                            float(left_values[local_index, 1]),
                            left_y_min,
                            left_y_max,
                        ),
                        is_anomaly=anomaly_value,
                        source_row=source_row,
                        score=score,
                        panel="a",
                    )
                )
                right_points.append(
                    marker(
                        x=map_x(
                            float(axis_values[local_index]),
                            right_x,
                            right_x_min,
                            right_x_max,
                        ),
                        y=map_y(score, score_min, score_max),
                        is_anomaly=anomaly_value,
                        source_row=source_row,
                        score=score,
                        panel="b",
                    )
                )

        threshold_y = map_y(decision_threshold, score_min, score_max)
        panel_a_title = (
            "(a) PCA projection of standardized features"
            if visualization_kind == "pca"
            else "(a) Standardized feature versus anomaly score"
        )
        panel_b_title = (
            "(b) Anomaly scores by source row"
            if observation_label == "Source row"
            else f"(b) Anomaly scores through {observation_label}"
        )
        displayed_rows = int(display_indices.size)
        total_rows = int(anomaly_mask.size)
        sample_note = (
            f"Displayed {displayed_rows:,} of {total_rows:,} usable rows; "
            "model fitting, scores, labels, and CSV use all usable rows."
            if displayed_rows < total_rows
            else f"Displayed all {total_rows:,} usable rows."
        )
        feature_note = (
            f"{len(feature_columns)} standardized feature"
            f"{'s' if len(feature_columns) != 1 else ''}"
        )
        root_title = escape("Anomaly detection diagnostics")
        projection_description = (
            "PCA is used only for visualization."
            if visualization_kind == "pca"
            else "The single standardized feature is shown against anomaly score."
        )
        root_description = escape(
            "Two-panel anomaly diagnostic. Normal observations are circles and "
            f"anomalies are diamonds. {projection_description}"
        )
        subtitle = escape(
            f"{model_display_name} - {source_filename} - {feature_note}"
        )
        scope_note = escape(
            "Projection is visualization only; detection used all selected "
            "standardized features. Higher scores are more anomalous."
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="figure-title figure-description">
<title id="figure-title">{root_title}</title>
<desc id="figure-description">{root_description}</desc>
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#18323a}}
.figure-title{{font-size:24px;font-weight:700}} .subtitle{{font-size:13px;fill:#5c7078}}
.panel-title{{font-size:16px;font-weight:700}} .panel-note{{font-size:12px;fill:#60757d}}
.panel-frame{{fill:#ffffff;stroke:#d6e6e3;stroke-width:1}}
.axis{{stroke:#244c54;stroke-width:1.3}} .grid{{stroke:#dde7e9;stroke-width:1;stroke-dasharray:4 5}}
.tick{{font-size:11px;fill:#60757d}} .axis-label{{font-size:13px;font-weight:600}}
.normal-point{{fill:#287fba;fill-opacity:.64;stroke:#ffffff;stroke-width:.65}}
.anomaly-point{{fill:#dc6a45;fill-opacity:.92;stroke:#71351f;stroke-width:1.1}}
.decision-boundary{{stroke:#68757b;stroke-width:1.5;stroke-dasharray:7 5}}
.threshold-label{{font-size:11px;fill:#59666b}} .legend{{font-size:12px;fill:#456169}}
.footer{{font-size:11px;fill:#60757d}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="42" y="38" class="figure-title">{root_title}</text>
<text x="42" y="61" class="subtitle">{subtitle}</text>
<text x="42" y="82" class="panel-note">{scope_note}</text>
<g class="legend" transform="translate(932 42)">
  <circle cx="0" cy="0" r="4" class="normal-point"/><text x="10" y="4">Normal</text>
  <path d="M 82 -5.5 L 87.5 0 L 82 5.5 L 76.5 0 Z" class="anomaly-point"/><text x="94" y="4">Anomaly</text>
</g>
<text x="{left_x}" y="116" class="panel-title">{escape(panel_a_title)}</text>
<text x="{left_x}" y="137" class="panel-note">{escape(feature_note)}; display coordinates are included in the CSV.</text>
<rect x="{left_x}" y="{plot_top}" width="{panel_width}" height="{plot_height}" class="panel-frame"/>
{axes(panel_x=left_x, x_minimum=left_x_min, x_maximum=left_x_max, y_minimum=left_y_min, y_maximum=left_y_max, x_label=visualization_x_label, y_label=visualization_y_label)}
{''.join(left_points)}
<text x="{right_x}" y="116" class="panel-title">{escape(panel_b_title)}</text>
<text x="{right_x}" y="137" class="panel-note">Dashed line: model decision threshold ({decision_threshold:.4g}).</text>
<rect x="{right_x}" y="{plot_top}" width="{panel_width}" height="{plot_height}" class="panel-frame"/>
{axes(panel_x=right_x, x_minimum=right_x_min, x_maximum=right_x_max, y_minimum=score_min, y_maximum=score_max, x_label=observation_label, y_label='Anomaly score', datetime_axis=observation_is_datetime)}
<line x1="{right_x}" y1="{threshold_y:.2f}" x2="{right_x + panel_width}" y2="{threshold_y:.2f}" class="decision-boundary"/>
<text x="{right_x + panel_width - 4}" y="{threshold_y - 7:.2f}" text-anchor="end" class="threshold-label">decision threshold</text>
{''.join(right_points)}
<text x="42" y="620" class="footer">{escape(sample_note)}</text>
<text x="42" y="640" class="footer">Normal observations use circles; anomalies use diamonds and a contrasting outline, so labels do not depend on colour alone.</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

    @staticmethod
    def _write_element_time_series_svg(
        path: Path,
        *,
        bins: list[TimeSeriesBinItem],
        age_unit: str,
        value_column: str,
        value_unit: str,
        bin_width: float,
    ) -> None:
        valid = [
            item
            for item in bins
            if item.mean_proportion is not None
            and item.uncertainty_2sigma is not None
        ]
        if not valid:
            raise InvalidDatasetError(
                "Element time series requires at least one bin with two samples"
            )
        ages = [item.age for item in valid]
        lower_values = [
            (item.mean_proportion or 0) - (item.uncertainty_2sigma or 0)
            for item in valid
        ]
        upper_values = [
            (item.mean_proportion or 0) + (item.uncertainty_2sigma or 0)
            for item in valid
        ]
        minimum_age, maximum_age = min(ages), max(ages)
        age_span = maximum_age - minimum_age or 1.0
        data_min, data_max = min(lower_values), max(upper_values)
        value_span = data_max - data_min or max(abs(data_max), 1.0)
        y_min = data_min - value_span * 0.08
        y_max = data_max + value_span * 0.08
        width, height = 960, 520
        left, right, top, bottom = 92, 36, 58, 76
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_position(age: float) -> float:
            return left + (maximum_age - age) / age_span * plot_width

        def y_position(value: float) -> float:
            return top + (y_max - value) / (y_max - y_min) * plot_height

        observations = []
        for item in valid:
            mean = item.mean_proportion or 0
            uncertainty = item.uncertainty_2sigma or 0
            x = x_position(item.age)
            mean_y = y_position(mean)
            upper_y = y_position(mean + uncertainty)
            lower_y = y_position(mean - uncertainty)
            observations.append(
                f'<g class="observation"><line x1="{x:.2f}" y1="{upper_y:.2f}" '
                f'x2="{x:.2f}" y2="{lower_y:.2f}" class="error-bar"/>'
                f'<line x1="{x-6:.2f}" y1="{upper_y:.2f}" x2="{x+6:.2f}" '
                f'y2="{upper_y:.2f}" class="error-bar"/>'
                f'<line x1="{x-6:.2f}" y1="{lower_y:.2f}" x2="{x+6:.2f}" '
                f'y2="{lower_y:.2f}" class="error-bar"/>'
                f'<circle cx="{x:.2f}" cy="{mean_y:.2f}" r="4.5" class="point"/></g>'
            )
        grid_lines = []
        for index in range(6):
            value = y_min + (y_max - y_min) * index / 5
            y = y_position(value)
            grid_lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                f'y2="{y:.2f}" class="grid"/>'
            )
            grid_lines.append(
                f'<text x="{left-14}" y="{y+5:.2f}" text-anchor="end" '
                f'class="tick">{value:.3g}</text>'
            )
        title = escape(f"{value_column} mean through time")
        y_label = escape(f"{value_column} ({value_unit})")
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#18323a}} .title{{font-size:22px;font-weight:700}}
.subtitle{{font-size:13px;fill:#5c7078}} .axis{{stroke:#18323a;stroke-width:1.4}}
.grid{{stroke:#dbe5e7;stroke-width:1;stroke-dasharray:4 5}} .tick{{font-size:12px;fill:#526970}}
.error-bar{{stroke:#287fba;stroke-width:1.6}} .point{{fill:#287fba;stroke:#fff;stroke-width:1.2}}
.label{{font-size:14px;font-weight:600}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">{title}</text>
<text x="{left}" y="49" class="subtitle">Bin width {bin_width:g} {age_unit} - independent means +/-2 SEM - no fitted curve</text>
{''.join(grid_lines)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
{''.join(observations)}
<text x="{left}" y="{height-bottom+26}" text-anchor="middle" class="tick">{maximum_age:g}</text>
<text x="{width-right}" y="{height-bottom+26}" text-anchor="middle" class="tick">{minimum_age:g}</text>
<text x="{left + plot_width/2:.2f}" y="{height-22}" text-anchor="middle" class="label">Age ({age_unit})</text>
<text x="24" y="{top + plot_height/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top + plot_height/2:.2f})">{y_label}</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

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

        observations = []
        for item in valid:
            mean = item.mean_proportion or 0
            uncertainty = item.uncertainty_2sigma or 0
            x = x_position(item.age)
            mean_y = y_position(mean)
            upper_y = y_position(mean + uncertainty)
            lower_y = y_position(mean - uncertainty)
            observations.append(
                f'<g class="observation"><line x1="{x:.2f}" y1="{upper_y:.2f}" '
                f'x2="{x:.2f}" y2="{lower_y:.2f}" class="error-bar"/>'
                f'<line x1="{x-6:.2f}" y1="{upper_y:.2f}" x2="{x+6:.2f}" '
                f'y2="{upper_y:.2f}" class="error-bar"/>'
                f'<line x1="{x-6:.2f}" y1="{lower_y:.2f}" x2="{x+6:.2f}" '
                f'y2="{lower_y:.2f}" class="error-bar"/>'
                f'<circle cx="{x:.2f}" cy="{mean_y:.2f}" r="4.5" class="point"/></g>'
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
.error-bar{{stroke:#287fba;stroke-width:1.6}} .point{{fill:#287fba;stroke:#fff;stroke-width:1.2}}
.label{{font-size:14px;font-weight:600}}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">Estimated proportion of subaerial basalts</text>
<text x="{left}" y="49" class="subtitle">Bin width {bin_width:g} {age_unit} - {bootstrap_iterations} bootstrap iterations - independent means +/-2 sigma - no fitted curve</text>
{''.join(grid_lines)}{''.join(x_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
{''.join(observations)}
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

    def validate_upload(self, filename: str | None, content: bytes) -> str:
        """Validate inexpensive upload rules before claiming the compute slot."""
        return self._validate_upload(filename, content)

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
