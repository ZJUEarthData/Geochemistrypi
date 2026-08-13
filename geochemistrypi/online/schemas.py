"""Response models for the Online API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    instance_id: str
    source_revision: str
    build_id: str
    max_upload_bytes: int
    task_timeout_seconds: int
    max_concurrent_tasks: int


class TaskStatusResponse(BaseModel):
    task_id: str
    label: str
    status: Literal[
        "queued",
        "running",
        "cancelling",
        "completed",
        "failed",
        "timed_out",
        "cancelled",
    ]
    progress: int
    queue_position: int | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float
    timeout_seconds: float
    cancellable: bool
    message: str
    error: str | None = None


class InputColumnItem(BaseModel):
    name: str
    label: str
    description: str
    data_type: str
    unit: str
    example: float | int | str
    required: bool = True
    minimum: float | None = None
    exclusive_minimum: bool = False


class MethodCatalogItem(BaseModel):
    name: str
    description: str
    elements: list[str]
    status: Literal["verified", "testing"]
    status_message: str
    formula: str | None = None
    input_columns: list[InputColumnItem] = Field(default_factory=list)
    input_notes: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)


class TaskCatalogItem(BaseModel):
    name: str
    available: bool
    methods: list[MethodCatalogItem] = Field(default_factory=list)
    error: str | None = None


class CatalogResponse(BaseModel):
    tasks: list[TaskCatalogItem]


class ArtifactResponse(BaseModel):
    name: str
    download_url: str
    size_bytes: int


class RunResponse(BaseModel):
    job_id: str
    status: str
    message: str
    artifacts: list[ArtifactResponse]


class HyperparameterItem(BaseModel):
    name: str
    display_name: str
    description: str
    value_type: Literal["integer", "number", "boolean", "select"]
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: list[Any] = Field(default_factory=list)


class DataMiningMethodItem(BaseModel):
    name: str
    display_name: str
    description: str
    status: Literal["verified", "testing"] = "verified"
    uses_cluster_count: bool = False
    hyperparameters: list[HyperparameterItem] = Field(default_factory=list)


class DataMiningFeatureItem(BaseModel):
    name: str
    description: str
    status: Literal["verified", "testing"]
    status_message: str
    input_formats: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    methods: list[DataMiningMethodItem] = Field(default_factory=list)


class DataMiningCatalogResponse(BaseModel):
    features: list[DataMiningFeatureItem]


class DatasetProfileSummary(BaseModel):
    rows: int
    columns: int
    total_cells: int
    missing_cells: int
    missing_rate: float
    duplicate_rows: int
    numeric_columns: int
    text_columns: int
    datetime_columns: int
    boolean_columns: int
    infinite_cells: int
    memory_bytes: int


class ColumnProfileItem(BaseModel):
    name: str
    data_type: Literal["number", "text", "datetime", "boolean"]
    pandas_dtype: str
    non_null: int
    missing: int
    missing_rate: float
    unique: int
    minimum: float | str | None = None
    maximum: float | str | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    sample_values: list[Any] = Field(default_factory=list)


class DatasetProfileResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    summary: DatasetProfileSummary
    columns: list[ColumnProfileItem]
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class DataPreprocessingSummary(BaseModel):
    original_rows: int
    original_columns: int
    processed_rows: int
    processed_columns: int
    removed_rows: int
    removed_columns: int
    original_missing_cells: int
    processed_missing_cells: int
    filled_cells: int


class DataPreprocessingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    selected_columns: list[str]
    missing_strategy: Literal[
        "keep",
        "drop_rows",
        "fill_mean",
        "fill_median",
        "fill_mode",
    ]
    summary: DataPreprocessingSummary
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class RegressionSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    train_rows: int
    test_rows: int
    feature_count: int


class RegressionMetrics(BaseModel):
    r2: float | None
    mean_absolute_error: float
    root_mean_squared_error: float


class RegressionCoefficientItem(BaseModel):
    feature: str
    coefficient: float


class CrossValidationMetricItem(BaseModel):
    name: str
    display_name: str
    mean: float
    standard_deviation: float
    higher_is_better: bool


class CrossValidationResult(BaseModel):
    folds: int
    strategy: str
    random_state: int
    metrics: list[CrossValidationMetricItem]


class RegressionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    model: str
    model_display_name: str
    target_column: str
    feature_columns: list[str]
    test_size: float
    random_state: int
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cross_validation: CrossValidationResult | None = None
    summary: RegressionSummary
    metrics: RegressionMetrics
    intercept: float | None = None
    coefficients: list[RegressionCoefficientItem]
    equation: str | None = None
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class ClassificationSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    class_count: int


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


class ClassificationConfusionItem(BaseModel):
    actual_class: str
    predicted_class: str
    count: int


class ClassificationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    model: str
    model_display_name: str
    target_column: str
    feature_columns: list[str]
    test_size: float
    random_state: int
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cross_validation: CrossValidationResult | None = None
    classes: list[str]
    summary: ClassificationSummary
    metrics: ClassificationMetrics
    confusion_matrix: list[ClassificationConfusionItem]
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class ModelComparisonItem(BaseModel):
    rank: int
    model: str
    model_display_name: str
    status: Literal["success", "failed"]
    primary_score: float | None = None
    metrics: list[CrossValidationMetricItem] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelComparisonResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    task_type: Literal["regression", "classification"]
    target_column: str
    feature_columns: list[str]
    cross_validation_folds: int
    comparison_metric: str
    best_model: str | None = None
    results: list[ModelComparisonItem]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class ModelInferenceSummary(BaseModel):
    original_rows: int
    predicted_rows: int
    excluded_rows: int
    imputed_rows: int
    feature_count: int


class ModelInferenceResponse(BaseModel):
    job_id: str
    training_job_id: str
    status: str
    message: str
    source_filename: str
    task_type: Literal["regression", "classification"]
    model: str
    model_display_name: str
    target_column: str
    feature_columns: list[str]
    prediction_column: str
    pipeline_schema_version: str
    software_version: str
    summary: ModelInferenceSummary
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class ClusteringSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    feature_count: int
    cluster_count: int


class ClusteringMetrics(BaseModel):
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float


class ClusterSizeItem(BaseModel):
    cluster: int
    rows: int


class ClusterCenterItem(BaseModel):
    cluster: int
    values: dict[str, float]


class ClusteringResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    model: str
    model_display_name: str
    feature_columns: list[str]
    cluster_count: int
    requested_cluster_count: int | None = None
    noise_rows: int = 0
    random_state: int
    summary: ClusteringSummary
    metrics: ClusteringMetrics
    cluster_sizes: list[ClusterSizeItem]
    cluster_centers: list[ClusterCenterItem]
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class DimensionalityReductionSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    feature_count: int
    component_count: int


class DimensionalityReductionMetrics(BaseModel):
    explained_variance_ratio: list[float] = Field(default_factory=list)
    cumulative_explained_variance_ratio: list[float] = Field(
        default_factory=list
    )
    total_explained_variance_ratio: float | None = None
    kl_divergence: float | None = None
    stress: float | None = None


class DimensionalityReductionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    model: str
    model_display_name: str
    feature_columns: list[str]
    component_count: int
    random_state: int
    summary: DimensionalityReductionSummary
    metrics: DimensionalityReductionMetrics
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class AnomalyDetectionSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    feature_count: int
    normal_rows: int
    anomaly_rows: int


class AnomalyScoreSummary(BaseModel):
    minimum: float
    maximum: float
    mean: float


class AnomalyDetectionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    model: str
    model_display_name: str
    feature_columns: list[str]
    random_state: int | None = None
    summary: AnomalyDetectionSummary
    score_summary: AnomalyScoreSummary
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)


class TimeSeriesSummary(BaseModel):
    original_rows: int
    usable_rows: int
    dropped_rows: int
    sampled_out_rows: int = 0
    bin_count: int
    populated_bins: int


class TimeSeriesBinItem(BaseModel):
    age: float
    mean_proportion: float | None = None
    uncertainty_2sigma: float | None = None
    sample_count: int = 0


class ProbabilityModelMetrics(BaseModel):
    validation_rows: int
    mean_absolute_error: float
    root_mean_squared_error: float
    r2: float


class ProbabilityModelInfo(BaseModel):
    version: str
    display_name: str
    training_rows: int
    training_sha256: str
    recognized_features: list[str]
    metrics: ProbabilityModelMetrics
    target_description: str


class ProbabilityPredictionSummary(BaseModel):
    predicted_rows: int
    insufficient_feature_rows: int
    eligible_time_series_rows: int
    sampled_time_series_rows: int
    minimum_features_per_row: int


class TimeSeriesResponse(BaseModel):
    job_id: str
    status: str
    message: str
    source_filename: str
    age_column: str
    age_max_column: str | None = None
    probability_column: str | None = None
    latitude_column: str | None = None
    longitude_column: str | None = None
    age_unit: str
    bin_width: float
    bootstrap_iterations: int = 0
    random_state: int | None = None
    analysis_type: Literal["subaerial_proportion", "element_mean"] = (
        "subaerial_proportion"
    )
    value_column: str | None = None
    value_unit: str | None = None
    uncertainty_method: Literal["bootstrap_2sigma", "2_sem"] = "bootstrap_2sigma"
    filter_column: str | None = None
    filter_min: float | None = None
    filter_max: float | None = None
    probability_source: Literal["uploaded", "model_predicted"] = "uploaded"
    probability_model: ProbabilityModelInfo | None = None
    prediction_summary: ProbabilityPredictionSummary | None = None
    summary: TimeSeriesSummary
    bins: list[TimeSeriesBinItem]
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
