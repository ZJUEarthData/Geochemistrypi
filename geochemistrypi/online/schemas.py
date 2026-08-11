"""Response models for the Online API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


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


class DataMiningMethodItem(BaseModel):
    name: str
    display_name: str
    description: str
    status: Literal["verified", "testing"] = "verified"
    uses_cluster_count: bool = False


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
    summary: RegressionSummary
    metrics: RegressionMetrics
    intercept: float
    coefficients: list[RegressionCoefficientItem]
    equation: str
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
    classes: list[str]
    summary: ClassificationSummary
    metrics: ClassificationMetrics
    confusion_matrix: list[ClassificationConfusionItem]
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
