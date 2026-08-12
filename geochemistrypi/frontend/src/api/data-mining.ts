import { API_BASE_URL, taskHeaders, type ArtifactResponse } from '@/api/online'

export type DataMiningStatus = 'verified' | 'testing'

export interface DataMiningMethodItem {
  name: string
  display_name: string
  description: string
  status: DataMiningStatus
  uses_cluster_count: boolean
}

export interface DataMiningFeatureItem {
  name: string
  description: string
  status: DataMiningStatus
  status_message: string
  input_formats: string[]
  outputs: string[]
  methods: DataMiningMethodItem[]
}

export interface DataMiningCatalogResponse {
  features: DataMiningFeatureItem[]
}

export interface DatasetProfileSummary {
  rows: number
  columns: number
  total_cells: number
  missing_cells: number
  missing_rate: number
  duplicate_rows: number
  numeric_columns: number
  text_columns: number
  datetime_columns: number
  boolean_columns: number
  infinite_cells: number
  memory_bytes: number
}

export interface ColumnProfileItem {
  name: string
  data_type: 'number' | 'text' | 'datetime' | 'boolean'
  pandas_dtype: string
  non_null: number
  missing: number
  missing_rate: number
  unique: number
  minimum: number | string | null
  maximum: number | string | null
  mean: number | null
  standard_deviation: number | null
  sample_values: unknown[]
}

export interface DatasetProfileResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  summary: DatasetProfileSummary
  columns: ColumnProfileItem[]
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export type MissingValueStrategy = 'keep' | 'drop_rows' | 'fill_mean' | 'fill_median' | 'fill_mode'

export interface DataPreprocessingSummary {
  original_rows: number
  original_columns: number
  processed_rows: number
  processed_columns: number
  removed_rows: number
  removed_columns: number
  original_missing_cells: number
  processed_missing_cells: number
  filled_cells: number
}

export interface DataPreprocessingResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  selected_columns: string[]
  missing_strategy: MissingValueStrategy
  summary: DataPreprocessingSummary
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface RegressionSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  train_rows: number
  test_rows: number
  feature_count: number
}

export interface RegressionMetrics {
  r2: number | null
  mean_absolute_error: number
  root_mean_squared_error: number
}

export interface RegressionCoefficientItem {
  feature: string
  coefficient: number
}

export interface RegressionResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  model: string
  model_display_name: string
  target_column: string
  feature_columns: string[]
  test_size: number
  random_state: number
  summary: RegressionSummary
  metrics: RegressionMetrics
  intercept: number
  coefficients: RegressionCoefficientItem[]
  equation: string
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface ClassificationSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  train_rows: number
  test_rows: number
  feature_count: number
  class_count: number
}

export interface ClassificationMetrics {
  accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
}

export interface ClassificationConfusionItem {
  actual_class: string
  predicted_class: string
  count: number
}

export interface ClassificationResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  model: string
  model_display_name: string
  target_column: string
  feature_columns: string[]
  test_size: number
  random_state: number
  classes: string[]
  summary: ClassificationSummary
  metrics: ClassificationMetrics
  confusion_matrix: ClassificationConfusionItem[]
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface ModelInferenceSummary {
  original_rows: number
  predicted_rows: number
  excluded_rows: number
  imputed_rows: number
  feature_count: number
}

export interface ModelInferenceResponse {
  job_id: string
  training_job_id: string
  status: string
  message: string
  source_filename: string
  task_type: 'regression' | 'classification'
  model: string
  model_display_name: string
  target_column: string
  feature_columns: string[]
  prediction_column: string
  pipeline_schema_version: string
  software_version: string
  summary: ModelInferenceSummary
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface ClusteringSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  feature_count: number
  cluster_count: number
}

export interface ClusteringMetrics {
  silhouette_score: number
  davies_bouldin_score: number
  calinski_harabasz_score: number
}

export interface ClusterSizeItem {
  cluster: number
  rows: number
}

export interface ClusterCenterItem {
  cluster: number
  values: Record<string, number>
}

export interface ClusteringResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  model: string
  model_display_name: string
  feature_columns: string[]
  cluster_count: number
  requested_cluster_count: number | null
  noise_rows: number
  random_state: number
  summary: ClusteringSummary
  metrics: ClusteringMetrics
  cluster_sizes: ClusterSizeItem[]
  cluster_centers: ClusterCenterItem[]
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface DimensionalityReductionSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  feature_count: number
  component_count: number
}

export interface DimensionalityReductionMetrics {
  explained_variance_ratio: number[]
  cumulative_explained_variance_ratio: number[]
  total_explained_variance_ratio: number | null
  kl_divergence: number | null
  stress: number | null
}

export interface DimensionalityReductionResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  model: string
  model_display_name: string
  feature_columns: string[]
  component_count: number
  random_state: number
  summary: DimensionalityReductionSummary
  metrics: DimensionalityReductionMetrics
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface AnomalyDetectionSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  feature_count: number
  normal_rows: number
  anomaly_rows: number
}

export interface AnomalyScoreSummary {
  minimum: number
  maximum: number
  mean: number
}

export interface AnomalyDetectionResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  model: string
  model_display_name: string
  feature_columns: string[]
  random_state: number | null
  summary: AnomalyDetectionSummary
  score_summary: AnomalyScoreSummary
  preview: Record<string, unknown>[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

export interface TimeSeriesSummary {
  original_rows: number
  usable_rows: number
  dropped_rows: number
  sampled_out_rows: number
  bin_count: number
  populated_bins: number
}

export interface TimeSeriesBinItem {
  age: number
  mean_proportion: number | null
  uncertainty_2sigma: number | null
  sample_count: number
}

export interface ProbabilityModelInfo {
  version: string
  display_name: string
  training_rows: number
  training_sha256: string
  recognized_features: string[]
  metrics: {
    validation_rows: number
    mean_absolute_error: number
    root_mean_squared_error: number
    r2: number
  }
  target_description: string
}

export interface ProbabilityPredictionSummary {
  predicted_rows: number
  insufficient_feature_rows: number
  eligible_time_series_rows: number
  sampled_time_series_rows: number
  minimum_features_per_row: number
}

export interface TimeSeriesResponse {
  job_id: string
  status: string
  message: string
  source_filename: string
  age_column: string
  age_max_column: string | null
  probability_column: string | null
  latitude_column: string | null
  longitude_column: string | null
  age_unit: 'Ma' | 'Ga'
  bin_width: number
  bootstrap_iterations: number
  random_state: number | null
  analysis_type: 'subaerial_proportion' | 'element_mean'
  value_column: string | null
  value_unit: string | null
  uncertainty_method: 'bootstrap_2sigma' | '2_sem'
  filter_column: string | null
  filter_min: number | null
  filter_max: number | null
  probability_source: 'uploaded' | 'model_predicted'
  probability_model: ProbabilityModelInfo | null
  prediction_summary: ProbabilityPredictionSummary | null
  summary: TimeSeriesSummary
  bins: TimeSeriesBinItem[]
  warnings: string[]
  artifacts: ArtifactResponse[]
}

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    const message =
      typeof detail === 'string' ? detail : JSON.stringify(detail || payload || response.statusText)
    throw new Error(message)
  }
  return payload as T
}

export async function getDataMiningCatalog(): Promise<DataMiningCatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/api/data-mining/catalog`)
  return parseResponse<DataMiningCatalogResponse>(response)
}

export async function profileDataset(dataset: File, taskId?: string): Promise<DatasetProfileResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/profile`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<DatasetProfileResponse>(response)
}

export async function preprocessDataset(
  dataset: File,
  selectedColumns: string[],
  missingStrategy: MissingValueStrategy,
  taskId?: string
): Promise<DataPreprocessingResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('selected_columns', JSON.stringify(selectedColumns))
  form.append('missing_strategy', missingStrategy)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/preprocess`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<DataPreprocessingResponse>(response)
}

export async function runRegression(
  dataset: File,
  targetColumn: string,
  featureColumns: string[],
  testSize: number,
  model: string = 'linear_regression',
  taskId?: string
): Promise<RegressionResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('target_column', targetColumn)
  form.append('feature_columns', JSON.stringify(featureColumns))
  form.append('test_size', String(testSize))
  form.append('model', model)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/regression`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<RegressionResponse>(response)
}

export async function runClassification(
  dataset: File,
  targetColumn: string,
  featureColumns: string[],
  testSize: number,
  model: string = 'logistic_regression',
  taskId?: string
): Promise<ClassificationResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('target_column', targetColumn)
  form.append('feature_columns', JSON.stringify(featureColumns))
  form.append('test_size', String(testSize))
  form.append('model', model)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/classification`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<ClassificationResponse>(response)
}

export async function runModelInference(
  trainingJobId: string,
  dataset: File,
  taskId?: string
): Promise<ModelInferenceResponse> {
  const form = new FormData()
  form.append('training_job_id', trainingJobId)
  form.append('dataset', dataset)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/inference`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<ModelInferenceResponse>(response)
}

export async function runClustering(
  dataset: File,
  featureColumns: string[],
  clusterCount: number,
  model: string = 'kmeans',
  taskId?: string
): Promise<ClusteringResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('feature_columns', JSON.stringify(featureColumns))
  form.append('cluster_count', String(clusterCount))
  form.append('model', model)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/clustering`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<ClusteringResponse>(response)
}

export async function runDimensionalityReduction(
  dataset: File,
  featureColumns: string[],
  componentCount: number,
  model: string = 'pca',
  taskId?: string
): Promise<DimensionalityReductionResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('feature_columns', JSON.stringify(featureColumns))
  form.append('component_count', String(componentCount))
  form.append('model', model)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/dimensionality-reduction`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<DimensionalityReductionResponse>(response)
}

export async function runAnomalyDetection(
  dataset: File,
  featureColumns: string[],
  model: string = 'isolation_forest',
  taskId?: string
): Promise<AnomalyDetectionResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('feature_columns', JSON.stringify(featureColumns))
  form.append('model', model)
  const response = await fetch(`${API_BASE_URL}/api/data-mining/anomaly-detection`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<AnomalyDetectionResponse>(response)
}

export async function runTimeSeries(
  dataset: File,
  columns: {
    age: string
    ageMax: string
    probability: string
    latitude: string
    longitude: string
  },
  ageUnit: 'Ma' | 'Ga',
  binWidth: number,
  bootstrapIterations: number,
  taskId?: string
): Promise<TimeSeriesResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('age_column', columns.age)
  form.append('age_max_column', columns.ageMax)
  form.append('probability_column', columns.probability)
  form.append('latitude_column', columns.latitude)
  form.append('longitude_column', columns.longitude)
  form.append('age_unit', ageUnit)
  form.append('bin_width', String(binWidth))
  form.append('bootstrap_iterations', String(bootstrapIterations))
  const response = await fetch(`${API_BASE_URL}/api/data-mining/time-series`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<TimeSeriesResponse>(response)
}

export async function runPredictedTimeSeries(
  dataset: File,
  columns: {
    age: string
    ageMax: string
    latitude: string
    longitude: string
  },
  ageUnit: 'Ma' | 'Ga',
  binWidth: number,
  bootstrapIterations: number,
  taskId?: string
): Promise<TimeSeriesResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('age_column', columns.age)
  form.append('age_max_column', columns.ageMax)
  form.append('latitude_column', columns.latitude)
  form.append('longitude_column', columns.longitude)
  form.append('age_unit', ageUnit)
  form.append('bin_width', String(binWidth))
  form.append('bootstrap_iterations', String(bootstrapIterations))
  const response = await fetch(`${API_BASE_URL}/api/data-mining/time-series/predict`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<TimeSeriesResponse>(response)
}

export async function runElementTimeSeries(
  dataset: File,
  columns: {
    age: string
    value: string
    filter?: string
  },
  ageUnit: 'Ma' | 'Ga',
  binWidth: number,
  valueUnit: string,
  filterRange?: { minimum: number; maximum: number },
  taskId?: string
): Promise<TimeSeriesResponse> {
  const form = new FormData()
  form.append('dataset', dataset)
  form.append('age_column', columns.age)
  form.append('value_column', columns.value)
  form.append('age_unit', ageUnit)
  form.append('bin_width', String(binWidth))
  form.append('value_unit', valueUnit)
  if (columns.filter && filterRange) {
    form.append('filter_column', columns.filter)
    form.append('filter_min', String(filterRange.minimum))
    form.append('filter_max', String(filterRange.maximum))
  }
  const response = await fetch(`${API_BASE_URL}/api/data-mining/time-series/element-mean`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<TimeSeriesResponse>(response)
}
