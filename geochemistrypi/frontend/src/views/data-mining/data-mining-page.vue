<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElInputNumber, ElMessage } from 'element-plus'

import MobileFieldCards from '@/components/mobile-field-cards.vue'
import RunSummary from '@/components/run-summary.vue'
import TaskProgress from '@/components/task-progress.vue'
import { useTaskTracking } from '@/composables/use-task-tracking'

import {
  getDataMiningCatalog,
  preprocessDataset,
  profileDataset,
  runAnomalyDetection,
  runClassification,
  runClustering,
  runDimensionalityReduction,
  runElementTimeSeries,
  runModelInference,
  runModelComparison,
  runPredictedTimeSeries,
  runRegression,
  runTimeSeries,
  type AnomalyContamination,
  type AnomalyDetectionResponse,
  type AnomalyReproductionProfile,
  type ClassificationResponse,
  type ClusteringResponse,
  type DataPreprocessingResponse,
  type DataMiningFeatureItem,
  type DatasetProfileResponse,
  type DimensionalityReductionResponse,
  type MissingValueStrategy,
  type ModelInferenceResponse,
  type ModelComparisonResponse,
  type HyperparameterItem,
  type RegressionResponse,
  type TimeSeriesResponse
} from '@/api/data-mining'
import {
  DEFAULT_MAX_UPLOAD_BYTES,
  artifactUrl,
  getHealth,
  type ArtifactResponse
} from '@/api/online'
import { apiText, dataMiningFeatureDescription, t, warningIsSuccess } from '@/i18n'

type ServiceState = 'checking' | 'online' | 'offline'
type TimeSeriesMode = 'direct' | 'model_predicted' | 'element_mean'
type SupervisedRunMode = 'single' | 'compare'
type HyperparameterValue = string | number | boolean

const serviceState = ref<ServiceState>('checking')
const softwareVersion = ref('')
const maxUploadBytes = ref(DEFAULT_MAX_UPLOAD_BYTES)
const taskTimeoutMinutes = ref(30)
const maxConcurrentTasks = ref(1)
const loadingCatalog = ref(true)
const running = ref(false)
const inspectingColumns = ref(false)
const features = ref<DataMiningFeatureItem[]>([])
const selectedFeature = ref('')
const datasetFile = ref<File | null>(null)
const resourceLimitNote = computed(() =>
  t(
    `Maximum file size: ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB. The site runs ${maxConcurrentTasks.value} calculation at a time; additional jobs wait in the queue. A running job stops after ${taskTimeoutMinutes.value} minutes.`,
    `文件大小不得超过 ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB。全站同时只运行 ${maxConcurrentTasks.value} 个计算任务，其他任务排队等待；任务开始运行 ${taskTimeoutMinutes.value} 分钟后将自动停止。`
  )
)
const result = ref<DatasetProfileResponse | null>(null)
const columnInspection = ref<DatasetProfileResponse | null>(null)
const preprocessingResult = ref<DataPreprocessingResponse | null>(null)
const regressionResult = ref<RegressionResponse | null>(null)
const classificationResult = ref<ClassificationResponse | null>(null)
const modelComparisonResult = ref<ModelComparisonResponse | null>(null)
const applicationDataFile = ref<File | null>(null)
const inferenceResult = ref<ModelInferenceResponse | null>(null)
const inferenceError = ref('')
const runningInference = ref(false)
const clusteringResult = ref<ClusteringResponse | null>(null)
const dimensionalityReductionResult = ref<DimensionalityReductionResponse | null>(null)
const anomalyDetectionResult = ref<AnomalyDetectionResponse | null>(null)
const exportingAnomalyPng = ref('')
const timeSeriesResult = ref<TimeSeriesResponse | null>(null)
const selectedColumns = ref<string[]>([])
const missingStrategy = ref<MissingValueStrategy>('keep')
const regressionTarget = ref('')
const regressionFeatures = ref<string[]>([])
const regressionTestSize = ref(0.2)
const regressionModel = ref('linear_regression')
const regressionRunMode = ref<SupervisedRunMode>('single')
const regressionComparisonModels = ref<string[]>(['linear_regression', 'random_forest', 'xgboost'])
const regressionHyperparameters = ref<Record<string, Record<string, HyperparameterValue>>>({})
const regressionCrossValidationEnabled = ref(false)
const regressionCrossValidationFolds = ref(5)
const classificationTarget = ref('')
const classificationFeatures = ref<string[]>([])
const classificationTestSize = ref(0.2)
const classificationModel = ref('logistic_regression')
const classificationRunMode = ref<SupervisedRunMode>('single')
const classificationComparisonModels = ref<string[]>([
  'logistic_regression',
  'random_forest',
  'xgboost'
])
const classificationHyperparameters = ref<Record<string, Record<string, HyperparameterValue>>>({})
const classificationCrossValidationEnabled = ref(false)
const classificationCrossValidationFolds = ref(5)
const clusteringFeatures = ref<string[]>([])
const clusterCount = ref(3)
const clusteringModel = ref('kmeans')
const dimensionalityReductionFeatures = ref<string[]>([])
const dimensionalityReductionModel = ref('pca')
const componentCount = ref(2)
const anomalyDetectionFeatures = ref<string[]>([])
const anomalyDetectionModel = ref('isolation_forest')
const anomalyContaminationAutomatic = ref(true)
const anomalyContamination = ref(0.05)
const anomalyReproductionProfile = ref<AnomalyReproductionProfile>('general')
const timeSeriesMode = ref<TimeSeriesMode>('direct')
const timeSeriesAgeColumn = ref('')
const timeSeriesAgeMaxColumn = ref('')
const timeSeriesProbabilityColumn = ref('')
const timeSeriesLatitudeColumn = ref('')
const timeSeriesLongitudeColumn = ref('')
const timeSeriesValueColumn = ref('')
const timeSeriesValueUnit = ref('wt%')
const timeSeriesFilterColumn = ref('')
const timeSeriesFilterMinimum = ref(43)
const timeSeriesFilterMaximum = ref(51)
const timeSeriesAgeUnit = ref<'Ma' | 'Ga'>('Ma')
const timeSeriesBinWidth = ref(10)
const timeSeriesBootstrapIterations = ref(100)
const errorMessage = ref('')
const {
  taskId,
  taskStatus,
  cancellingTask,
  cancelledByUser,
  beginTask,
  finishTask,
  cancelCurrentTask
} = useTaskTracking()
const {
  taskId: inferenceTaskId,
  taskStatus: inferenceTaskStatus,
  cancellingTask: cancellingInference,
  cancelledByUser: inferenceCancelledByUser,
  beginTask: beginInferenceTask,
  finishTask: finishInferenceTask,
  cancelCurrentTask: cancelInferenceTask
} = useTaskTracking()

const missingStrategyOptions = computed<
  Array<{
    value: MissingValueStrategy
    label: string
    description: string
  }>
>(() => [
  {
    value: 'keep',
    label: t('Keep missing values', '保留缺失值'),
    description: t('Do not replace or remove missing values.', '不替换或删除缺失值。')
  },
  {
    value: 'drop_rows',
    label: t('Drop incomplete rows', '删除不完整行'),
    description: t(
      'Remove rows containing a missing value in any selected column.',
      '删除任一已选列中含缺失值的行。'
    )
  },
  {
    value: 'fill_mean',
    label: t('Fill numeric columns with mean', '用均值填充数值列'),
    description: t(
      'Fill numeric missing values with the mean of each column.',
      '用各列均值填充数值缺失值。'
    )
  },
  {
    value: 'fill_median',
    label: t('Fill numeric columns with median', '用中位数填充数值列'),
    description: t(
      'Fill numeric missing values with the median of each column.',
      '用各列中位数填充数值缺失值。'
    )
  },
  {
    value: 'fill_mode',
    label: t('Fill each column with mode', '用众数填充各列'),
    description: t(
      'Fill missing values with the most frequent non-empty value in each column.',
      '用每列最常见的非空值填充缺失值。'
    )
  }
])

const currentFeature = computed(() =>
  features.value.find((feature) => feature.name === selectedFeature.value)
)
const currentFeatureIsVerified = computed(() => currentFeature.value?.status === 'verified')
const isPreprocessing = computed(() => selectedFeature.value === 'data_preprocessing')
const isRegression = computed(() => selectedFeature.value === 'regression')
const isClassification = computed(() => selectedFeature.value === 'classification')
const isClustering = computed(() => selectedFeature.value === 'clustering')
const isDimensionalityReduction = computed(
  () => selectedFeature.value === 'dimensionality_reduction'
)
const isAnomalyDetection = computed(() => selectedFeature.value === 'anomaly_detection')
const isTimeSeries = computed(() => selectedFeature.value === 'time_series')
const previewColumns = computed(() => Object.keys(result.value?.preview[0] || {}))
const preprocessingPreviewColumns = computed(() =>
  Object.keys(preprocessingResult.value?.preview[0] || {})
)
const regressionPreviewColumns = computed(() =>
  Object.keys(regressionResult.value?.preview[0] || {})
)
const classificationPreviewColumns = computed(() =>
  Object.keys(classificationResult.value?.preview[0] || {})
)
const trainedSupervisedResult = computed(() => regressionResult.value || classificationResult.value)
const inferencePreviewColumns = computed(() => Object.keys(inferenceResult.value?.preview[0] || {}))
const clusteringPreviewColumns = computed(() =>
  Object.keys(clusteringResult.value?.preview[0] || {})
)
const dimensionalityReductionPreviewColumns = computed(() =>
  Object.keys(dimensionalityReductionResult.value?.preview[0] || {})
)
const anomalyDetectionPreviewColumns = computed(() =>
  Object.keys(anomalyDetectionResult.value?.preview[0] || {})
)
const effectiveAnomalyContamination = computed<AnomalyContamination>(() =>
  anomalyContaminationAutomatic.value ? 'auto' : anomalyContamination.value
)
const anomalyProfileIsLocked = computed(() => anomalyReproductionProfile.value !== 'general')
const anomalyFigureArtifact = computed(() =>
  anomalyDetectionResult.value?.artifacts.find(
    (artifact) => artifact.name === 'anomaly_detection_figure.svg'
  )
)
const paperReproductionFigureArtifact = computed(() =>
  anomalyDetectionResult.value?.artifacts.find(
    (artifact) => artifact.name === 'paper_reproduction_figure.svg'
  )
)
const completedAnomalyReproductionProfile = computed<AnomalyReproductionProfile>(
  () => anomalyDetectionResult.value?.reproduction_profile || anomalyReproductionProfile.value
)
const anomalyFigureCards = computed<
  Array<{
    artifact: ArtifactResponse
    kicker: string
    title: string
    note: string
    methodNote: string
    alt: string
    reference: boolean
  }>
>(() => {
  const cards = []
  if (anomalyFigureArtifact.value) {
    cards.push({
      artifact: anomalyFigureArtifact.value,
      kicker: t('ANOMALY VISUALIZATION', '异常检测可视化'),
      title: t('Projected distribution and anomaly scores', '投影分布与异常分数'),
      note: t(
        'The projection is used only for visualization; detection labels and scores were computed in the complete standardized feature space.',
        '投影仅用于可视化；异常标签与分数由完整的标准化特征空间计算。'
      ),
      methodNote: '',
      alt: t('Anomaly detection diagnostic figure', '异常检测诊断图'),
      reference: false
    })
  }
  if (paperReproductionFigureArtifact.value) {
    const profile = completedAnomalyReproductionProfile.value
    cards.push({
      artifact: paperReproductionFigureArtifact.value,
      kicker: t('PAPER REPRODUCTION', '论文图件复现'),
      title: anomalyReproductionProfileTitle(profile),
      note: t(
        'This reference reconstruction follows the selected paper figure contract and uses archived paper coordinates or labels when the uploaded audit workbook supplies them. The summary cards and Online diagnostic above report the freshly computed Online model; agreement between the two evidence layers is not implied.',
        '该参考复现图遵循所选论文的图件规范，并在上传的审计工作簿提供时使用论文存档坐标或标签。上方统计卡和 Online 诊断图报告的是本次 Online 模型重新计算的结果；两类证据并不默认一致。'
      ),
      methodNote:
        profile === 'zhu_2024_figure_8a'
          ? t(
              'For Zhu et al. (2024), P = 0.08 is the published target outlier proportion; M = 30 d is the earthquake-linkage window, not an LOF fitting parameter.',
              '对于 Zhu et al. (2024)，P = 0.08 是论文设定的目标异常比例；M = 30 d 是地震关联时间窗，并非 LOF 拟合参数。'
            )
          : t(
              'For Sharapatov et al. (2025), the PC1-PC2 plane is a visualization layer; paper-reference markers must remain distinct from labels produced by this Online run.',
              '对于 Sharapatov et al. (2025)，PC1-PC2 平面属于可视化层；论文参考标记必须与本次 Online 运行生成的标签明确区分。'
            ),
      alt: t('Published-reference paper reproduction figure', '基于论文参考结果的复现图'),
      reference: true
    })
  }
  return cards
})
const anomalyDownloadArtifacts = computed(() =>
  (anomalyDetectionResult.value?.artifacts || []).filter(
    (artifact) =>
      !['anomaly_detection_figure.svg', 'paper_reproduction_figure.svg'].includes(artifact.name)
  )
)
const timeSeriesMappedColumns = computed(() => {
  if (timeSeriesMode.value === 'element_mean') {
    return [
      timeSeriesAgeColumn.value,
      timeSeriesValueColumn.value,
      ...(timeSeriesFilterColumn.value ? [timeSeriesFilterColumn.value] : [])
    ]
  }
  const columns = [
    timeSeriesAgeColumn.value,
    timeSeriesAgeMaxColumn.value,
    timeSeriesLatitudeColumn.value,
    timeSeriesLongitudeColumn.value
  ]
  if (timeSeriesMode.value === 'direct') columns.splice(2, 0, timeSeriesProbabilityColumn.value)
  return columns
})
const timeSeriesRequiredColumnCount = computed(() =>
  timeSeriesMode.value === 'element_mean'
    ? timeSeriesFilterColumn.value
      ? 3
      : 2
    : timeSeriesMode.value === 'direct'
      ? 5
      : 4
)
const timeSeriesMappingComplete = computed(
  () =>
    timeSeriesMappedColumns.value.every(Boolean) &&
    new Set(timeSeriesMappedColumns.value).size === timeSeriesMappedColumns.value.length
)
const timeSeriesChart = computed(() => {
  const valid = (timeSeriesResult.value?.bins || []).filter(
    (item) => item.mean_proportion !== null && item.uncertainty_2sigma !== null
  )
  if (!valid.length) return null
  const width = 760
  const height = 320
  const left = 64
  const right = 20
  const top = 24
  const bottom = 48
  const minimumAge = Math.min(...valid.map((item) => item.age))
  const maximumAge = Math.max(...valid.map((item) => item.age))
  const ageSpan = maximumAge - minimumAge || 1
  const x = (age: number) => left + ((maximumAge - age) / ageSpan) * (width - left - right)
  const lowerValues = valid.map(
    (item) => (item.mean_proportion || 0) - (item.uncertainty_2sigma || 0)
  )
  const upperValues = valid.map(
    (item) => (item.mean_proportion || 0) + (item.uncertainty_2sigma || 0)
  )
  const rawMinimum =
    timeSeriesResult.value?.analysis_type === 'element_mean' ? Math.min(...lowerValues) : 0
  const rawMaximum =
    timeSeriesResult.value?.analysis_type === 'element_mean' ? Math.max(...upperValues) : 100
  const rawSpan = rawMaximum - rawMinimum || Math.max(Math.abs(rawMaximum), 1)
  const yMinimum =
    timeSeriesResult.value?.analysis_type === 'element_mean' ? rawMinimum - rawSpan * 0.08 : 0
  const yMaximum =
    timeSeriesResult.value?.analysis_type === 'element_mean' ? rawMaximum + rawSpan * 0.08 : 100
  const y = (value: number) =>
    top + ((yMaximum - value) / (yMaximum - yMinimum)) * (height - top - bottom)
  const yTicks = Array.from(
    { length: 6 },
    (_, index) => yMinimum + ((yMaximum - yMinimum) * index) / 5
  )
  const points = valid.map((item) => {
    const mean = item.mean_proportion as number
    const uncertainty = item.uncertainty_2sigma as number
    return {
      age: item.age,
      mean,
      uncertainty,
      x: x(item.age),
      y: y(mean),
      upperY: y(mean + uncertainty),
      lowerY: y(mean - uncertainty)
    }
  })
  return {
    width,
    height,
    left,
    right,
    top,
    bottom,
    minimumAge,
    maximumAge,
    yTicks,
    points,
    y
  }
})
const numericColumns = computed(
  () =>
    columnInspection.value?.columns
      .filter((column) => column.data_type === 'number')
      .map((column) => column.name) || []
)
const regressionFeatureOptions = computed(() =>
  numericColumns.value.filter((column) => column !== regressionTarget.value)
)
const regressionMethodOptions = computed(() => currentFeature.value?.methods || [])
const selectedRegressionMethod = computed(() =>
  regressionMethodOptions.value.find((method) => method.name === regressionModel.value)
)
const classificationTargetColumns = computed(
  () => columnInspection.value?.columns.map((column) => column.name) || []
)
const classificationFeatureOptions = computed(() =>
  numericColumns.value.filter((column) => column !== classificationTarget.value)
)
const classificationMethodOptions = computed(() => currentFeature.value?.methods || [])
const selectedClassificationMethod = computed(() =>
  classificationMethodOptions.value.find((method) => method.name === classificationModel.value)
)
const comparisonBestModelDisplayName = computed(
  () =>
    modelComparisonResult.value?.results.find(
      (item) => item.model === modelComparisonResult.value?.best_model
    )?.model_display_name || '—'
)

function initializeModelHyperparameters(
  methods: Array<{ name: string; hyperparameters: HyperparameterItem[] }>,
  destination: Record<string, Record<string, HyperparameterValue>>
) {
  for (const method of methods) {
    if (destination[method.name]) continue
    destination[method.name] = Object.fromEntries(
      (method.hyperparameters || []).map((parameter) => [parameter.name, parameter.default])
    )
  }
}

function getParameterValue(
  taskType: 'regression' | 'classification',
  model: string,
  parameter: HyperparameterItem
) {
  const store =
    taskType === 'regression'
      ? regressionHyperparameters.value
      : classificationHyperparameters.value
  store[model] ||= {}
  if (!(parameter.name in store[model])) store[model][parameter.name] = parameter.default
  return store[model][parameter.name]
}

function setParameterValue(
  taskType: 'regression' | 'classification',
  model: string,
  parameter: HyperparameterItem,
  value: HyperparameterValue
) {
  const store =
    taskType === 'regression'
      ? regressionHyperparameters.value
      : classificationHyperparameters.value
  store[model] ||= {}
  store[model][parameter.name] = value
  modelComparisonResult.value = null
  if (taskType === 'regression') regressionResult.value = null
  else classificationResult.value = null
}

function regressionMethodByName(name: string) {
  return regressionMethodOptions.value.find((method) => method.name === name)
}

function classificationMethodByName(name: string) {
  return classificationMethodOptions.value.find((method) => method.name === name)
}

function comparisonMetric(item: { metrics: Array<{ name: string; mean: number }> }, name: string) {
  return item.metrics.find((metric) => metric.name === name)?.mean ?? null
}

function comparisonMetricStandardDeviation(
  item: { metrics: Array<{ name: string; standard_deviation: number }> },
  name: string
) {
  return item.metrics.find((metric) => metric.name === name)?.standard_deviation ?? null
}
const clusteringMethodOptions = computed(() => currentFeature.value?.methods || [])
const selectedClusteringMethod = computed(() =>
  clusteringMethodOptions.value.find((method) => method.name === clusteringModel.value)
)
const clusteringUsesClusterCount = computed(
  () => selectedClusteringMethod.value?.uses_cluster_count ?? true
)
const dimensionalityReductionMethodOptions = computed(() => currentFeature.value?.methods || [])
const selectedDimensionalityReductionMethod = computed(() =>
  dimensionalityReductionMethodOptions.value.find(
    (method) => method.name === dimensionalityReductionModel.value
  )
)
const anomalyDetectionMethodOptions = computed(() => currentFeature.value?.methods || [])
const selectedAnomalyDetectionMethod = computed(() =>
  anomalyDetectionMethodOptions.value.find((method) => method.name === anomalyDetectionModel.value)
)
const selectedAnomalyReproductionDescription = computed(() =>
  anomalyReproductionProfileDescription(anomalyReproductionProfile.value)
)
const selectedStrategy = computed(() =>
  missingStrategyOptions.value.find((option) => option.value === missingStrategy.value)
)
const canRun = computed(() => {
  const baseReady =
    serviceState.value === 'online' &&
    currentFeatureIsVerified.value &&
    Boolean(datasetFile.value) &&
    !running.value
  if (!baseReady) return false
  if (isPreprocessing.value) {
    return selectedColumns.value.length > 0 && !inspectingColumns.value
  }
  if (isRegression.value) {
    return (
      Boolean(regressionTarget.value) &&
      regressionFeatures.value.length > 0 &&
      (regressionRunMode.value === 'single' || regressionComparisonModels.value.length >= 2) &&
      !inspectingColumns.value
    )
  }
  if (isClassification.value) {
    return (
      Boolean(classificationTarget.value) &&
      classificationFeatures.value.length > 0 &&
      (classificationRunMode.value === 'single' ||
        classificationComparisonModels.value.length >= 2) &&
      !inspectingColumns.value
    )
  }
  if (isClustering.value) {
    return clusteringFeatures.value.length > 0 && !inspectingColumns.value
  }
  if (isDimensionalityReduction.value) {
    return (
      dimensionalityReductionFeatures.value.length >= componentCount.value &&
      !inspectingColumns.value
    )
  }
  if (isAnomalyDetection.value) {
    return anomalyDetectionFeatures.value.length > 0 && !inspectingColumns.value
  }
  if (isTimeSeries.value) {
    return (
      timeSeriesMappingComplete.value &&
      timeSeriesBinWidth.value > 0 &&
      (timeSeriesMode.value === 'element_mean' ||
        (timeSeriesBootstrapIterations.value >= 10 &&
          timeSeriesBootstrapIterations.value <= 1000)) &&
      (timeSeriesMode.value !== 'element_mean' ||
        !timeSeriesFilterColumn.value ||
        timeSeriesFilterMinimum.value <= timeSeriesFilterMaximum.value) &&
      !inspectingColumns.value
    )
  }
  return selectedFeature.value === 'dataset_profile'
})
const runButtonLabel = computed(() => {
  if (running.value) {
    if (isPreprocessing.value) return t('Processing…', '正在处理…')
    if (isRegression.value || isClassification.value) return t('Training…', '正在训练…')
    if (isClustering.value) return t('Clustering…', '正在聚类…')
    if (isDimensionalityReduction.value) return t('Reducing dimensions…', '正在降维…')
    if (isAnomalyDetection.value) return t('Detecting anomalies…', '正在检测异常…')
    if (isTimeSeries.value)
      return timeSeriesMode.value === 'element_mean'
        ? t('Calculating element means…', '正在计算元素均值…')
        : timeSeriesMode.value === 'direct'
          ? t('Calculating time series…', '正在计算时间序列…')
          : t('Predicting probability and calculating…', '正在预测概率并计算…')
    return t('Analyzing…', '正在分析…')
  }
  if (!currentFeatureIsVerified.value)
    return t('This function is not available yet', '该功能暂不可运行')
  if (isPreprocessing.value) return t('Run preprocessing', '运行预处理')
  if (isRegression.value)
    return regressionRunMode.value === 'compare'
      ? t('Compare regression models', '比较回归模型')
      : t('Run regression', '运行回归')
  if (isClassification.value)
    return classificationRunMode.value === 'compare'
      ? t('Compare classification models', '比较分类模型')
      : t('Run classification', '运行分类')
  if (isClustering.value) return t('Run clustering', '运行聚类')
  if (isDimensionalityReduction.value) return t('Run dimensionality reduction', '运行降维')
  if (isAnomalyDetection.value) return t('Run anomaly detection', '运行异常检测')
  if (isTimeSeries.value) return t('Run time series analysis', '运行时间序列分析')
  return t('Analyze dataset', '分析数据集')
})

const operationCompleted = computed(() =>
  Boolean(
    result.value ||
    preprocessingResult.value ||
    regressionResult.value ||
    classificationResult.value ||
    clusteringResult.value ||
    dimensionalityReductionResult.value ||
    anomalyDetectionResult.value ||
    timeSeriesResult.value
  )
)

const activeJobId = computed(
  () =>
    result.value?.job_id ||
    preprocessingResult.value?.job_id ||
    regressionResult.value?.job_id ||
    classificationResult.value?.job_id ||
    clusteringResult.value?.job_id ||
    dimensionalityReductionResult.value?.job_id ||
    anomalyDetectionResult.value?.job_id ||
    timeSeriesResult.value?.job_id ||
    columnInspection.value?.job_id ||
    ''
)

const runSummaryStatus = computed(() => {
  if (serviceState.value === 'checking') return t('Checking service', '正在检查服务')
  if (serviceState.value === 'offline') return t('Backend offline', '后端离线')
  if (inspectingColumns.value) return t('Inspecting dataset', '正在检查数据集')
  if (taskStatus.value?.status === 'queued') return t('Waiting in queue', '正在排队')
  if (taskStatus.value?.status === 'cancelling') return t('Cancelling', '正在取消')
  if (taskStatus.value?.status === 'cancelled') return t('Cancelled', '已取消')
  if (running.value) return t('Running analysis', '正在运行分析')
  if (operationCompleted.value) return t('Completed', '已完成')
  if (errorMessage.value) return t('Needs attention', '需要检查')
  if (datasetFile.value && columnInspection.value) return t('Ready to run', '可开始运行')
  return t('Waiting for dataset', '等待数据集')
})

const runSummaryTone = computed(() => {
  if (serviceState.value === 'offline' || errorMessage.value) return 'danger' as const
  if (running.value || inspectingColumns.value || serviceState.value === 'checking')
    return 'info' as const
  if (operationCompleted.value) return 'success' as const
  if (datasetFile.value) return 'warning' as const
  return 'neutral' as const
})

const runSummaryMethod = computed(() =>
  currentFeature.value
    ? dataMiningFeatureDescription(currentFeature.value.name, currentFeature.value.description)
    : ''
)

const runSummaryParameters = computed(() => {
  if (isPreprocessing.value) {
    return [
      `${t('Columns', '列')}: ${selectedColumns.value.length}`,
      `${t('Missing values', '缺失值')}: ${selectedStrategy.value?.label || '—'}`
    ]
  }
  if (isRegression.value) {
    return [
      `${t('Target', '目标')}: ${regressionTarget.value || '—'}`,
      `${t('Model', '模型')}: ${
        regressionMethodOptions.value.find((method) => method.name === regressionModel.value)
          ?.display_name || regressionModel.value
      }`,
      `${t('Features', '特征')}: ${regressionFeatures.value.length}`,
      `${t('Test size', '测试集')}: ${formatPercent(regressionTestSize.value)}`
    ]
  }
  if (isClassification.value) {
    return [
      `${t('Target', '目标')}: ${classificationTarget.value || '—'}`,
      `${t('Model', '模型')}: ${
        classificationMethodOptions.value.find(
          (method) => method.name === classificationModel.value
        )?.display_name || classificationModel.value
      }`,
      `${t('Features', '特征')}: ${classificationFeatures.value.length}`,
      `${t('Test size', '测试集')}: ${formatPercent(classificationTestSize.value)}`
    ]
  }
  if (isClustering.value) {
    const parameters = [
      `${t('Model', '模型')}: ${selectedClusteringMethod.value?.display_name || clusteringModel.value}`,
      `${t('Features', '特征')}: ${clusteringFeatures.value.length}`
    ]
    if (clusteringUsesClusterCount.value) parameters.push(`k: ${clusterCount.value}`)
    else parameters.push(t('Cluster count: automatic', '簇数：自动估计'))
    return parameters
  }
  if (isDimensionalityReduction.value) {
    return [
      `${t('Model', '模型')}: ${selectedDimensionalityReductionMethod.value?.display_name || dimensionalityReductionModel.value}`,
      `${t('Features', '特征')}: ${dimensionalityReductionFeatures.value.length}`,
      `${t('Components', '维度')}: ${componentCount.value}`
    ]
  }
  if (isAnomalyDetection.value) {
    return [
      `${t('Model', '模型')}: ${selectedAnomalyDetectionMethod.value?.display_name || anomalyDetectionModel.value}`,
      `${t('Features', '特征')}: ${anomalyDetectionFeatures.value.length}`,
      `${t('Template', '模板')}: ${anomalyReproductionProfileTitle(anomalyReproductionProfile.value)}`,
      `${t('Contamination', '异常比例')}: ${formatAnomalyContamination(effectiveAnomalyContamination.value)}`
    ]
  }
  if (isTimeSeries.value) {
    if (timeSeriesMode.value === 'element_mean') {
      return [
        `${t('Analysis', '分析')}: ${t('Element mean', '元素均值')}`,
        `${t('Target', '目标')}: ${timeSeriesValueColumn.value || '—'}`,
        `${t('Bin width', '分箱宽度')}: ${timeSeriesBinWidth.value} ${timeSeriesAgeUnit.value}`,
        `${t('Uncertainty', '不确定性')}: ±2 SEM`
      ]
    }
    return [
      `${t('Age unit', '年龄单位')}: ${timeSeriesAgeUnit.value}`,
      `${t('Bin width', '分箱宽度')}: ${timeSeriesBinWidth.value} ${timeSeriesAgeUnit.value}`,
      `${t('Bootstrap iterations', 'Bootstrap 次数')}: ${timeSeriesBootstrapIterations.value}`,
      `${t('Probability source', '概率来源')}: ${timeSeriesMode.value === 'direct' ? t('Uploaded column', '上传数据列') : t('Liu 2024 surrogate', 'Liu 2024 替代模型')}`,
      `${t('Mapped columns', '映射列')}: ${timeSeriesMappedColumns.value.filter(Boolean).length}/${timeSeriesRequiredColumnCount.value}`
    ]
  }
  return [`${t('Scope', '范围')}: ${t('Full dataset', '完整数据集')}`]
})

watch(selectedFeature, async () => {
  clearResult()
  errorMessage.value = ''
  columnInspection.value = null
  selectedColumns.value = []
  regressionTarget.value = ''
  regressionFeatures.value = []
  classificationTarget.value = ''
  classificationFeatures.value = []
  clusteringFeatures.value = []
  dimensionalityReductionFeatures.value = []
  anomalyDetectionFeatures.value = []
  resetTimeSeriesColumns()
  if (datasetFile.value) {
    await inspectDatasetColumns()
  }
})
watch(missingStrategy, () => {
  preprocessingResult.value = null
})
watch(
  selectedColumns,
  () => {
    preprocessingResult.value = null
  },
  { deep: true }
)
watch(regressionTarget, (target) => {
  regressionFeatures.value = regressionFeatures.value.filter((feature) => feature !== target)
  regressionResult.value = null
})
watch(
  regressionFeatures,
  () => {
    regressionResult.value = null
  },
  { deep: true }
)
watch(regressionTestSize, () => {
  regressionResult.value = null
})
watch(regressionModel, () => {
  regressionResult.value = null
})
watch(regressionRunMode, () => {
  regressionResult.value = null
  modelComparisonResult.value = null
})
watch(
  [regressionComparisonModels, regressionCrossValidationEnabled, regressionCrossValidationFolds],
  () => {
    regressionResult.value = null
    modelComparisonResult.value = null
  },
  { deep: true }
)
watch(classificationTarget, (target) => {
  classificationFeatures.value = classificationFeatures.value.filter(
    (feature) => feature !== target
  )
  classificationResult.value = null
})
watch(
  classificationFeatures,
  () => {
    classificationResult.value = null
  },
  { deep: true }
)
watch(classificationTestSize, () => {
  classificationResult.value = null
})
watch(classificationModel, () => {
  classificationResult.value = null
})
watch(classificationRunMode, () => {
  classificationResult.value = null
  modelComparisonResult.value = null
})
watch(
  [
    classificationComparisonModels,
    classificationCrossValidationEnabled,
    classificationCrossValidationFolds
  ],
  () => {
    classificationResult.value = null
    modelComparisonResult.value = null
  },
  { deep: true }
)
watch(
  clusteringFeatures,
  () => {
    clusteringResult.value = null
  },
  { deep: true }
)
watch(clusterCount, () => {
  clusteringResult.value = null
})
watch(clusteringModel, () => {
  clusteringResult.value = null
})
watch(
  dimensionalityReductionFeatures,
  () => {
    dimensionalityReductionResult.value = null
  },
  { deep: true }
)
watch(dimensionalityReductionModel, () => {
  dimensionalityReductionResult.value = null
})
watch(componentCount, () => {
  dimensionalityReductionResult.value = null
})
watch(
  anomalyDetectionFeatures,
  () => {
    anomalyDetectionResult.value = null
  },
  { deep: true }
)
watch(anomalyDetectionModel, () => {
  anomalyDetectionResult.value = null
})
watch(anomalyReproductionProfile, (profile) => {
  if (profile === 'sharapatov_2025_figure_3a') {
    anomalyDetectionModel.value = 'isolation_forest'
    anomalyContaminationAutomatic.value = false
    anomalyContamination.value = 0.05
  } else if (profile === 'zhu_2024_figure_8a') {
    anomalyDetectionModel.value = 'local_outlier_factor'
    anomalyContaminationAutomatic.value = false
    anomalyContamination.value = 0.08
  } else {
    anomalyContaminationAutomatic.value = true
  }
  anomalyDetectionResult.value = null
})
watch([anomalyContaminationAutomatic, anomalyContamination], () => {
  anomalyDetectionResult.value = null
})
watch(timeSeriesMode, (mode) => {
  if (mode !== 'element_mean') return
  timeSeriesAgeUnit.value = 'Ma'
  timeSeriesBinWidth.value = 100
  timeSeriesValueColumn.value ||= findDetectedColumn('MGO', 'MgO')
  timeSeriesFilterColumn.value ||= findDetectedColumn('SIO2', 'SiO2')
})
watch(
  [
    timeSeriesMode,
    timeSeriesAgeColumn,
    timeSeriesAgeMaxColumn,
    timeSeriesProbabilityColumn,
    timeSeriesLatitudeColumn,
    timeSeriesLongitudeColumn,
    timeSeriesValueColumn,
    timeSeriesValueUnit,
    timeSeriesFilterColumn,
    timeSeriesFilterMinimum,
    timeSeriesFilterMaximum,
    timeSeriesAgeUnit,
    timeSeriesBinWidth,
    timeSeriesBootstrapIterations
  ],
  () => {
    timeSeriesResult.value = null
    const mappingMessages = [2, 3, 4, 5].map((count) =>
      t(
        `Map all ${count} required time-series variables to different numeric columns.`,
        `请将 ${count} 个必需的时间序列变量分别映射到不同数值列。`
      )
    )
    if (timeSeriesMappingComplete.value && mappingMessages.includes(errorMessage.value)) {
      errorMessage.value = ''
    }
  }
)
onMounted(loadPage)

async function loadPage() {
  serviceState.value = 'checking'
  loadingCatalog.value = true
  errorMessage.value = ''
  try {
    const health = await getHealth()
    softwareVersion.value = health.version
    maxUploadBytes.value = health.max_upload_bytes
    taskTimeoutMinutes.value = Math.round(health.task_timeout_seconds / 60)
    maxConcurrentTasks.value = health.max_concurrent_tasks
    serviceState.value = 'online'
    const catalog = await getDataMiningCatalog()
    features.value = catalog.features
    const regressionMethods =
      features.value.find((feature) => feature.name === 'regression')?.methods || []
    const classificationMethods =
      features.value.find((feature) => feature.name === 'classification')?.methods || []
    initializeModelHyperparameters(regressionMethods, regressionHyperparameters.value)
    initializeModelHyperparameters(classificationMethods, classificationHyperparameters.value)
    regressionComparisonModels.value = regressionComparisonModels.value.filter((model) =>
      regressionMethods.some((method) => method.name === model)
    )
    classificationComparisonModels.value = classificationComparisonModels.value.filter((model) =>
      classificationMethods.some((method) => method.name === model)
    )
    selectedFeature.value =
      features.value.find((feature) => feature.status === 'verified')?.name ||
      features.value[0]?.name ||
      ''
  } catch (error) {
    serviceState.value = 'offline'
    errorMessage.value =
      error instanceof Error
        ? error.message
        : t('Cannot connect to the backend service.', '无法连接后端服务。')
  } finally {
    loadingCatalog.value = false
  }
}

async function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] || null
  clearResult()
  columnInspection.value = null
  selectedColumns.value = []
  regressionTarget.value = ''
  regressionFeatures.value = []
  classificationTarget.value = ''
  classificationFeatures.value = []
  clusteringFeatures.value = []
  dimensionalityReductionFeatures.value = []
  anomalyDetectionFeatures.value = []
  resetTimeSeriesColumns()
  errorMessage.value = ''

  if (file && !/\.(xlsx|csv)$/i.test(file.name)) {
    datasetFile.value = null
    input.value = ''
    errorMessage.value = t(
      'Data Mining currently supports .xlsx and .csv files.',
      '数据挖掘目前支持 .xlsx 和 .csv 文件。'
    )
    return
  }
  if (file && file.size > maxUploadBytes.value) {
    datasetFile.value = null
    input.value = ''
    errorMessage.value = t(
      `The selected file exceeds the ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB upload limit.`,
      `所选文件超过 ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB 上传限制。`
    )
    return
  }
  datasetFile.value = file
  if (file) {
    await inspectDatasetColumns()
  }
}

async function inspectDatasetColumns() {
  if (!datasetFile.value) return

  inspectingColumns.value = true
  columnInspection.value = null
  selectedColumns.value = []
  regressionTarget.value = ''
  regressionFeatures.value = []
  classificationTarget.value = ''
  classificationFeatures.value = []
  clusteringFeatures.value = []
  dimensionalityReductionFeatures.value = []
  anomalyDetectionFeatures.value = []
  errorMessage.value = ''
  const trackingId = beginTask('Dataset profile')
  try {
    columnInspection.value = await profileDataset(datasetFile.value, trackingId)
    if (isPreprocessing.value) {
      selectedColumns.value = columnInspection.value.columns.map((column) => column.name)
    } else if (isRegression.value) {
      const detectedNumericColumns = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      regressionTarget.value = detectedNumericColumns[detectedNumericColumns.length - 1] || ''
      regressionFeatures.value = detectedNumericColumns.slice(0, -1)
      if (detectedNumericColumns.length < 2) {
        errorMessage.value = t(
          'Regression requires at least two numeric columns: one target and one feature.',
          '回归至少需要两个数值列：一个目标列和一个特征列。'
        )
      }
    } else if (isClassification.value) {
      const detectedColumns = columnInspection.value.columns.map((column) => column.name)
      const detectedNumericColumns = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      classificationTarget.value = detectedColumns[detectedColumns.length - 1] || ''
      classificationFeatures.value = detectedNumericColumns.filter(
        (column) => column !== classificationTarget.value
      )
      if (!classificationTarget.value || classificationFeatures.value.length === 0) {
        errorMessage.value = t(
          'Classification requires a target column and at least one numeric feature column.',
          '分类需要一个目标列和至少一个数值特征列。'
        )
      }
    } else if (isClustering.value) {
      clusteringFeatures.value = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      if (clusteringFeatures.value.length === 0) {
        errorMessage.value = t(
          'Clustering requires at least one numeric feature column.',
          '聚类至少需要一个数值特征列。'
        )
      }
    } else if (isDimensionalityReduction.value) {
      dimensionalityReductionFeatures.value = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      if (dimensionalityReductionFeatures.value.length < 2) {
        errorMessage.value = t(
          'Dimensionality reduction requires at least two numeric feature columns.',
          '降维至少需要两个数值特征列。'
        )
      }
    } else if (isAnomalyDetection.value) {
      anomalyDetectionFeatures.value = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      if (anomalyDetectionFeatures.value.length === 0) {
        errorMessage.value = t(
          'Anomaly detection requires at least one numeric feature column.',
          '异常检测至少需要一个数值特征列。'
        )
      }
    } else if (isTimeSeries.value) {
      timeSeriesAgeColumn.value = findDetectedColumn('R_AGE', 'AGE', 'Age')
      timeSeriesAgeMaxColumn.value = findDetectedColumn('R_MAX_AGE', 'MAX_AGE', 'AgeMax')
      timeSeriesProbabilityColumn.value = findDetectedColumn(
        'Estimated Proportion of Subaerial Basalts',
        'SBAP',
        'Probability'
      )
      timeSeriesMode.value = timeSeriesProbabilityColumn.value ? 'direct' : 'model_predicted'
      timeSeriesLatitudeColumn.value = findDetectedColumn('LATITUDE', 'Latitude')
      timeSeriesLongitudeColumn.value = findDetectedColumn('LONGITUDE', 'Longitude')
      timeSeriesValueColumn.value = findDetectedColumn('MGO', 'MgO')
      timeSeriesFilterColumn.value = findDetectedColumn('SIO2', 'SiO2')
      if (!timeSeriesMappingComplete.value) {
        errorMessage.value = t(
          `Map all ${timeSeriesRequiredColumnCount.value} required time-series variables to different numeric columns.`,
          `请将 ${timeSeriesRequiredColumnCount.value} 个必需的时间序列变量分别映射到不同数值列。`
        )
      }
    }
  } catch (error) {
    if (!cancelledByUser.value) {
      errorMessage.value =
        error instanceof Error
          ? error.message
          : t('Could not inspect dataset columns.', '无法检测数据集列。')
    }
  } finally {
    await finishTask()
    inspectingColumns.value = false
  }
}

async function submitJob() {
  if (!canRun.value || !datasetFile.value) return

  running.value = true
  clearResult()
  errorMessage.value = ''
  const trackingId = beginTask(runSummaryMethod.value || 'Data Mining')
  try {
    if (isPreprocessing.value) {
      preprocessingResult.value = await preprocessDataset(
        datasetFile.value,
        selectedColumns.value,
        missingStrategy.value,
        trackingId
      )
      ElMessage.success(t('Data preprocessing completed', '数据预处理完成'))
    } else if (isRegression.value) {
      if (regressionRunMode.value === 'compare') {
        modelComparisonResult.value = await runModelComparison(
          datasetFile.value,
          'regression',
          regressionTarget.value,
          regressionFeatures.value,
          regressionComparisonModels.value,
          Object.fromEntries(
            regressionComparisonModels.value.map((model) => [
              model,
              regressionHyperparameters.value[model] || {}
            ])
          ),
          regressionCrossValidationFolds.value,
          trackingId
        )
        ElMessage.success(t('Regression model comparison completed', '回归模型比较已完成'))
      } else {
        regressionResult.value = await runRegression(
          datasetFile.value,
          regressionTarget.value,
          regressionFeatures.value,
          regressionTestSize.value,
          regressionModel.value,
          regressionHyperparameters.value[regressionModel.value] || {},
          regressionCrossValidationEnabled.value ? regressionCrossValidationFolds.value : 0,
          trackingId
        )
        ElMessage.success(
          `${regressionResult.value.model_display_name} ${t('completed', '已完成')}`
        )
      }
    } else if (isClassification.value) {
      if (classificationRunMode.value === 'compare') {
        modelComparisonResult.value = await runModelComparison(
          datasetFile.value,
          'classification',
          classificationTarget.value,
          classificationFeatures.value,
          classificationComparisonModels.value,
          Object.fromEntries(
            classificationComparisonModels.value.map((model) => [
              model,
              classificationHyperparameters.value[model] || {}
            ])
          ),
          classificationCrossValidationFolds.value,
          trackingId
        )
        ElMessage.success(t('Classification model comparison completed', '分类模型比较已完成'))
      } else {
        classificationResult.value = await runClassification(
          datasetFile.value,
          classificationTarget.value,
          classificationFeatures.value,
          classificationTestSize.value,
          classificationModel.value,
          classificationHyperparameters.value[classificationModel.value] || {},
          classificationCrossValidationEnabled.value ? classificationCrossValidationFolds.value : 0,
          trackingId
        )
        ElMessage.success(
          `${classificationResult.value.model_display_name} ${t('completed', '已完成')}`
        )
      }
    } else if (isClustering.value) {
      clusteringResult.value = await runClustering(
        datasetFile.value,
        clusteringFeatures.value,
        clusterCount.value,
        clusteringModel.value,
        trackingId
      )
      ElMessage.success(`${clusteringResult.value.model_display_name} ${t('completed', '已完成')}`)
    } else if (isDimensionalityReduction.value) {
      dimensionalityReductionResult.value = await runDimensionalityReduction(
        datasetFile.value,
        dimensionalityReductionFeatures.value,
        componentCount.value,
        dimensionalityReductionModel.value,
        trackingId
      )
      ElMessage.success(
        `${dimensionalityReductionResult.value.model_display_name} ${t('completed', '已完成')}`
      )
    } else if (isAnomalyDetection.value) {
      anomalyDetectionResult.value = await runAnomalyDetection(
        datasetFile.value,
        anomalyDetectionFeatures.value,
        anomalyDetectionModel.value,
        effectiveAnomalyContamination.value,
        anomalyReproductionProfile.value,
        trackingId
      )
      ElMessage.success(
        `${anomalyDetectionResult.value.model_display_name} ${t('completed', '已完成')}`
      )
    } else if (isTimeSeries.value) {
      const sharedColumns = {
        age: timeSeriesAgeColumn.value,
        ageMax: timeSeriesAgeMaxColumn.value,
        latitude: timeSeriesLatitudeColumn.value,
        longitude: timeSeriesLongitudeColumn.value
      }
      timeSeriesResult.value =
        timeSeriesMode.value === 'element_mean'
          ? await runElementTimeSeries(
              datasetFile.value,
              {
                age: timeSeriesAgeColumn.value,
                value: timeSeriesValueColumn.value,
                filter: timeSeriesFilterColumn.value || undefined
              },
              timeSeriesAgeUnit.value,
              timeSeriesBinWidth.value,
              timeSeriesValueUnit.value,
              timeSeriesFilterColumn.value
                ? {
                    minimum: timeSeriesFilterMinimum.value,
                    maximum: timeSeriesFilterMaximum.value
                  }
                : undefined,
              trackingId
            )
          : timeSeriesMode.value === 'direct'
            ? await runTimeSeries(
                datasetFile.value,
                { ...sharedColumns, probability: timeSeriesProbabilityColumn.value },
                timeSeriesAgeUnit.value,
                timeSeriesBinWidth.value,
                timeSeriesBootstrapIterations.value,
                trackingId
              )
            : await runPredictedTimeSeries(
                datasetFile.value,
                sharedColumns,
                timeSeriesAgeUnit.value,
                timeSeriesBinWidth.value,
                timeSeriesBootstrapIterations.value,
                trackingId
              )
      ElMessage.success(t('Time series analysis completed', '时间序列分析完成'))
    } else {
      result.value = await profileDataset(datasetFile.value, trackingId)
      ElMessage.success(t('Dataset profile completed', '数据集概览完成'))
    }
  } catch (error) {
    if (cancelledByUser.value) {
      ElMessage.info(t('Task cancelled', '任务已取消'))
    } else {
      errorMessage.value =
        error instanceof Error
          ? error.message
          : t('The Data Mining operation failed.', '数据挖掘操作失败。')
      ElMessage.error(t('Data Mining operation failed', '数据挖掘操作失败'))
    }
  } finally {
    await finishTask()
    running.value = false
  }
}

function onApplicationDataChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] || null
  inferenceResult.value = null
  inferenceError.value = ''
  if (file && !/\.(xlsx|csv)$/i.test(file.name)) {
    applicationDataFile.value = null
    input.value = ''
    inferenceError.value = t(
      'Application Data must be an .xlsx or .csv file.',
      'Application Data 必须是 .xlsx 或 .csv 文件。'
    )
    return
  }
  if (file && file.size > maxUploadBytes.value) {
    applicationDataFile.value = null
    input.value = ''
    inferenceError.value = t(
      `The selected file exceeds the ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB upload limit.`,
      `所选文件超过 ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB 上传限制。`
    )
    return
  }
  applicationDataFile.value = file
}

async function submitInference() {
  if (!trainedSupervisedResult.value || !applicationDataFile.value || runningInference.value) return
  runningInference.value = true
  inferenceResult.value = null
  inferenceError.value = ''
  const trackingId = beginInferenceTask('Application Data inference')
  try {
    inferenceResult.value = await runModelInference(
      trainedSupervisedResult.value.job_id,
      applicationDataFile.value,
      trackingId
    )
    ElMessage.success(t('Application inference completed', 'Application Data 推理完成'))
  } catch (error) {
    if (inferenceCancelledByUser.value) {
      ElMessage.info(t('Inference cancelled', '推理任务已取消'))
    } else {
      inferenceError.value =
        error instanceof Error
          ? error.message
          : t('Application inference failed.', 'Application Data 推理失败。')
      ElMessage.error(t('Application inference failed', 'Application Data 推理失败'))
    }
  } finally {
    await finishInferenceTask()
    runningInference.value = false
  }
}

function clearResult() {
  result.value = null
  preprocessingResult.value = null
  regressionResult.value = null
  classificationResult.value = null
  modelComparisonResult.value = null
  clusteringResult.value = null
  dimensionalityReductionResult.value = null
  anomalyDetectionResult.value = null
  timeSeriesResult.value = null
  applicationDataFile.value = null
  inferenceResult.value = null
  inferenceError.value = ''
}

function resetTimeSeriesColumns() {
  timeSeriesMode.value = 'direct'
  timeSeriesAgeColumn.value = ''
  timeSeriesAgeMaxColumn.value = ''
  timeSeriesProbabilityColumn.value = ''
  timeSeriesLatitudeColumn.value = ''
  timeSeriesLongitudeColumn.value = ''
  timeSeriesValueColumn.value = ''
  timeSeriesValueUnit.value = 'wt%'
  timeSeriesFilterColumn.value = ''
  timeSeriesFilterMinimum.value = 43
  timeSeriesFilterMaximum.value = 51
}

function findDetectedColumn(...candidates: string[]) {
  const normalized = candidates.map((candidate) => candidate.toLowerCase())
  return (
    columnInspection.value?.columns.find((column) => normalized.includes(column.name.toLowerCase()))
      ?.name || ''
  )
}

function selectAllColumns() {
  selectedColumns.value = columnInspection.value?.columns.map((column) => column.name) || []
}

function clearSelectedColumns() {
  selectedColumns.value = []
}

function anomalyReproductionProfileTitle(profile: AnomalyReproductionProfile) {
  if (profile === 'sharapatov_2025_figure_3a') {
    return 'Sharapatov et al. (2025), Figure 3a'
  }
  if (profile === 'zhu_2024_figure_8a') return 'Zhu et al. (2024), Figure 8a'
  return t('General diagnostics', '通用诊断')
}

function anomalyReproductionProfileDescription(profile: AnomalyReproductionProfile) {
  if (profile === 'sharapatov_2025_figure_3a') {
    return t(
      'Paper template: Isolation Forest with a fixed 5% contamination target and a PC1-PC2 reproduction panel.',
      '论文模板：固定使用 5% 目标异常比例的 Isolation Forest，并生成 PC1-PC2 复现面板。'
    )
  }
  if (profile === 'zhu_2024_figure_8a') {
    return t(
      'Paper template: Local Outlier Factor with P = 0.08 and the archived GA ratio/event timeline when the audit sheets are present.',
      '论文模板：使用 P = 0.08 的 Local Outlier Factor；当审计工作表存在时生成 GA 离子比值与事件时间线。'
    )
  }
  return t(
    'Generate the standard Online diagnostic figure with the selected detector and an automatic or user-defined contamination threshold.',
    '使用所选检测算法和自动或自定义异常比例阈值生成标准 Online 诊断图。'
  )
}

function formatAnomalyContamination(value: AnomalyContamination) {
  return value === 'auto' ? t('Automatic', '自动') : formatPercent(value)
}

function formatLabel(value: string) {
  return value.replace(/_/g, ' ')
}

function statusLabel(status: 'verified' | 'testing') {
  return status === 'verified' ? t('Verified', '已验证') : t('Testing', '测试中')
}

function statusType(status: 'verified' | 'testing') {
  return status === 'verified' ? 'success' : 'warning'
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function formatNumber(value: number | string | null, digits = 3) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: digits })
  }
  return value
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

async function downloadAnomalyFigurePng(artifact: ArtifactResponse) {
  if (exportingAnomalyPng.value) return
  exportingAnomalyPng.value = artifact.name
  let sourceUrl = ''
  let downloadUrl = ''
  try {
    const response = await fetch(artifactUrl(artifact.download_url))
    if (!response.ok) throw new Error(`Figure download failed with status ${response.status}`)
    const svgText = await response.text()
    sourceUrl = URL.createObjectURL(new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' }))
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const candidate = new Image()
      candidate.onload = () => resolve(candidate)
      candidate.onerror = () => reject(new Error('The SVG figure could not be rendered.'))
      candidate.src = sourceUrl
    })
    const scale = 2
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, image.naturalWidth) * scale
    canvas.height = Math.max(1, image.naturalHeight) * scale
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Canvas rendering is unavailable.')
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    const pngBlob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed.'))),
        'image/png'
      )
    })
    downloadUrl = URL.createObjectURL(pngBlob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = artifact.name.replace(/\.svg$/i, '.png')
    document.body.appendChild(link)
    link.click()
    link.remove()
  } catch (error) {
    console.error(error)
    ElMessage.error(
      t('Could not export the anomaly figure as PNG.', '无法将异常检测图导出为 PNG。')
    )
  } finally {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl)
    if (downloadUrl) URL.revokeObjectURL(downloadUrl)
    exportingAnomalyPng.value = ''
  }
}

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<template>
  <main class="data-mining-workbench">
    <section class="data-mining-page">
      <section class="page-heading">
        <div>
          <p class="eyebrow">GEOCHEMISTRY π ONLINE</p>
          <h1>{{ t('Data mining', '数据挖掘') }}</h1>
          <p class="intro">
            {{
              t(
                'Upload a dataset, inspect its structure and quality, then continue to preprocessing and modeling.',
                '上传数据集，检查其结构与质量，再继续进行预处理和建模。'
              )
            }}
          </p>
        </div>
        <div class="service-status" :class="serviceState">
          <span class="status-dot"></span>
          <span v-if="serviceState === 'checking'">{{
            t('Checking service', '正在检查服务')
          }}</span>
          <span v-else-if="serviceState === 'online'">{{ t('Backend online', '后端已连接') }}</span>
          <span v-else>{{ t('Backend offline', '后端离线') }}</span>
        </div>
      </section>

      <el-card v-loading="loadingCatalog" class="workflow-card" shadow="never">
        <el-form label-position="top">
          <div class="form-grid">
            <el-form-item :label="t('Function', '功能')">
              <el-select
                v-model="selectedFeature"
                :placeholder="t('Select a function', '选择功能')"
                :disabled="running"
              >
                <el-option
                  v-for="feature in features"
                  :key="feature.name"
                  :label="dataMiningFeatureDescription(feature.name, feature.description)"
                  :value="feature.name"
                >
                  <div class="feature-option">
                    <span>{{
                      dataMiningFeatureDescription(feature.name, feature.description)
                    }}</span>
                    <el-tag :type="statusType(feature.status)" size="small" effect="plain">
                      {{ statusLabel(feature.status) }}
                    </el-tag>
                  </div>
                </el-option>
              </el-select>
            </el-form-item>

            <el-form-item :label="t('Supported data', '支持的数据')">
              <div class="format-tags">
                <el-tag
                  v-for="format in currentFeature?.input_formats || []"
                  :key="format"
                  effect="plain"
                >
                  {{ format }}
                </el-tag>
              </div>
            </el-form-item>
          </div>

          <section v-if="currentFeature" class="feature-guide">
            <div class="feature-guide-heading">
              <div>
                <p class="guide-kicker">
                  {{ t('FUNCTION STATUS AND OUTPUT GUIDE', '功能状态与输出说明') }}
                </p>
                <h2>
                  {{
                    dataMiningFeatureDescription(currentFeature.name, currentFeature.description)
                  }}
                </h2>
              </div>
              <el-tag :type="statusType(currentFeature.status)" effect="dark">
                {{ statusLabel(currentFeature.status) }}
              </el-tag>
            </div>

            <p class="status-message">{{ apiText(currentFeature.status_message) }}</p>
            <div v-if="currentFeature.outputs.length" class="output-list">
              <strong>{{ t('Outputs', '当前输出') }}</strong>
              <el-tag
                v-for="output in currentFeature.outputs"
                :key="output"
                type="info"
                effect="plain"
              >
                {{ apiText(output) }}
              </el-tag>
            </div>

            <el-alert
              v-if="!currentFeatureIsVerified"
              :title="
                t(
                  'This function is still being integrated and will become available after input and result validation.',
                  '该功能仍在接入中，完成输入和结果验证后开放运行。'
                )
              "
              type="warning"
              :closable="false"
              show-icon
            />
          </section>

          <el-form-item :label="t('Dataset (.xlsx or .csv)', '数据集（.xlsx 或 .csv）')">
            <label class="file-picker" :class="{ disabled: running || !currentFeatureIsVerified }">
              <input
                type="file"
                accept=".xlsx,.csv"
                :disabled="running || !currentFeatureIsVerified"
                @change="onFileChange"
              />
              <el-icon class="upload-icon"><UploadFilled /></el-icon>
              <span class="file-copy">
                <strong class="file-button">{{ t('Choose dataset', '选择数据集') }}</strong>
                <small>{{
                  t(
                    'Drop an XLSX or CSV file here, or click to browse.',
                    '拖放 XLSX 或 CSV 文件到此处，或点击浏览。'
                  )
                }}</small>
                <small>{{ resourceLimitNote }}</small>
                <em class="file-name">{{
                  datasetFile?.name || t('No file selected', '未选择文件')
                }}</em>
              </span>
            </label>
          </el-form-item>

          <section v-if="isPreprocessing" v-loading="inspectingColumns" class="preprocessing-panel">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('PREPROCESSING CONFIGURATION', '预处理配置') }}</p>
                <h3>
                  {{
                    t(
                      'Select output columns and a missing-value rule',
                      '选择输出列和缺失值处理规则'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ columnInspection.columns.length }} {{ t('columns detected', '列已识别') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <el-form-item :label="t('Columns to keep', '要保留的列')">
                <div class="column-selection">
                  <div class="selection-tools">
                    <span>
                      {{ selectedColumns.length }} / {{ columnInspection.columns.length }}
                      {{ t('selected', '已选择') }}
                    </span>
                    <div>
                      <el-button link type="primary" :disabled="running" @click="selectAllColumns">
                        {{ t('Select all', '全选') }}
                      </el-button>
                      <el-button link :disabled="running" @click="clearSelectedColumns">
                        {{ t('Clear', '清空') }}
                      </el-button>
                    </div>
                  </div>
                  <el-select
                    v-model="selectedColumns"
                    multiple
                    filterable
                    collapse-tags
                    collapse-tags-tooltip
                    :placeholder="t('Select at least one column', '至少选择一列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in columnInspection.columns"
                      :key="column.name"
                      :label="column.name"
                      :value="column.name"
                    >
                      <div class="column-option">
                        <span>{{ column.name }}</span>
                        <small>
                          {{ column.data_type }} · {{ column.missing }} {{ t('missing', '缺失') }}
                        </small>
                      </div>
                    </el-option>
                  </el-select>
                </div>
              </el-form-item>

              <el-form-item :label="t('Missing-value handling', '缺失值处理')">
                <el-select v-model="missingStrategy" :disabled="running">
                  <el-option
                    v-for="option in missingStrategyOptions"
                    :key="option.value"
                    :label="option.label"
                    :value="option.value"
                  />
                </el-select>
                <p class="field-help">{{ selectedStrategy?.description }}</p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Its columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别其列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              v-if="timeSeriesMode !== 'element_mean'"
              :title="
                t(
                  'The uploaded source file is never modified. A new UTF-8 CSV file and a JSON processing record will be generated.',
                  '上传的源文件不会被修改。系统将生成新的 UTF-8 CSV 文件和 JSON 处理记录。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section v-if="isRegression" v-loading="inspectingColumns" class="preprocessing-panel">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('REGRESSION CONFIGURATION', '回归配置') }}</p>
                <h3>
                  {{
                    t(
                      'Choose a numeric target and one or more numeric features',
                      '选择一个数值目标列和一个或多个数值特征'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ numericColumns.length }} {{ t('numeric columns detected', '个数值列已识别') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <div class="supervised-mode-picker">
                <div>
                  <strong>{{ t('Training mode', '训练模式') }}</strong>
                  <span>
                    {{
                      t(
                        'Train one model or compare several models under identical cross-validation folds.',
                        '训练单个模型，或在完全相同的交叉验证分折下比较多个模型。'
                      )
                    }}
                  </span>
                </div>
                <el-radio-group v-model="regressionRunMode" :disabled="running">
                  <el-radio-button value="single">{{
                    t('Single model', '单模型')
                  }}</el-radio-button>
                  <el-radio-button value="compare">{{
                    t('Compare models', '多模型比较')
                  }}</el-radio-button>
                </el-radio-group>
              </div>

              <div class="form-grid">
                <el-form-item v-if="regressionRunMode === 'single'" :label="t('Model', '模型')">
                  <el-select v-model="regressionModel" :disabled="running">
                    <el-option
                      v-for="method in regressionMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">
                    {{
                      regressionMethodOptions.find((method) => method.name === regressionModel)
                        ?.description
                    }}
                  </p>
                </el-form-item>

                <el-form-item
                  v-if="regressionRunMode === 'single'"
                  :label="t('Test dataset size', '测试集比例')"
                >
                  <el-select v-model="regressionTestSize" :disabled="running">
                    <el-option :label="t('20% (recommended)', '20%（推荐）')" :value="0.2" />
                    <el-option label="25%" :value="0.25" />
                    <el-option label="30%" :value="0.3" />
                    <el-option label="40%" :value="0.4" />
                  </el-select>
                </el-form-item>

                <el-form-item
                  v-if="regressionRunMode === 'compare'"
                  :label="t('Models to compare', '要比较的模型')"
                  class="wide-form-item"
                >
                  <el-select
                    v-model="regressionComparisonModels"
                    multiple
                    filterable
                    collapse-tags
                    collapse-tags-tooltip
                    :max-collapse-tags="4"
                    :disabled="running"
                    :placeholder="t('Select at least two models', '至少选择两个模型')"
                  >
                    <el-option
                      v-for="method in regressionMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">
                    {{
                      t(
                        'Models are ranked by mean cross-validated R²; MAE and RMSE are reported as secondary metrics.',
                        '模型按交叉验证平均 R² 排名，同时报告 MAE 和 RMSE。'
                      )
                    }}
                  </p>
                </el-form-item>
              </div>

              <div
                v-if="regressionRunMode === 'single' && selectedRegressionMethod"
                class="model-settings"
              >
                <div class="model-settings-heading">
                  <div>
                    <strong>{{ t('Hyperparameters', '超参数') }}</strong>
                    <span>{{
                      t('Validated safe settings for this model.', '当前模型经过校验的安全设置。')
                    }}</span>
                  </div>
                  <el-tag effect="plain">{{ selectedRegressionMethod.display_name }}</el-tag>
                </div>
                <div class="hyperparameter-grid">
                  <label
                    v-for="parameter in selectedRegressionMethod.hyperparameters"
                    :key="parameter.name"
                    class="hyperparameter-field"
                  >
                    <span>{{ parameter.display_name }}</span>
                    <el-switch
                      v-if="parameter.value_type === 'boolean'"
                      :model-value="
                        Boolean(getParameterValue('regression', regressionModel, parameter))
                      "
                      :disabled="running"
                      @update:model-value="
                        setParameterValue('regression', regressionModel, parameter, $event)
                      "
                    />
                    <el-select
                      v-else-if="parameter.value_type === 'select'"
                      :model-value="getParameterValue('regression', regressionModel, parameter)"
                      :disabled="running"
                      @update:model-value="
                        setParameterValue('regression', regressionModel, parameter, $event)
                      "
                    >
                      <el-option
                        v-for="option in parameter.options"
                        :key="String(option)"
                        :label="String(option)"
                        :value="option"
                      />
                    </el-select>
                    <el-input-number
                      v-else
                      :model-value="
                        Number(getParameterValue('regression', regressionModel, parameter))
                      "
                      :min="parameter.minimum ?? undefined"
                      :max="parameter.maximum ?? undefined"
                      :step="parameter.step ?? 1"
                      :precision="parameter.value_type === 'integer' ? 0 : undefined"
                      :disabled="running"
                      controls-position="right"
                      @update:model-value="
                        setParameterValue('regression', regressionModel, parameter, Number($event))
                      "
                    />
                    <small>{{ parameter.description }}</small>
                  </label>
                </div>
              </div>

              <div class="cross-validation-setting">
                <div>
                  <strong>{{ t('Cross-validation', '交叉验证') }}</strong>
                  <span>
                    {{
                      regressionRunMode === 'compare'
                        ? t('Required for a fair model comparison.', '多模型公平比较必须启用。')
                        : t(
                            'Optional validation beyond the held-out test set.',
                            '在独立测试集之外进行可选验证。'
                          )
                    }}
                  </span>
                </div>
                <el-switch
                  v-if="regressionRunMode === 'single'"
                  v-model="regressionCrossValidationEnabled"
                  :disabled="running"
                />
                <label v-if="regressionRunMode === 'compare' || regressionCrossValidationEnabled">
                  <span>{{ t('Folds', '折数') }}</span>
                  <el-input-number
                    v-model="regressionCrossValidationFolds"
                    :min="2"
                    :max="10"
                    :step="1"
                    :precision="0"
                    :disabled="running"
                    controls-position="right"
                  />
                </label>
              </div>

              <div v-if="regressionRunMode === 'compare'" class="comparison-model-settings">
                <el-collapse>
                  <el-collapse-item
                    v-for="modelName in regressionComparisonModels"
                    :key="modelName"
                    :title="`${regressionMethodByName(modelName)?.display_name || modelName} · ${t('Hyperparameters', '超参数')}`"
                    :name="modelName"
                  >
                    <div class="hyperparameter-grid compact">
                      <label
                        v-for="parameter in regressionMethodByName(modelName)?.hyperparameters ||
                        []"
                        :key="parameter.name"
                        class="hyperparameter-field"
                      >
                        <span>{{ parameter.display_name }}</span>
                        <el-switch
                          v-if="parameter.value_type === 'boolean'"
                          :model-value="
                            Boolean(getParameterValue('regression', modelName, parameter))
                          "
                          :disabled="running"
                          @update:model-value="
                            setParameterValue('regression', modelName, parameter, $event)
                          "
                        />
                        <el-select
                          v-else-if="parameter.value_type === 'select'"
                          :model-value="getParameterValue('regression', modelName, parameter)"
                          :disabled="running"
                          @update:model-value="
                            setParameterValue('regression', modelName, parameter, $event)
                          "
                        >
                          <el-option
                            v-for="option in parameter.options"
                            :key="String(option)"
                            :label="String(option)"
                            :value="option"
                          />
                        </el-select>
                        <el-input-number
                          v-else
                          :model-value="
                            Number(getParameterValue('regression', modelName, parameter))
                          "
                          :min="parameter.minimum ?? undefined"
                          :max="parameter.maximum ?? undefined"
                          :step="parameter.step ?? 1"
                          :precision="parameter.value_type === 'integer' ? 0 : undefined"
                          :disabled="running"
                          controls-position="right"
                          @update:model-value="
                            setParameterValue('regression', modelName, parameter, Number($event))
                          "
                        />
                        <small>{{ parameter.description }}</small>
                      </label>
                    </div>
                  </el-collapse-item>
                </el-collapse>
              </div>

              <el-form-item :label="t('Target column', '目标列')">
                <el-select
                  v-model="regressionTarget"
                  filterable
                  :placeholder="t('Select the value to predict', '选择要预测的值')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in numericColumns"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'The target is the numeric value the model will learn to predict.',
                      '目标列是模型要学习预测的数值。'
                    )
                  }}
                </p>
              </el-form-item>

              <el-form-item :label="t('Feature columns', '特征列')">
                <el-select
                  v-model="regressionFeatures"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  :placeholder="t('Select at least one numeric feature', '至少选择一个数值特征')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in regressionFeatureOptions"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'The target column is automatically excluded from the feature list.',
                      '目标列会自动从特征列表中排除。'
                    )
                  }}
                </p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Numeric columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别数值列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'Rows containing missing or infinite values in the selected columns are removed before training. At least 10 complete rows are required. The split uses random state 42 for reproducibility.',
                  '训练前会删除已选列中含缺失值或无穷值的行。至少需要 10 行完整数据。数据划分使用随机种子 42 以保证可复现性。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section
            v-if="isClassification"
            v-loading="inspectingColumns"
            class="preprocessing-panel"
          >
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CLASSIFICATION CONFIGURATION', '分类配置') }}</p>
                <h3>
                  {{
                    t(
                      'Choose a label column and one or more numeric features',
                      '选择标签列和一个或多个数值特征'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ numericColumns.length }} {{ t('numeric features available', '个数值特征可用') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <div class="supervised-mode-picker">
                <div>
                  <strong>{{ t('Training mode', '训练模式') }}</strong>
                  <span>
                    {{
                      t(
                        'Train one model or compare several models under identical stratified folds.',
                        '训练单个模型，或在完全相同的分层交叉验证分折下比较多个模型。'
                      )
                    }}
                  </span>
                </div>
                <el-radio-group v-model="classificationRunMode" :disabled="running">
                  <el-radio-button value="single">{{
                    t('Single model', '单模型')
                  }}</el-radio-button>
                  <el-radio-button value="compare">{{
                    t('Compare models', '多模型比较')
                  }}</el-radio-button>
                </el-radio-group>
              </div>

              <div class="form-grid">
                <el-form-item v-if="classificationRunMode === 'single'" :label="t('Model', '模型')">
                  <el-select v-model="classificationModel" :disabled="running">
                    <el-option
                      v-for="method in classificationMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">
                    {{
                      classificationMethodOptions.find(
                        (method) => method.name === classificationModel
                      )?.description
                    }}
                  </p>
                </el-form-item>

                <el-form-item
                  v-if="classificationRunMode === 'single'"
                  :label="t('Test dataset size', '测试集比例')"
                >
                  <el-select v-model="classificationTestSize" :disabled="running">
                    <el-option :label="t('20% (recommended)', '20%（推荐）')" :value="0.2" />
                    <el-option label="25%" :value="0.25" />
                    <el-option label="30%" :value="0.3" />
                    <el-option label="40%" :value="0.4" />
                  </el-select>
                </el-form-item>

                <el-form-item
                  v-if="classificationRunMode === 'compare'"
                  :label="t('Models to compare', '要比较的模型')"
                  class="wide-form-item"
                >
                  <el-select
                    v-model="classificationComparisonModels"
                    multiple
                    filterable
                    collapse-tags
                    collapse-tags-tooltip
                    :max-collapse-tags="4"
                    :disabled="running"
                    :placeholder="t('Select at least two models', '至少选择两个模型')"
                  >
                    <el-option
                      v-for="method in classificationMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">
                    {{
                      t(
                        'Models are ranked by mean cross-validated Macro F1; accuracy, precision and recall remain visible.',
                        '模型按交叉验证平均 Macro F1 排名，同时保留准确率、精确率和召回率。'
                      )
                    }}
                  </p>
                </el-form-item>
              </div>

              <div
                v-if="classificationRunMode === 'single' && selectedClassificationMethod"
                class="model-settings"
              >
                <div class="model-settings-heading">
                  <div>
                    <strong>{{ t('Hyperparameters', '超参数') }}</strong>
                    <span>{{
                      t('Validated safe settings for this model.', '当前模型经过校验的安全设置。')
                    }}</span>
                  </div>
                  <el-tag effect="plain">{{ selectedClassificationMethod.display_name }}</el-tag>
                </div>
                <div class="hyperparameter-grid">
                  <label
                    v-for="parameter in selectedClassificationMethod.hyperparameters"
                    :key="parameter.name"
                    class="hyperparameter-field"
                  >
                    <span>{{ parameter.display_name }}</span>
                    <el-switch
                      v-if="parameter.value_type === 'boolean'"
                      :model-value="
                        Boolean(getParameterValue('classification', classificationModel, parameter))
                      "
                      :disabled="running"
                      @update:model-value="
                        setParameterValue('classification', classificationModel, parameter, $event)
                      "
                    />
                    <el-select
                      v-else-if="parameter.value_type === 'select'"
                      :model-value="
                        getParameterValue('classification', classificationModel, parameter)
                      "
                      :disabled="running"
                      @update:model-value="
                        setParameterValue('classification', classificationModel, parameter, $event)
                      "
                    >
                      <el-option
                        v-for="option in parameter.options"
                        :key="String(option)"
                        :label="String(option)"
                        :value="option"
                      />
                    </el-select>
                    <el-input-number
                      v-else
                      :model-value="
                        Number(getParameterValue('classification', classificationModel, parameter))
                      "
                      :min="parameter.minimum ?? undefined"
                      :max="parameter.maximum ?? undefined"
                      :step="parameter.step ?? 1"
                      :precision="parameter.value_type === 'integer' ? 0 : undefined"
                      :disabled="running"
                      controls-position="right"
                      @update:model-value="
                        setParameterValue(
                          'classification',
                          classificationModel,
                          parameter,
                          Number($event)
                        )
                      "
                    />
                    <small>{{ parameter.description }}</small>
                  </label>
                </div>
              </div>

              <div class="cross-validation-setting">
                <div>
                  <strong>{{ t('Stratified cross-validation', '分层交叉验证') }}</strong>
                  <span>
                    {{
                      classificationRunMode === 'compare'
                        ? t('Required for a fair model comparison.', '多模型公平比较必须启用。')
                        : t(
                            'Optional validation that preserves class proportions.',
                            '保持类别比例的可选验证。'
                          )
                    }}
                  </span>
                </div>
                <el-switch
                  v-if="classificationRunMode === 'single'"
                  v-model="classificationCrossValidationEnabled"
                  :disabled="running"
                />
                <label
                  v-if="classificationRunMode === 'compare' || classificationCrossValidationEnabled"
                >
                  <span>{{ t('Folds', '折数') }}</span>
                  <el-input-number
                    v-model="classificationCrossValidationFolds"
                    :min="2"
                    :max="10"
                    :step="1"
                    :precision="0"
                    :disabled="running"
                    controls-position="right"
                  />
                </label>
              </div>

              <div v-if="classificationRunMode === 'compare'" class="comparison-model-settings">
                <el-collapse>
                  <el-collapse-item
                    v-for="modelName in classificationComparisonModels"
                    :key="modelName"
                    :title="`${classificationMethodByName(modelName)?.display_name || modelName} · ${t('Hyperparameters', '超参数')}`"
                    :name="modelName"
                  >
                    <div class="hyperparameter-grid compact">
                      <label
                        v-for="parameter in classificationMethodByName(modelName)
                          ?.hyperparameters || []"
                        :key="parameter.name"
                        class="hyperparameter-field"
                      >
                        <span>{{ parameter.display_name }}</span>
                        <el-switch
                          v-if="parameter.value_type === 'boolean'"
                          :model-value="
                            Boolean(getParameterValue('classification', modelName, parameter))
                          "
                          :disabled="running"
                          @update:model-value="
                            setParameterValue('classification', modelName, parameter, $event)
                          "
                        />
                        <el-select
                          v-else-if="parameter.value_type === 'select'"
                          :model-value="getParameterValue('classification', modelName, parameter)"
                          :disabled="running"
                          @update:model-value="
                            setParameterValue('classification', modelName, parameter, $event)
                          "
                        >
                          <el-option
                            v-for="option in parameter.options"
                            :key="String(option)"
                            :label="String(option)"
                            :value="option"
                          />
                        </el-select>
                        <el-input-number
                          v-else
                          :model-value="
                            Number(getParameterValue('classification', modelName, parameter))
                          "
                          :min="parameter.minimum ?? undefined"
                          :max="parameter.maximum ?? undefined"
                          :step="parameter.step ?? 1"
                          :precision="parameter.value_type === 'integer' ? 0 : undefined"
                          :disabled="running"
                          controls-position="right"
                          @update:model-value="
                            setParameterValue(
                              'classification',
                              modelName,
                              parameter,
                              Number($event)
                            )
                          "
                        />
                        <small>{{ parameter.description }}</small>
                      </label>
                    </div>
                  </el-collapse-item>
                </el-collapse>
              </div>

              <el-form-item :label="t('Target class column', '目标类别列')">
                <el-select
                  v-model="classificationTarget"
                  filterable
                  :placeholder="t('Select the label to predict', '选择要预测的标签')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in classificationTargetColumns"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'Text and numeric class labels are supported; missing labels are removed.',
                      '支持文本和数值类别标签；缺失标签会被删除。'
                    )
                  }}
                </p>
              </el-form-item>

              <el-form-item :label="t('Numeric feature columns', '数值特征列')">
                <el-select
                  v-model="classificationFeatures"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  :placeholder="t('Select at least one numeric feature', '至少选择一个数值特征')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in classificationFeatureOptions"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'Features are standardized automatically before the classifier is fitted.',
                      '拟合分类器前会自动标准化特征。'
                    )
                  }}
                </p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Label and numeric feature columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别标签列和数值特征列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'Incomplete rows are removed before training. At least 12 complete rows and two rows per class are required. The split is stratified and uses random state 42.',
                  '训练前会删除不完整行。至少需要 12 行完整数据，且每类至少有 2 行。数据按类别分层划分，随机种子为 42。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section v-if="isClustering" v-loading="inspectingColumns" class="preprocessing-panel">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CLUSTERING CONFIGURATION', '聚类配置') }}</p>
                <h3>
                  {{
                    t('Choose numeric features and a clustering method', '选择数值特征和聚类方法')
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ numericColumns.length }} {{ t('numeric columns detected', '个数值列已识别') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <div class="form-grid">
                <el-form-item :label="t('Model', '模型')">
                  <el-select v-model="clusteringModel" :disabled="running">
                    <el-option
                      v-for="method in clusteringMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">{{ selectedClusteringMethod?.description }}</p>
                </el-form-item>

                <el-form-item :label="t('Number of clusters', '簇数')">
                  <el-select
                    v-if="clusteringUsesClusterCount"
                    v-model="clusterCount"
                    :disabled="running"
                  >
                    <el-option
                      v-for="count in 9"
                      :key="count + 1"
                      :label="String(count + 1)"
                      :value="count + 1"
                    />
                  </el-select>
                  <el-input
                    v-else
                    :model-value="t('Estimated automatically', '由算法自动估计')"
                    disabled
                  />
                </el-form-item>
              </div>

              <el-form-item :label="t('Numeric feature columns', '数值特征列')">
                <el-select
                  v-model="clusteringFeatures"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  :placeholder="t('Select at least one numeric feature', '至少选择一个数值特征')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in numericColumns"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'Features are standardized for fitting; reported cluster centers use the original units.',
                      '特征会先标准化再拟合；报告的聚类中心使用原始单位。'
                    )
                  }}
                </p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Numeric feature columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别数值特征列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'Incomplete rows are removed before clustering. At least max(10, 2 × cluster count) complete rows are required. K-means uses random state 42.',
                  '聚类前会删除不完整行。至少需要 max(10, 2 × 簇数) 行完整数据。K-means 使用随机种子 42。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section
            v-if="isDimensionalityReduction"
            v-loading="inspectingColumns"
            class="preprocessing-panel"
          >
            <div class="section-heading">
              <div>
                <p class="guide-kicker">
                  {{ t('DIMENSIONALITY REDUCTION CONFIGURATION', '降维配置') }}
                </p>
                <h3>
                  {{
                    t(
                      'Choose numeric features, an algorithm and output dimensions',
                      '选择数值特征、降维算法和输出维度'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ numericColumns.length }} {{ t('numeric columns detected', '个数值列已识别') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <div class="form-grid">
                <el-form-item :label="t('Model', '模型')">
                  <el-select v-model="dimensionalityReductionModel" :disabled="running">
                    <el-option
                      v-for="method in dimensionalityReductionMethodOptions"
                      :key="method.name"
                      :label="method.display_name"
                      :value="method.name"
                    />
                  </el-select>
                  <p class="field-help">{{ selectedDimensionalityReductionMethod?.description }}</p>
                </el-form-item>

                <el-form-item :label="t('Output dimensions', '输出维度')">
                  <el-select v-model="componentCount" :disabled="running">
                    <el-option :label="t('2 dimensions', '二维')" :value="2" />
                    <el-option :label="t('3 dimensions', '三维')" :value="3" />
                  </el-select>
                </el-form-item>
              </div>

              <el-form-item :label="t('Numeric feature columns', '数值特征列')">
                <el-select
                  v-model="dimensionalityReductionFeatures"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  :placeholder="t('Select at least two numeric features', '至少选择两个数值特征')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in numericColumns"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'Features are standardized automatically. PCA also reports explained variance.',
                      '特征会自动标准化；PCA 还会报告解释方差。'
                    )
                  }}
                </p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Numeric feature columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别数值特征列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'Incomplete rows are removed. T-SNE is limited to 5,000 complete rows and MDS to 2,000 rows to protect the local service.',
                  '不完整行会被删除。为保护本地服务，T-SNE 最多处理 5,000 行完整数据，MDS 最多处理 2,000 行。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section
            v-if="isAnomalyDetection"
            v-loading="inspectingColumns"
            class="preprocessing-panel"
          >
            <div class="section-heading">
              <div>
                <p class="guide-kicker">
                  {{ t('ANOMALY DETECTION CONFIGURATION', '异常检测配置') }}
                </p>
                <h3>
                  {{
                    t(
                      'Choose numeric features and an anomaly detector',
                      '选择数值特征和异常检测算法'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ numericColumns.length }} {{ t('numeric columns detected', '个数值列已识别') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <el-form-item
                :label="t('Visualization / reproduction template', '可视化 / 复现模板')"
              >
                <el-select v-model="anomalyReproductionProfile" :disabled="running">
                  <el-option :label="t('General diagnostics', '通用诊断')" value="general" />
                  <el-option
                    label="Sharapatov et al. (2025), Figure 3a"
                    value="sharapatov_2025_figure_3a"
                  />
                  <el-option label="Zhu et al. (2024), Figure 8a" value="zhu_2024_figure_8a" />
                </el-select>
                <p class="field-help">{{ selectedAnomalyReproductionDescription }}</p>
              </el-form-item>

              <el-form-item :label="t('Model', '模型')">
                <el-select
                  v-model="anomalyDetectionModel"
                  :disabled="running || anomalyProfileIsLocked"
                >
                  <el-option
                    v-for="method in anomalyDetectionMethodOptions"
                    :key="method.name"
                    :label="method.display_name"
                    :value="method.name"
                  />
                </el-select>
                <p class="field-help">{{ selectedAnomalyDetectionMethod?.description }}</p>
              </el-form-item>

              <el-form-item :label="t('Contamination threshold', '异常比例阈值')">
                <div class="anomaly-contamination-control">
                  <el-switch
                    v-model="anomalyContaminationAutomatic"
                    :disabled="running || anomalyProfileIsLocked"
                    :aria-label="t('Use automatic contamination', '使用自动异常比例')"
                  />
                  <strong>
                    {{
                      anomalyContaminationAutomatic
                        ? t('Automatic', '自动')
                        : t('Fixed proportion', '固定比例')
                    }}
                  </strong>
                  <el-input-number
                    v-if="!anomalyContaminationAutomatic"
                    v-model="anomalyContamination"
                    class="anomaly-contamination-input"
                    :min="0.001"
                    :max="0.5"
                    :step="0.01"
                    :precision="3"
                    :disabled="running || anomalyProfileIsLocked"
                    controls-position="right"
                  />
                </div>
                <p class="field-help">
                  {{
                    anomalyProfileIsLocked
                      ? t(
                          'The selected paper template fixes the detector and target anomaly proportion.',
                          '所选论文模板固定了检测算法和目标异常比例。'
                        )
                      : t(
                          'Automatic uses the algorithm default; a fixed proportion must be between 0.001 and 0.5.',
                          '自动模式采用算法默认值；固定比例必须介于 0.001 和 0.5 之间。'
                        )
                  }}
                </p>
              </el-form-item>

              <el-form-item :label="t('Numeric feature columns', '数值特征列')">
                <el-select
                  v-model="anomalyDetectionFeatures"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  :placeholder="t('Select at least one numeric feature', '至少选择一个数值特征')"
                  :disabled="running"
                >
                  <el-option
                    v-for="column in numericColumns"
                    :key="column"
                    :label="column"
                    :value="column"
                  />
                </el-select>
                <p class="field-help">
                  {{
                    t(
                      'Features are standardized automatically; higher reported scores indicate stronger anomalies.',
                      '特征会自动标准化；报告的分数越大，表示异常程度越高。'
                    )
                  }}
                </p>
              </el-form-item>
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Numeric feature columns will be detected automatically.',
                  '请先选择数据集，系统将自动识别数值特征列。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'At least 10 complete rows are required. Paper templates add a separate reference-reproduction artifact when the uploaded workbook contains the required audit data.',
                  '至少需要 10 行完整数据；当上传的工作簿包含所需审计数据时，论文模板会额外生成独立的参考复现图件。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <section v-if="isTimeSeries" v-loading="inspectingColumns" class="preprocessing-panel">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">
                  {{ t('TIME SERIES CONFIGURATION', '时间序列配置') }}
                </p>
                <h3>
                  {{
                    t(
                      'Map geological variables and configure age-bin uncertainty analysis',
                      '映射地质变量并配置年龄分箱不确定度分析'
                    )
                  }}
                </h3>
              </div>
              <el-tag v-if="columnInspection" type="success" effect="plain">
                {{ timeSeriesMappedColumns.filter(Boolean).length }}/{{
                  timeSeriesRequiredColumnCount
                }}
                {{ t('variables mapped', '个变量已映射') }}
              </el-tag>
            </div>

            <template v-if="columnInspection">
              <div class="time-series-mode-picker">
                <span>{{ t('Analysis mode', '分析模式') }}</span>
                <el-radio-group v-model="timeSeriesMode" :disabled="running">
                  <el-radio-button value="element_mean">
                    {{ t('Element mean', '元素均值') }}
                  </el-radio-button>
                  <el-radio-button value="direct">
                    {{ t('Uploaded probability column', '使用上传的概率列') }}
                  </el-radio-button>
                  <el-radio-button value="model_predicted">
                    {{ t('Predict from geochemistry', '根据地球化学数据预测') }}
                  </el-radio-button>
                </el-radio-group>
              </div>

              <el-alert
                v-if="timeSeriesMode === 'model_predicted'"
                type="warning"
                :closable="false"
                show-icon
                :title="
                  t(
                    'Liu-2024 surrogate v1 predicts probability before Time Series. It is a reproducible surrogate of published probabilities, not the authors’ original trained model.',
                    'Liu-2024 替代模型 v1 会先预测概率再运行时间序列；它是对已发布概率的可复现替代模型，并非论文作者的原始训练模型。'
                  )
                "
              />

              <div class="time-series-mapping-grid">
                <el-form-item :label="t('Central age', '中心年龄')">
                  <el-select
                    v-model="timeSeriesAgeColumn"
                    filterable
                    :placeholder="t('Select age column', '选择年龄列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">R_AGE</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode !== 'element_mean'"
                  :label="t('Maximum age', '最大年龄')"
                >
                  <el-select
                    v-model="timeSeriesAgeMaxColumn"
                    filterable
                    :placeholder="t('Select maximum-age column', '选择最大年龄列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">R_MAX_AGE</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode === 'element_mean'"
                  :label="t('Target element', '目标元素')"
                >
                  <el-select
                    v-model="timeSeriesValueColumn"
                    filterable
                    :placeholder="t('Select numeric element column', '选择数值元素列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">MGO</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode === 'element_mean'"
                  :label="t('Optional filter column', '可选筛选列')"
                >
                  <el-select
                    v-model="timeSeriesFilterColumn"
                    filterable
                    clearable
                    :placeholder="t('No composition filter', '不筛选成分')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">Keller Figure 1: SIO2 43–51 wt%</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode === 'direct'"
                  :label="t('Subaerial probability', '陆上玄武岩概率')"
                >
                  <el-select
                    v-model="timeSeriesProbabilityColumn"
                    filterable
                    :placeholder="t('Select probability column', '选择概率列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">0–1 · SBAP</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode !== 'element_mean'"
                  :label="t('Latitude', '纬度')"
                >
                  <el-select
                    v-model="timeSeriesLatitudeColumn"
                    filterable
                    :placeholder="t('Select latitude column', '选择纬度列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">−90°–90°</p>
                </el-form-item>

                <el-form-item
                  v-if="timeSeriesMode !== 'element_mean'"
                  :label="t('Longitude', '经度')"
                >
                  <el-select
                    v-model="timeSeriesLongitudeColumn"
                    filterable
                    :placeholder="t('Select longitude column', '选择经度列')"
                    :disabled="running"
                  >
                    <el-option
                      v-for="column in numericColumns"
                      :key="column"
                      :label="column"
                      :value="column"
                    />
                  </el-select>
                  <p class="field-help">−180°–180°</p>
                </el-form-item>
              </div>

              <div class="time-series-parameter-grid">
                <el-form-item :label="t('Age unit', '年龄单位')">
                  <el-select v-model="timeSeriesAgeUnit" :disabled="running">
                    <el-option label="Ma" value="Ma" />
                    <el-option label="Ga" value="Ga" />
                  </el-select>
                </el-form-item>
                <el-form-item :label="`${t('Bin width', '分箱宽度')} (${timeSeriesAgeUnit})`">
                  <el-input-number
                    v-model="timeSeriesBinWidth"
                    :min="0.000001"
                    :precision="timeSeriesAgeUnit === 'Ga' ? 3 : 1"
                    :step="timeSeriesAgeUnit === 'Ga' ? 0.01 : 10"
                    :disabled="running"
                    controls-position="right"
                  />
                </el-form-item>
                <el-form-item
                  v-if="timeSeriesMode === 'element_mean'"
                  :label="t('Value unit', '数值单位')"
                >
                  <el-input v-model="timeSeriesValueUnit" :disabled="running" />
                </el-form-item>
                <el-form-item
                  v-if="timeSeriesMode === 'element_mean' && timeSeriesFilterColumn"
                  :label="t('Filter range', '筛选范围')"
                >
                  <div class="filter-range-inputs">
                    <el-input-number
                      v-model="timeSeriesFilterMinimum"
                      :disabled="running"
                      controls-position="right"
                    />
                    <span>–</span>
                    <el-input-number
                      v-model="timeSeriesFilterMaximum"
                      :disabled="running"
                      controls-position="right"
                    />
                  </div>
                </el-form-item>
                <el-form-item
                  v-if="timeSeriesMode !== 'element_mean'"
                  :label="t('Bootstrap iterations', 'Bootstrap 次数')"
                >
                  <el-input-number
                    v-model="timeSeriesBootstrapIterations"
                    :min="10"
                    :max="1000"
                    :step="10"
                    :disabled="running"
                    controls-position="right"
                  />
                </el-form-item>
              </div>
              <el-alert
                v-if="timeSeriesMode === 'element_mean'"
                type="info"
                :closable="false"
                show-icon
                :title="
                  t(
                    'Each age bin reports the unweighted arithmetic mean and ±2 SEM. Use 100 Ma bins and SIO2 = 43–51 wt% for a basic comparison with Keller Figure 1.',
                    '每个年龄箱输出未加权算术平均值和 ±2 SEM。与 Keller Figure 1 基础对比时使用100 Ma分箱，并筛选 SIO2 = 43–51 wt%。'
                  )
                "
              />
            </template>

            <el-alert
              v-else-if="!datasetFile"
              :title="
                t(
                  'Choose a dataset first. Standard v0.8 column names will be mapped automatically when available.',
                  '请先选择数据集；若存在 v0.8 标准列名，系统将自动完成映射。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              :title="
                t(
                  'This workflow estimates binned subaerial-basalt proportions with bootstrap uncertainty. It is not a forecasting model.',
                  '此工作流通过 Bootstrap 不确定度估计分箱后的陆上玄武岩比例，并非预测模型。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />
          </section>

          <el-alert
            v-if="errorMessage"
            class="message-block"
            :title="errorMessage"
            type="error"
            :closable="false"
            show-icon
          />

          <div class="actions">
            <el-button
              type="primary"
              size="large"
              :loading="running"
              :disabled="!canRun"
              @click="submitJob"
            >
              {{ runButtonLabel }}
            </el-button>
            <el-button v-if="serviceState === 'offline'" size="large" @click="loadPage">
              {{ t('Retry connection', '重新连接') }}
            </el-button>
          </div>
          <TaskProgress
            v-if="taskStatus && (running || inspectingColumns)"
            :task="taskStatus"
            :cancelling="cancellingTask"
            @cancel="cancelCurrentTask"
          />
        </el-form>
      </el-card>

      <template v-if="result">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Rows', '行数') }}</span>
            <strong>{{ formatNumber(result.summary.rows) }}</strong>
          </article>
          <article class="summary-card">
            <span>{{ t('Columns', '列数') }}</span>
            <strong>{{ formatNumber(result.summary.columns) }}</strong>
          </article>
          <article class="summary-card" :class="{ warning: result.summary.missing_cells > 0 }">
            <span>{{ t('Missing cells', '缺失单元格') }}</span>
            <strong>{{ formatPercent(result.summary.missing_rate) }}</strong>
            <small
              >{{ formatNumber(result.summary.missing_cells) }} {{ t('cells', '个单元格') }}</small
            >
          </article>
          <article class="summary-card" :class="{ warning: result.summary.duplicate_rows > 0 }">
            <span>{{ t('Duplicate rows', '重复行') }}</span>
            <strong>{{ formatNumber(result.summary.duplicate_rows) }}</strong>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>{{ t('Dataset profile completed', '数据集概览完成') }}</h2>
                <p>
                  {{ result.source_filename }} · {{ t('Job ID', '任务 ID') }}: {{ result.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="warning-list">
            <el-alert
              v-for="warning in result.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('COLUMN PROFILE', '列概览') }}</p>
                <h3>{{ t('Column quality and descriptive statistics', '列质量与描述性统计') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="result.columns" border size="small">
                <el-table-column prop="name" :label="t('Column', '列')" min-width="140" fixed />
                <el-table-column prop="data_type" :label="t('Type', '类型')" min-width="95" />
                <el-table-column prop="non_null" :label="t('Non-null', '非空值')" min-width="90" />
                <el-table-column prop="missing" :label="t('Missing', '缺失')" min-width="85" />
                <el-table-column :label="t('Missing rate', '缺失率')" min-width="105">
                  <template #default="scope">{{ formatPercent(scope.row.missing_rate) }}</template>
                </el-table-column>
                <el-table-column prop="unique" :label="t('Unique', '唯一值')" min-width="85" />
                <el-table-column :label="t('Minimum', '最小值')" min-width="105">
                  <template #default="scope">{{ formatNumber(scope.row.minimum) }}</template>
                </el-table-column>
                <el-table-column :label="t('Maximum', '最大值')" min-width="105">
                  <template #default="scope">{{ formatNumber(scope.row.maximum) }}</template>
                </el-table-column>
                <el-table-column :label="t('Mean', '均值')" min-width="105">
                  <template #default="scope">{{ formatNumber(scope.row.mean) }}</template>
                </el-table-column>
                <el-table-column :label="t('Std. dev.', '标准差')" min-width="105">
                  <template #default="scope">{{
                    formatNumber(scope.row.standard_deviation)
                  }}</template>
                </el-table-column>
                <el-table-column :label="t('Sample values', '示例值')" min-width="200">
                  <template #default="scope">
                    {{ scope.row.sample_values.map(formatCell).join(' · ') || '—' }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="result.columns" />
          </section>

          <section v-if="result.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('DATA PREVIEW', '数据预览') }}</p>
                <h3>{{ t('First', '前') }} {{ result.preview.length }} {{ t('rows', '行') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="result.preview" border size="small">
                <el-table-column
                  v-for="column in previewColumns"
                  :key="column"
                  :prop="column"
                  :label="column"
                  min-width="130"
                >
                  <template #default="scope">{{ formatCell(scope.row[column]) }}</template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="result.preview" />
          </section>

          <div
            v-for="artifact in result.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{ t('Download JSON report', '下载 JSON 报告') }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="preprocessingResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Rows', '行数') }}</span>
            <strong>{{ formatNumber(preprocessingResult.summary.processed_rows) }}</strong>
            <small>
              {{ formatNumber(preprocessingResult.summary.original_rows) }}
              {{ t('original', '原始') }} ·
              {{ formatNumber(preprocessingResult.summary.removed_rows) }}
              {{ t('removed', '已删除') }}
            </small>
          </article>
          <article class="summary-card">
            <span>{{ t('Columns', '列数') }}</span>
            <strong>{{ formatNumber(preprocessingResult.summary.processed_columns) }}</strong>
            <small>
              {{ formatNumber(preprocessingResult.summary.original_columns) }}
              {{ t('original', '原始') }} ·
              {{ formatNumber(preprocessingResult.summary.removed_columns) }}
              {{ t('removed', '已删除') }}
            </small>
          </article>
          <article
            class="summary-card"
            :class="{ warning: preprocessingResult.summary.processed_missing_cells > 0 }"
          >
            <span>{{ t('Remaining missing cells', '剩余缺失单元格') }}</span>
            <strong>{{ formatNumber(preprocessingResult.summary.processed_missing_cells) }}</strong>
            <small>
              {{ formatNumber(preprocessingResult.summary.original_missing_cells) }}
              {{ t('before processing', '处理前') }}
            </small>
          </article>
          <article class="summary-card">
            <span>{{ t('Filled cells', '已填充单元格') }}</span>
            <strong>{{ formatNumber(preprocessingResult.summary.filled_cells) }}</strong>
            <small>{{ selectedStrategy?.label }}</small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>{{ t('Data preprocessing completed', '数据预处理完成') }}</h2>
                <p>
                  {{ preprocessingResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ preprocessingResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta">
            <div>
              <span>{{ t('Missing-value rule', '缺失值规则') }}</span>
              <strong>{{
                selectedStrategy?.label || formatLabel(preprocessingResult.missing_strategy)
              }}</strong>
            </div>
            <div>
              <span>{{ t('Selected columns', '已选列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in preprocessingResult.selected_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in preprocessingResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section v-if="preprocessingResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('PROCESSED DATA PREVIEW', '处理后数据预览') }}</p>
                <h3>
                  {{ t('First', '前') }} {{ preprocessingResult.preview.length }}
                  {{ t('rows', '行') }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="preprocessingResult.preview" border size="small">
                <el-table-column
                  v-for="column in preprocessingPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="column"
                  min-width="130"
                >
                  <template #default="scope">{{ formatCell(scope.row[column]) }}</template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="preprocessingResult.preview" />
          </section>

          <div
            v-for="artifact in preprocessingResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download processed CSV', '下载处理后 CSV')
                  : t('Download processing record', '下载处理记录')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="regressionResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>R²</span>
            <strong>{{ formatNumber(regressionResult.metrics.r2, 6) }}</strong>
            <small>{{ t('Test dataset', '测试数据集') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Mean absolute error', '平均绝对误差') }}</span>
            <strong>
              {{ formatNumber(regressionResult.metrics.mean_absolute_error, 6) }}
            </strong>
            <small>{{ t('Lower is better', '越低越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Root mean squared error', '均方根误差') }}</span>
            <strong>
              {{ formatNumber(regressionResult.metrics.root_mean_squared_error, 6) }}
            </strong>
            <small>{{ t('Lower is better', '越低越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Train / test rows', '训练 / 测试行数') }}</span>
            <strong>
              {{ regressionResult.summary.train_rows }} /
              {{ regressionResult.summary.test_rows }}
            </strong>
            <small>
              {{ regressionResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
            </small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>{{ regressionResult.model_display_name }} {{ t('completed', '已完成') }}</h2>
                <p>
                  {{ regressionResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ regressionResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Target column', '目标列') }}</span>
              <strong>{{ regressionResult.target_column }}</strong>
            </div>
            <div>
              <span>{{ t('Feature columns', '特征列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in regressionResult.feature_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
            <div>
              <span>{{ t('Reproducible split', '可复现划分') }}</span>
              <strong>
                {{ formatPercent(regressionResult.test_size) }} {{ t('test', '测试集') }} ·
                {{ t('random state', '随机种子') }}
                {{ regressionResult.random_state }}
              </strong>
            </div>
          </div>

          <div v-if="regressionResult.equation" class="equation-card">
            <span>{{ t('Fitted equation', '拟合方程') }}</span>
            <code>{{ regressionResult.equation }}</code>
          </div>
          <div v-else class="equation-card">
            <span>{{ t('Model interpretation', '模型解释') }}</span>
            <code>
              {{
                t(
                  'This estimator does not expose one global equation in the original feature units.',
                  '该模型不提供一个以原始特征单位表示的全局方程。'
                )
              }}
            </code>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in regressionResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section v-if="regressionResult.cross_validation" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">CROSS-VALIDATION</p>
                <h3>
                  {{ regressionResult.cross_validation.folds }}-{{
                    t('fold validation stability', '折验证稳定性')
                  }}
                </h3>
              </div>
              <el-tag type="info" effect="plain">
                {{ regressionResult.cross_validation.strategy }} · {{ t('seed', '种子') }}
                {{ regressionResult.cross_validation.random_state }}
              </el-tag>
            </div>
            <div class="validation-metric-grid">
              <article
                v-for="metric in regressionResult.cross_validation.metrics"
                :key="metric.name"
              >
                <span>{{ metric.display_name }}</span>
                <strong>{{ formatNumber(metric.mean, 6) }}</strong>
                <small>± {{ formatNumber(metric.standard_deviation, 6) }}</small>
              </article>
            </div>
          </section>

          <section v-if="regressionResult.coefficients.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('MODEL COEFFICIENTS', '模型系数') }}</p>
                <h3>
                  {{
                    t('Feature effects in the fitted linear model', '特征在拟合线性模型中的影响')
                  }}
                </h3>
              </div>
              <el-tag type="info" effect="plain">
                {{ t('Intercept', '截距') }}: {{ formatNumber(regressionResult.intercept, 6) }}
              </el-tag>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="regressionResult.coefficients" border size="small">
                <el-table-column prop="feature" :label="t('Feature', '特征')" min-width="180" />
                <el-table-column :label="t('Coefficient', '系数')" min-width="160">
                  <template #default="scope">
                    {{ formatNumber(scope.row.coefficient, 8) }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="regressionResult.coefficients" />
          </section>

          <section v-if="regressionResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('TEST PREDICTIONS', '测试集预测') }}</p>
                <h3>{{ t('Actual values, predictions and residuals', '实际值、预测值和残差') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="regressionResult.preview" border size="small">
                <el-table-column
                  v-for="column in regressionPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="formatLabel(column)"
                  min-width="130"
                >
                  <template #default="scope">
                    {{
                      typeof scope.row[column] === 'number'
                        ? formatNumber(scope.row[column], 8)
                        : formatCell(scope.row[column])
                    }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="regressionResult.preview" />
          </section>

          <div
            v-for="artifact in regressionResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download predictions CSV', '下载预测 CSV')
                  : artifact.name.endsWith('.joblib')
                    ? t('Download trained Pipeline', '下载已训练 Pipeline')
                    : t('Download regression report', '下载回归报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="classificationResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Accuracy', '准确率') }}</span>
            <strong>{{ formatPercent(classificationResult.metrics.accuracy) }}</strong>
            <small>{{ t('Test dataset', '测试数据集') }}</small>
          </article>
          <article class="summary-card">
            <span>Macro F1</span>
            <strong>{{ formatPercent(classificationResult.metrics.f1_macro) }}</strong>
            <small>{{ t('Equal weight per class', '各类别等权重') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Macro precision / recall', '宏平均精确率 / 召回率') }}</span>
            <strong>
              {{ formatPercent(classificationResult.metrics.precision_macro) }} /
              {{ formatPercent(classificationResult.metrics.recall_macro) }}
            </strong>
            <small>{{ t('Test dataset', '测试数据集') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Train / test rows', '训练 / 测试行数') }}</span>
            <strong>
              {{ classificationResult.summary.train_rows }} /
              {{ classificationResult.summary.test_rows }}
            </strong>
            <small>
              {{ classificationResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
            </small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>
                  {{ classificationResult.model_display_name }} {{ t('completed', '已完成') }}
                </h2>
                <p>
                  {{ classificationResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ classificationResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Target class', '目标类别') }}</span>
              <strong>{{ classificationResult.target_column }}</strong>
            </div>
            <div>
              <span>{{ t('Feature columns', '特征列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in classificationResult.feature_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
            <div>
              <span>{{ t('Classes and split', '类别与数据划分') }}</span>
              <strong>
                {{ classificationResult.classes.join(' · ') }} ·
                {{ formatPercent(classificationResult.test_size) }} {{ t('test', '测试集') }} ·
                {{ t('random state', '随机种子') }}
                {{ classificationResult.random_state }}
              </strong>
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in classificationResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section v-if="classificationResult.cross_validation" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">STRATIFIED CROSS-VALIDATION</p>
                <h3>
                  {{ classificationResult.cross_validation.folds }}-{{
                    t('fold validation stability', '折验证稳定性')
                  }}
                </h3>
              </div>
              <el-tag type="info" effect="plain">
                {{ classificationResult.cross_validation.strategy }} · {{ t('seed', '种子') }}
                {{ classificationResult.cross_validation.random_state }}
              </el-tag>
            </div>
            <div class="validation-metric-grid">
              <article
                v-for="metric in classificationResult.cross_validation.metrics"
                :key="metric.name"
              >
                <span>{{ metric.display_name }}</span>
                <strong>{{ formatPercent(metric.mean) }}</strong>
                <small>± {{ formatPercent(metric.standard_deviation) }}</small>
              </article>
            </div>
          </section>

          <section class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CONFUSION MATRIX', '混淆矩阵') }}</p>
                <h3>{{ t('Actual and predicted class counts', '实际类别与预测类别计数') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="classificationResult.confusion_matrix" border size="small">
                <el-table-column
                  prop="actual_class"
                  :label="t('Actual class', '实际类别')"
                  min-width="160"
                />
                <el-table-column
                  prop="predicted_class"
                  :label="t('Predicted class', '预测类别')"
                  min-width="160"
                />
                <el-table-column prop="count" :label="t('Rows', '行数')" min-width="100" />
              </el-table>
            </div>
            <MobileFieldCards :rows="classificationResult.confusion_matrix" />
          </section>

          <section v-if="classificationResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('TEST PREDICTIONS', '测试集预测') }}</p>
                <h3>
                  {{
                    t(
                      'Actual class, predicted class and correctness',
                      '实际类别、预测类别和判定结果'
                    )
                  }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="classificationResult.preview" border size="small">
                <el-table-column
                  v-for="column in classificationPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="formatLabel(column)"
                  min-width="130"
                >
                  <template #default="scope">{{ formatCell(scope.row[column]) }}</template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="classificationResult.preview" />
          </section>

          <div
            v-for="artifact in classificationResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download predictions CSV', '下载预测 CSV')
                  : artifact.name.endsWith('.joblib')
                    ? t('Download trained Pipeline', '下载已训练 Pipeline')
                    : t('Download classification report', '下载分类报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="modelComparisonResult">
        <section class="summary-grid comparison-summary-grid">
          <article class="summary-card">
            <span>{{ t('Best model', '最佳模型') }}</span>
            <strong>
              {{ comparisonBestModelDisplayName }}
            </strong>
            <small>{{ t('Ranked by cross-validation mean', '按交叉验证均值排名') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Models compared', '已比较模型') }}</span>
            <strong>{{ modelComparisonResult.results.length }}</strong>
            <small>{{ modelComparisonResult.cross_validation_folds }}-fold CV</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Ranking metric', '排名指标') }}</span>
            <strong>{{ formatLabel(modelComparisonResult.comparison_metric) }}</strong>
            <small>{{ t('Higher is better', '越高越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Target column', '目标列') }}</span>
            <strong>{{ modelComparisonResult.target_column }}</strong>
            <small
              >{{ modelComparisonResult.feature_columns.length }}
              {{ t('features', '个特征') }}</small
            >
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <p class="guide-kicker">MODEL BENCHMARK</p>
                <h2>{{ t('Cross-validated model ranking', '交叉验证模型排名') }}</h2>
                <p>
                  {{ modelComparisonResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ modelComparisonResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="warning-list">
            <el-alert
              v-for="warning in modelComparisonResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              type="warning"
              :closable="false"
              show-icon
            />
          </div>

          <section class="result-section">
            <div class="table-wrap desktop-data-table">
              <el-table :data="modelComparisonResult.results" border size="small">
                <el-table-column prop="rank" :label="t('Rank', '排名')" width="78" />
                <el-table-column
                  prop="model_display_name"
                  :label="t('Model', '模型')"
                  min-width="210"
                />
                <el-table-column :label="t('Status', '状态')" width="110">
                  <template #default="scope">
                    <el-tag
                      :type="scope.row.status === 'success' ? 'success' : 'danger'"
                      effect="plain"
                    >
                      {{ scope.row.status }}
                    </el-tag>
                  </template>
                </el-table-column>
                <template v-if="modelComparisonResult.task_type === 'regression'">
                  <el-table-column label="R² mean ± SD" min-width="170">
                    <template #default="scope">
                      {{ formatNumber(comparisonMetric(scope.row, 'r2'), 5) }} ±
                      {{ formatNumber(comparisonMetricStandardDeviation(scope.row, 'r2'), 5) }}
                    </template>
                  </el-table-column>
                  <el-table-column label="MAE" min-width="120">
                    <template #default="scope">{{
                      formatNumber(comparisonMetric(scope.row, 'mean_absolute_error'), 5)
                    }}</template>
                  </el-table-column>
                  <el-table-column label="RMSE" min-width="120">
                    <template #default="scope">{{
                      formatNumber(comparisonMetric(scope.row, 'root_mean_squared_error'), 5)
                    }}</template>
                  </el-table-column>
                </template>
                <template v-else>
                  <el-table-column label="Macro F1 mean" min-width="150">
                    <template #default="scope">{{
                      formatPercent(comparisonMetric(scope.row, 'f1_macro'))
                    }}</template>
                  </el-table-column>
                  <el-table-column :label="t('Accuracy', '准确率')" min-width="130">
                    <template #default="scope">{{
                      formatPercent(comparisonMetric(scope.row, 'accuracy'))
                    }}</template>
                  </el-table-column>
                  <el-table-column :label="t('Precision', '精确率')" min-width="130">
                    <template #default="scope">{{
                      formatPercent(comparisonMetric(scope.row, 'precision_macro'))
                    }}</template>
                  </el-table-column>
                  <el-table-column :label="t('Recall', '召回率')" min-width="130">
                    <template #default="scope">{{
                      formatPercent(comparisonMetric(scope.row, 'recall_macro'))
                    }}</template>
                  </el-table-column>
                </template>
              </el-table>
            </div>
            <MobileFieldCards :rows="modelComparisonResult.results" />
          </section>

          <div
            v-for="artifact in modelComparisonResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download ranking CSV', '下载排名 CSV')
                  : t('Download comparison report', '下载比较报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="trainedSupervisedResult">
        <el-card class="result-card application-inference-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <p class="guide-kicker">APPLICATION DATA</p>
                <h2>{{ t('Apply the trained Pipeline', '使用已训练 Pipeline 进行推理') }}</h2>
                <p>
                  {{
                    t(
                      'Upload an independent dataset. The saved training Pipeline will validate the required features, impute missing values with training-set medians and generate predictions.',
                      '上传独立数据集。保存的训练 Pipeline 会校验必需特征、使用训练集的中位数填补缺失值，并生成预测。'
                    )
                  }}
                </p>
              </div>
              <el-tag type="success" effect="plain">
                {{ trainedSupervisedResult.model_display_name }}
              </el-tag>
            </div>
          </template>

          <el-alert
            :title="
              t(
                'The target column is not required. Column names must match the training features; arbitrary model-file uploads are not accepted.',
                'Application Data 不需要目标列；列名必须与训练特征一致，服务端不接受任意模型文件上传。'
              )
            "
            type="info"
            :closable="false"
            show-icon
          />

          <section class="inference-requirements">
            <span>{{ t('Required feature columns', '必需特征列') }}</span>
            <div class="selected-column-tags">
              <el-tag
                v-for="column in trainedSupervisedResult.feature_columns"
                :key="column"
                size="small"
                effect="plain"
              >
                {{ column }}
              </el-tag>
            </div>
          </section>

          <div class="inference-actions">
            <label class="file-picker">
              <input type="file" accept=".xlsx,.csv" @change="onApplicationDataChange" />
              <span class="file-button">{{
                t('Choose Application Data', '选择 Application Data')
              }}</span>
              <span class="file-name">
                {{ applicationDataFile?.name || t('No file selected', '未选择文件') }}
              </span>
            </label>
            <el-button
              type="primary"
              :disabled="!applicationDataFile || runningInference"
              :loading="runningInference"
              @click="submitInference"
            >
              {{
                runningInference
                  ? t('Running inference…', '正在推理…')
                  : t('Run application inference', '运行 Application Data 推理')
              }}
            </el-button>
          </div>

          <TaskProgress
            v-if="inferenceTaskStatus && runningInference"
            :task="inferenceTaskStatus"
            :cancelling="cancellingInference"
            @cancel="cancelInferenceTask"
          />
          <el-alert
            v-if="inferenceError"
            class="message-block"
            :title="inferenceError"
            type="error"
            :closable="false"
            show-icon
          />

          <template v-if="inferenceResult">
            <section class="summary-grid inference-summary-grid">
              <article class="summary-card">
                <span>{{ t('Predicted rows', '已预测行数') }}</span>
                <strong>{{ inferenceResult.summary.predicted_rows }}</strong>
              </article>
              <article class="summary-card">
                <span>{{ t('Imputed rows', '填补后预测行数') }}</span>
                <strong>{{ inferenceResult.summary.imputed_rows }}</strong>
              </article>
              <article class="summary-card">
                <span>{{ t('Excluded rows', '排除行数') }}</span>
                <strong>{{ inferenceResult.summary.excluded_rows }}</strong>
              </article>
              <article class="summary-card">
                <span>{{ t('Prediction column', '预测结果列') }}</span>
                <strong>{{ inferenceResult.prediction_column }}</strong>
              </article>
            </section>

            <div class="warning-list">
              <el-alert
                v-for="warning in inferenceResult.warnings"
                :key="warning"
                :title="warning"
                :type="warning.startsWith('All ') ? 'success' : 'warning'"
                :closable="false"
                show-icon
              />
            </div>

            <section v-if="inferenceResult.preview.length" class="result-section">
              <div class="section-heading">
                <div>
                  <p class="guide-kicker">INFERENCE PREVIEW</p>
                  <h3>{{ t('Application predictions', 'Application Data 预测结果') }}</h3>
                </div>
                <el-tag type="info" effect="plain">
                  {{ t('Training Job ID', '训练任务 ID') }}:
                  {{ inferenceResult.training_job_id }}
                </el-tag>
              </div>
              <div class="table-wrap desktop-data-table">
                <el-table :data="inferenceResult.preview" border size="small">
                  <el-table-column
                    v-for="column in inferencePreviewColumns"
                    :key="column"
                    :prop="column"
                    :label="formatLabel(column)"
                    min-width="140"
                  >
                    <template #default="scope">
                      {{
                        typeof scope.row[column] === 'number'
                          ? formatNumber(scope.row[column], 8)
                          : formatCell(scope.row[column])
                      }}
                    </template>
                  </el-table-column>
                </el-table>
              </div>
              <MobileFieldCards :rows="inferenceResult.preview" />
            </section>

            <div
              v-for="artifact in inferenceResult.artifacts"
              :key="artifact.download_url"
              class="artifact-row"
            >
              <div>
                <strong>{{ artifact.name }}</strong>
                <span>{{ formatBytes(artifact.size_bytes) }}</span>
              </div>
              <el-button
                type="success"
                plain
                tag="a"
                :href="artifactUrl(artifact.download_url)"
                download
              >
                {{
                  artifact.name.endsWith('.csv')
                    ? t('Download all predictions', '下载完整预测结果')
                    : t('Download inference report', '下载推理报告')
                }}
              </el-button>
            </div>
          </template>
        </el-card>
      </template>

      <template v-if="clusteringResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Silhouette score', '轮廓系数') }}</span>
            <strong>{{ formatNumber(clusteringResult.metrics.silhouette_score, 6) }}</strong>
            <small>{{ t('Higher is better', '越高越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Davies–Bouldin score', 'Davies–Bouldin 指标') }}</span>
            <strong>{{ formatNumber(clusteringResult.metrics.davies_bouldin_score, 6) }}</strong>
            <small>{{ t('Lower is better', '越低越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Calinski–Harabasz score', 'Calinski–Harabasz 指标') }}</span>
            <strong>
              {{ formatNumber(clusteringResult.metrics.calinski_harabasz_score, 3) }}
            </strong>
            <small>{{ t('Higher is better', '越高越好') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Usable rows', '可用行数') }}</span>
            <strong>{{ formatNumber(clusteringResult.summary.usable_rows) }}</strong>
            <small>
              {{ clusteringResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
            </small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>{{ clusteringResult.model_display_name }} {{ t('completed', '已完成') }}</h2>
                <p>
                  {{ clusteringResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ clusteringResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Clusters', '簇数') }}</span>
              <strong>{{ clusteringResult.cluster_count }}</strong>
              <small v-if="clusteringResult.noise_rows">
                {{ clusteringResult.noise_rows }} {{ t('noise rows', '行噪声数据') }}
              </small>
            </div>
            <div>
              <span>{{ t('Feature columns', '特征列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in clusteringResult.feature_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
            <div>
              <span>{{ t('Reproducibility', '可复现性') }}</span>
              <strong
                >{{ t('Standardized input', '标准化输入') }} · {{ t('random state', '随机种子') }}
                {{ clusteringResult.random_state }}</strong
              >
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in clusteringResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CLUSTER SIZES', '簇大小') }}</p>
                <h3>{{ t('Rows assigned to each cluster', '分配到各簇的行数') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="clusteringResult.cluster_sizes" border size="small">
                <el-table-column prop="cluster" :label="t('Cluster', '簇')" min-width="130" />
                <el-table-column prop="rows" :label="t('Rows', '行数')" min-width="130" />
              </el-table>
            </div>
            <MobileFieldCards :rows="clusteringResult.cluster_sizes" />
          </section>

          <section class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CLUSTER CENTERS', '聚类中心') }}</p>
                <h3>
                  {{
                    t(
                      'Centers converted back to original feature units',
                      '已转换回原始特征单位的聚类中心'
                    )
                  }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="clusteringResult.cluster_centers" border size="small">
                <el-table-column prop="cluster" :label="t('Cluster', '簇')" min-width="100" fixed />
                <el-table-column
                  v-for="column in clusteringResult.feature_columns"
                  :key="column"
                  :label="column"
                  min-width="140"
                >
                  <template #default="scope">
                    {{ formatNumber(scope.row.values[column], 6) }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="clusteringResult.cluster_centers" />
          </section>

          <section v-if="clusteringResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('CLUSTER ASSIGNMENTS', '聚类分配') }}</p>
                <h3>
                  {{ t('First', '前') }} {{ clusteringResult.preview.length }}
                  {{ t('usable rows', '行可用数据') }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="clusteringResult.preview" border size="small">
                <el-table-column
                  v-for="column in clusteringPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="formatLabel(column)"
                  min-width="130"
                >
                  <template #default="scope">
                    {{
                      typeof scope.row[column] === 'number'
                        ? formatNumber(scope.row[column], 8)
                        : formatCell(scope.row[column])
                    }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="clusteringResult.preview" />
          </section>

          <div
            v-for="artifact in clusteringResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download assignments CSV', '下载聚类分配 CSV')
                  : t('Download clustering report', '下载聚类报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="dimensionalityReductionResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Method', '方法') }}</span>
            <strong>{{ dimensionalityReductionResult.model_display_name }}</strong>
            <small>{{ t('Standardized numeric input', '标准化数值输入') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Output dimensions', '输出维度') }}</span>
            <strong>{{ dimensionalityReductionResult.component_count }}</strong>
            <small>
              {{ dimensionalityReductionResult.summary.feature_count }}
              {{ t('input features', '个输入特征') }}
            </small>
          </article>
          <article class="summary-card">
            <span
              v-if="dimensionalityReductionResult.metrics.total_explained_variance_ratio !== null"
            >
              {{ t('Total explained variance', '总解释方差') }}
            </span>
            <span v-else-if="dimensionalityReductionResult.metrics.kl_divergence !== null">
              KL divergence
            </span>
            <span v-else>MDS stress</span>
            <strong
              v-if="dimensionalityReductionResult.metrics.total_explained_variance_ratio !== null"
            >
              {{
                formatPercent(dimensionalityReductionResult.metrics.total_explained_variance_ratio)
              }}
            </strong>
            <strong v-else-if="dimensionalityReductionResult.metrics.kl_divergence !== null">
              {{ formatNumber(dimensionalityReductionResult.metrics.kl_divergence, 6) }}
            </strong>
            <strong v-else>{{
              formatNumber(dimensionalityReductionResult.metrics.stress, 6)
            }}</strong>
            <small>{{ t('Model diagnostic', '模型诊断值') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Usable rows', '可用行数') }}</span>
            <strong>{{ formatNumber(dimensionalityReductionResult.summary.usable_rows) }}</strong>
            <small>
              {{ dimensionalityReductionResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
            </small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>
                  {{ dimensionalityReductionResult.model_display_name }}
                  {{ t('completed', '已完成') }}
                </h2>
                <p>
                  {{ dimensionalityReductionResult.source_filename }} ·
                  {{ t('Job ID', '任务 ID') }}: {{ dimensionalityReductionResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Feature columns', '特征列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in dimensionalityReductionResult.feature_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
            <div v-if="dimensionalityReductionResult.metrics.explained_variance_ratio.length">
              <span>{{ t('Explained variance by component', '各维度解释方差') }}</span>
              <strong>
                {{
                  dimensionalityReductionResult.metrics.explained_variance_ratio
                    .map((value) => formatPercent(value))
                    .join(' · ')
                }}
              </strong>
            </div>
            <div>
              <span>{{ t('Reproducibility', '可复现性') }}</span>
              <strong>
                {{ t('Standardized input', '标准化输入') }} · {{ t('random state', '随机种子') }}
                {{ dimensionalityReductionResult.random_state }}
              </strong>
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in dimensionalityReductionResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section v-if="dimensionalityReductionResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('REDUCED COORDINATES', '降维坐标') }}</p>
                <h3>
                  {{ t('First', '前') }} {{ dimensionalityReductionResult.preview.length }}
                  {{ t('usable rows', '行可用数据') }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="dimensionalityReductionResult.preview" border size="small">
                <el-table-column
                  v-for="column in dimensionalityReductionPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="formatLabel(column)"
                  min-width="140"
                >
                  <template #default="scope">
                    {{
                      typeof scope.row[column] === 'number'
                        ? formatNumber(scope.row[column], 8)
                        : formatCell(scope.row[column])
                    }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="dimensionalityReductionResult.preview" />
          </section>

          <div
            v-for="artifact in dimensionalityReductionResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download coordinates CSV', '下载降维坐标 CSV')
                  : t('Download dimensionality reduction report', '下载降维报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="anomalyDetectionResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Anomaly rows', '异常行数') }}</span>
            <strong>{{ formatNumber(anomalyDetectionResult.summary.anomaly_rows) }}</strong>
            <small>
              {{
                formatPercent(
                  anomalyDetectionResult.summary.anomaly_rows /
                    anomalyDetectionResult.summary.usable_rows
                )
              }}
              {{ t('of usable rows', '占可用数据') }}
            </small>
          </article>
          <article class="summary-card">
            <span>{{ t('Normal rows', '正常行数') }}</span>
            <strong>{{ formatNumber(anomalyDetectionResult.summary.normal_rows) }}</strong>
            <small>{{ t('Algorithm classification', '算法判定') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Maximum anomaly score', '最高异常分数') }}</span>
            <strong>{{ formatNumber(anomalyDetectionResult.score_summary.maximum, 6) }}</strong>
            <small>{{ t('Higher means more anomalous', '数值越大越异常') }}</small>
          </article>
          <article class="summary-card">
            <span>{{ t('Usable rows', '可用行数') }}</span>
            <strong>{{ formatNumber(anomalyDetectionResult.summary.usable_rows) }}</strong>
            <small>
              {{ anomalyDetectionResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
            </small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>
                  {{ anomalyDetectionResult.model_display_name }} {{ t('completed', '已完成') }}
                </h2>
                <p>
                  {{ anomalyDetectionResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ anomalyDetectionResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Feature columns', '特征列') }}</span>
              <div class="selected-column-tags">
                <el-tag
                  v-for="column in anomalyDetectionResult.feature_columns"
                  :key="column"
                  size="small"
                  effect="plain"
                >
                  {{ column }}
                </el-tag>
              </div>
            </div>
            <div>
              <span>{{ t('Score range', '分数范围') }}</span>
              <strong>
                {{ formatNumber(anomalyDetectionResult.score_summary.minimum, 6) }} –
                {{ formatNumber(anomalyDetectionResult.score_summary.maximum, 6) }}
              </strong>
            </div>
            <div>
              <span>{{ t('Run configuration', '运行配置') }}</span>
              <strong>
                {{ anomalyReproductionProfileTitle(completedAnomalyReproductionProfile) }} ·
                {{ t('contamination', '异常比例') }}
                {{ formatAnomalyContamination(anomalyDetectionResult.contamination) }}
              </strong>
            </div>
            <div>
              <span>{{ t('Reproducibility', '可复现性') }}</span>
              <strong>
                {{ t('Standardized input', '标准化输入') }} ·
                <template v-if="anomalyDetectionResult.random_state !== null">
                  {{ t('random state', '随机种子') }} {{ anomalyDetectionResult.random_state }}
                </template>
                <template v-else>{{ t('deterministic fit', '确定性拟合') }}</template>
              </strong>
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in anomalyDetectionResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section
            v-for="figure in anomalyFigureCards"
            :key="figure.artifact.download_url"
            class="result-section anomaly-chart-section"
          >
            <div class="section-heading anomaly-chart-heading">
              <div>
                <p class="guide-kicker">{{ figure.kicker }}</p>
                <h3>{{ figure.title }}</h3>
              </div>
              <div class="anomaly-chart-actions">
                <el-button
                  type="success"
                  plain
                  tag="a"
                  :href="artifactUrl(figure.artifact.download_url)"
                  :download="figure.artifact.name"
                >
                  {{ t('Download SVG', '下载 SVG') }}
                </el-button>
                <el-button
                  type="success"
                  plain
                  :loading="exportingAnomalyPng === figure.artifact.name"
                  :disabled="
                    Boolean(exportingAnomalyPng) && exportingAnomalyPng !== figure.artifact.name
                  "
                  @click="downloadAnomalyFigurePng(figure.artifact)"
                >
                  {{ t('Download PNG', '下载 PNG') }}
                </el-button>
              </div>
            </div>
            <el-alert
              v-if="figure.reference"
              class="paper-reproduction-alert"
              :title="figure.note"
              type="info"
              :closable="false"
              show-icon
            />
            <p v-else class="chart-note">{{ figure.note }}</p>
            <p v-if="figure.methodNote" class="chart-note paper-method-note">
              {{ figure.methodNote }}
            </p>
            <div class="anomaly-chart-wrap">
              <img
                class="anomaly-chart"
                :src="artifactUrl(figure.artifact.download_url)"
                :alt="figure.alt"
              />
            </div>
          </section>

          <section v-if="anomalyDetectionResult.preview.length" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('HIGHEST ANOMALY SCORES', '最高异常分数') }}</p>
                <h3>
                  {{ t('Rows ranked from most to least anomalous', '按异常程度从高到低排列') }}
                </h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="anomalyDetectionResult.preview" border size="small">
                <el-table-column
                  v-for="column in anomalyDetectionPreviewColumns"
                  :key="column"
                  :prop="column"
                  :label="formatLabel(column)"
                  min-width="140"
                >
                  <template #default="scope">
                    {{
                      typeof scope.row[column] === 'number'
                        ? formatNumber(scope.row[column], 8)
                        : formatCell(scope.row[column])
                    }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            <MobileFieldCards :rows="anomalyDetectionResult.preview" />
          </section>

          <div
            v-for="artifact in anomalyDownloadArtifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              {{
                artifact.name.endsWith('.csv')
                  ? t('Download anomaly results CSV', '下载异常检测结果 CSV')
                  : t('Download anomaly detection report', '下载异常检测报告')
              }}
            </el-button>
          </div>
        </el-card>
      </template>

      <template v-if="timeSeriesResult">
        <section class="summary-grid">
          <article class="summary-card">
            <span>{{ t('Usable rows', '可用行数') }}</span>
            <strong>{{ formatNumber(timeSeriesResult.summary.usable_rows) }}</strong>
            <small>
              {{ timeSeriesResult.summary.dropped_rows }}
              {{ t('incomplete rows removed', '行不完整数据已删除') }}
              <template v-if="timeSeriesResult.summary.sampled_out_rows">
                · {{ formatNumber(timeSeriesResult.summary.sampled_out_rows) }}
                {{
                  t('eligible rows excluded by deterministic sampling', '个有效行未进入确定性抽样')
                }}
              </template>
            </small>
          </article>
          <article class="summary-card">
            <span>{{ t('Populated age bins', '有效年龄分箱') }}</span>
            <strong>{{ timeSeriesResult.summary.populated_bins }}</strong>
            <small>
              {{ timeSeriesResult.summary.bin_count }} {{ t('total bins', '个总分箱') }}
            </small>
          </article>
          <article class="summary-card">
            <span>{{ t('Bin width', '分箱宽度') }}</span>
            <strong
              >{{ formatNumber(timeSeriesResult.bin_width) }}
              {{ timeSeriesResult.age_unit }}</strong
            >
            <small>{{ t('Age resolution', '年龄分辨率') }}</small>
          </article>
          <article class="summary-card">
            <span>
              {{
                timeSeriesResult.analysis_type === 'element_mean'
                  ? t('Uncertainty', '不确定性')
                  : t('Bootstrap iterations', 'Bootstrap 次数')
              }}
            </span>
            <strong>
              {{
                timeSeriesResult.analysis_type === 'element_mean'
                  ? '±2 SEM'
                  : formatNumber(timeSeriesResult.bootstrap_iterations)
              }}
            </strong>
            <small v-if="timeSeriesResult.analysis_type !== 'element_mean'">
              {{ t('Random state', '随机种子') }} {{ timeSeriesResult.random_state }}
            </small>
            <small v-else>{{ t('Unweighted arithmetic mean', '未加权算术平均值') }}</small>
          </article>
        </section>

        <el-card class="result-card" shadow="never">
          <template #header>
            <div class="result-heading">
              <div>
                <h2>
                  {{
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? t('Element mean time series completed', '元素均值时间序列已完成')
                      : t(
                          'Subaerial proportion time series completed',
                          '陆上玄武岩比例时间序列已完成'
                        )
                  }}
                </h2>
                <p>
                  {{ timeSeriesResult.source_filename }} · {{ t('Job ID', '任务 ID') }}:
                  {{ timeSeriesResult.job_id }}
                </p>
              </div>
              <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
            </div>
          </template>

          <div class="result-meta regression-meta">
            <div>
              <span>{{ t('Age column', '年龄列') }}</span>
              <strong>
                {{ timeSeriesResult.age_column }}
                <template v-if="timeSeriesResult.age_max_column">
                  · {{ timeSeriesResult.age_max_column }}
                </template>
              </strong>
            </div>
            <div v-if="timeSeriesResult.analysis_type === 'element_mean'">
              <span>{{ t('Target element', '目标元素') }}</span>
              <strong
                >{{ timeSeriesResult.value_column }} ({{ timeSeriesResult.value_unit }})</strong
              >
              <small v-if="timeSeriesResult.filter_column">
                {{ timeSeriesResult.filter_column }}:
                {{ formatNumber(timeSeriesResult.filter_min) }}–{{
                  formatNumber(timeSeriesResult.filter_max)
                }}
              </small>
            </div>
            <div v-else>
              <span>{{ t('Subaerial probability', '陆上玄武岩概率') }}</span>
              <strong>{{ timeSeriesResult.probability_column }}</strong>
            </div>
            <div v-if="timeSeriesResult.probability_model">
              <span>{{ t('Probability model', '概率模型') }}</span>
              <strong>
                {{ timeSeriesResult.probability_model.display_name }} ·
                {{ timeSeriesResult.probability_model.version }}
              </strong>
              <small>
                R² {{ formatNumber(timeSeriesResult.probability_model.metrics.r2, 3) }} · MAE
                {{
                  formatNumber(timeSeriesResult.probability_model.metrics.mean_absolute_error, 3)
                }}
              </small>
            </div>
            <div v-if="timeSeriesResult.analysis_type !== 'element_mean'">
              <span>{{ t('Spatial coordinates', '空间坐标') }}</span>
              <strong>
                {{ timeSeriesResult.latitude_column }} · {{ timeSeriesResult.longitude_column }}
              </strong>
            </div>
          </div>

          <div class="warning-list">
            <el-alert
              v-for="warning in timeSeriesResult.warnings"
              :key="warning"
              :title="apiText(warning)"
              :type="warningIsSuccess(warning) ? 'success' : 'warning'"
              :closable="false"
              show-icon
            />
          </div>

          <section v-if="timeSeriesChart" class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('TIME SERIES OBSERVATIONS', '时间序列观测值') }}</p>
                <h3>
                  {{
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? `${timeSeriesResult.value_column} ${t('mean with ±2 SEM', '均值与 ±2 SEM')}`
                      : t('Estimated proportion with ±2σ uncertainty', '估计比例与 ±2σ 不确定度')
                  }}
                </h3>
              </div>
              <div class="chart-legend">
                <span>
                  <svg class="legend-error-bar" viewBox="0 0 24 16" aria-hidden="true">
                    <line x1="12" x2="12" y1="2" y2="14" />
                    <line x1="7" x2="17" y1="2" y2="2" />
                    <line x1="7" x2="17" y1="14" y2="14" />
                    <circle cx="12" cy="8" r="3" />
                  </svg>
                  {{
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? t('Mean ±2 SEM', '均值 ±2 SEM')
                      : t('Mean ±2σ', '均值 ±2σ')
                  }}
                </span>
              </div>
            </div>
            <p class="chart-note">
              {{
                t(
                  'Independent age-bin estimates; points are not connected or fitted.',
                  '各年龄分箱独立显示；散点之间不连线，也不进行曲线拟合。'
                )
              }}
            </p>
            <div class="time-series-chart-wrap">
              <svg
                class="time-series-chart"
                :viewBox="`0 0 ${timeSeriesChart.width} ${timeSeriesChart.height}`"
                role="img"
                :aria-label="
                  timeSeriesResult.analysis_type === 'element_mean'
                    ? t('Element mean time-series chart', '元素均值时间序列图')
                    : t('Subaerial proportion time-series chart', '陆上玄武岩比例时间序列图')
                "
              >
                <g v-for="tick in timeSeriesChart.yTicks" :key="tick">
                  <line
                    :x1="timeSeriesChart.left"
                    :x2="timeSeriesChart.width - timeSeriesChart.right"
                    :y1="timeSeriesChart.y(tick)"
                    :y2="timeSeriesChart.y(tick)"
                    class="chart-grid-line"
                  />
                  <text
                    :x="timeSeriesChart.left - 12"
                    :y="timeSeriesChart.y(tick) + 4"
                    text-anchor="end"
                    class="chart-tick"
                  >
                    {{ formatNumber(tick, 3) }}
                  </text>
                </g>
                <line
                  :x1="timeSeriesChart.left"
                  :x2="timeSeriesChart.left"
                  :y1="timeSeriesChart.top"
                  :y2="timeSeriesChart.height - timeSeriesChart.bottom"
                  class="chart-axis"
                />
                <line
                  :x1="timeSeriesChart.left"
                  :x2="timeSeriesChart.width - timeSeriesChart.right"
                  :y1="timeSeriesChart.height - timeSeriesChart.bottom"
                  :y2="timeSeriesChart.height - timeSeriesChart.bottom"
                  class="chart-axis"
                />
                <g
                  v-for="point in timeSeriesChart.points"
                  :key="point.age"
                  class="chart-observation"
                >
                  <title>
                    {{
                      `${t('Age', '年龄')}: ${formatNumber(point.age)} ${timeSeriesResult.age_unit}; ${t('Mean', '均值')}: ${formatNumber(point.mean, 4)}; ${timeSeriesResult.analysis_type === 'element_mean' ? '±2 SEM' : '±2σ'}: ${formatNumber(point.uncertainty, 4)}`
                    }}
                  </title>
                  <line
                    :x1="point.x"
                    :x2="point.x"
                    :y1="point.upperY"
                    :y2="point.lowerY"
                    class="chart-error-bar"
                  />
                  <line
                    :x1="point.x - 5"
                    :x2="point.x + 5"
                    :y1="point.upperY"
                    :y2="point.upperY"
                    class="chart-error-cap"
                  />
                  <line
                    :x1="point.x - 5"
                    :x2="point.x + 5"
                    :y1="point.lowerY"
                    :y2="point.lowerY"
                    class="chart-error-cap"
                  />
                  <circle :cx="point.x" :cy="point.y" r="4.5" class="chart-point" />
                </g>
                <text
                  :x="timeSeriesChart.left"
                  :y="timeSeriesChart.height - 20"
                  text-anchor="middle"
                  class="chart-tick"
                >
                  {{ formatNumber(timeSeriesChart.maximumAge) }}
                </text>
                <text
                  :x="timeSeriesChart.width - timeSeriesChart.right"
                  :y="timeSeriesChart.height - 20"
                  text-anchor="middle"
                  class="chart-tick"
                >
                  {{ formatNumber(timeSeriesChart.minimumAge) }}
                </text>
                <text
                  :x="(timeSeriesChart.left + timeSeriesChart.width - timeSeriesChart.right) / 2"
                  :y="timeSeriesChart.height - 3"
                  text-anchor="middle"
                  class="chart-label"
                >
                  {{ t('Age', '年龄') }} ({{ timeSeriesResult.age_unit }})
                </text>
                <text
                  x="16"
                  :y="timeSeriesChart.height / 2"
                  text-anchor="middle"
                  class="chart-label"
                  :transform="`rotate(-90 16 ${timeSeriesChart.height / 2})`"
                >
                  {{
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? `${timeSeriesResult.value_column} (${timeSeriesResult.value_unit})`
                      : t('Estimated proportion (%)', '估计比例（%）')
                  }}
                </text>
              </svg>
            </div>
          </section>

          <section class="result-section">
            <div class="section-heading">
              <div>
                <p class="guide-kicker">{{ t('BINNED RESULTS', '分箱结果') }}</p>
                <h3>{{ t('Age-bin statistics', '年龄分箱统计') }}</h3>
              </div>
            </div>
            <div class="table-wrap desktop-data-table">
              <el-table :data="timeSeriesResult.bins" border size="small" max-height="520">
                <el-table-column :label="`Age (${timeSeriesResult.age_unit})`" min-width="150">
                  <template #default="scope">{{ formatNumber(scope.row.age, 6) }}</template>
                </el-table-column>
                <el-table-column
                  :label="
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? `${t('Mean', '均值')} (${timeSeriesResult.value_unit})`
                      : t('Mean proportion (%)', '平均比例（%）')
                  "
                  min-width="180"
                >
                  <template #default="scope">{{
                    formatNumber(scope.row.mean_proportion, 6)
                  }}</template>
                </el-table-column>
                <el-table-column
                  :label="
                    timeSeriesResult.analysis_type === 'element_mean'
                      ? `±2 SEM (${timeSeriesResult.value_unit})`
                      : '±2σ (%)'
                  "
                  min-width="160"
                >
                  <template #default="scope">{{
                    formatNumber(scope.row.uncertainty_2sigma, 6)
                  }}</template>
                </el-table-column>
                <el-table-column
                  v-if="timeSeriesResult.analysis_type === 'element_mean'"
                  :label="t('Samples', '样本数')"
                  prop="sample_count"
                  min-width="120"
                />
              </el-table>
            </div>
            <MobileFieldCards :rows="timeSeriesResult.bins" />
          </section>

          <div
            v-for="artifact in timeSeriesResult.artifacts"
            :key="artifact.download_url"
            class="artifact-row"
          >
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <el-button
              type="success"
              plain
              tag="a"
              :href="artifactUrl(artifact.download_url)"
              download
            >
              <template v-if="artifact.name.endsWith('.csv')">
                {{ t('Download time-series CSV', '下载时间序列 CSV') }}
              </template>
              <template v-else-if="artifact.name.endsWith('.svg')">
                {{ t('Download vector figure', '下载矢量图') }}
              </template>
              <template v-else>{{ t('Download analysis report', '下载分析报告') }}</template>
            </el-button>
          </div>
        </el-card>
      </template>
    </section>

    <aside class="insight-rail">
      <RunSummary
        :file-name="datasetFile?.name"
        :rows="columnInspection?.summary.rows ?? result?.summary.rows"
        :columns="columnInspection?.summary.columns ?? result?.summary.columns"
        :missing-cells="columnInspection?.summary.missing_cells ?? result?.summary.missing_cells"
        :method="runSummaryMethod"
        :parameters="runSummaryParameters"
        :status="runSummaryStatus"
        :status-tone="runSummaryTone"
        :job-id="activeJobId || taskId"
        :software-version="softwareVersion"
      />
    </aside>
  </main>
</template>

<style lang="scss" scoped>
.data-mining-page {
  width: min(1080px, calc(100% - 40px));
  margin: 0 auto;
  padding: 48px 0 80px;
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 32px;
  margin-bottom: 28px;

  h1 {
    margin: 4px 0 8px;
    color: #1f2937;
    font-size: 38px;
    font-weight: 650;
  }

  .eyebrow {
    color: #f25b28;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
  }

  .intro {
    max-width: 720px;
    color: #64748b;
    font-size: 16px;
  }
}

.service-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  border: 1px solid #d9e1ea;
  border-radius: 999px;
  color: #64748b;
  background: #fff;
  white-space: nowrap;

  .status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #94a3b8;
  }

  &.online .status-dot {
    background: #22c55e;
  }

  &.offline .status-dot {
    background: #ef4444;
  }
}

.workflow-card,
.result-card {
  border-color: #e2e8f0;
  border-radius: 12px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 20px;

  :deep(.el-select) {
    width: 100%;
  }
}

.feature-option,
.feature-guide-heading,
.result-heading,
.section-heading,
.artifact-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.format-tags,
.output-list {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  min-height: 32px;
}

.feature-guide {
  margin-bottom: 22px;
  padding: 18px;
  border: 1px solid #dbe3ec;
  border-radius: 10px;
  background: #fbfdff;
}

.feature-guide-heading {
  align-items: flex-start;

  h2 {
    margin: 2px 0 0;
    color: #1f2937;
    font-size: 19px;
    font-weight: 650;
  }
}

.guide-kicker {
  margin: 0;
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.status-message {
  margin: 12px 0;
  color: #475569;
  line-height: 1.7;
}

.output-list {
  margin-bottom: 14px;

  strong {
    margin-right: 4px;
    color: #475569;
    font-size: 13px;
  }
}

.preprocessing-panel {
  margin: 6px 0 22px;
  padding: 18px;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  background: #f8fafc;

  .section-heading {
    margin-bottom: 16px;

    h3 {
      margin: 3px 0 0;
      color: #1f2937;
      font-size: 18px;
      font-weight: 650;
    }
  }

  :deep(.el-select) {
    width: 100%;
  }

  :deep(.el-alert) {
    margin-top: 14px;
  }
}

.column-selection {
  width: 100%;
}

.selection-tools,
.column-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.selection-tools {
  min-height: 32px;
  color: #64748b;
  font-size: 13px;
}

.column-option {
  width: 100%;

  small {
    color: #94a3b8;
  }
}

.field-help {
  margin: 6px 0 0;
  color: #64748b;
  font-size: 13px;
  line-height: 1.5;
}

.time-series-mapping-grid,
.time-series-parameter-grid {
  display: grid;
  gap: 14px;
  width: 100%;
}

.time-series-mode-picker {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 14px;
  color: #334155;
  font-size: 14px;
  font-weight: 650;

  :deep(.el-radio-group) {
    display: flex;
    flex-wrap: wrap;
  }
}

.supervised-mode-picker,
.model-settings-heading,
.cross-validation-setting {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.supervised-mode-picker {
  margin-bottom: 18px;
  padding: 16px;
  border: 1px solid #d6e6e3;
  border-radius: 8px;
  background: #f6f8f8;

  > div,
  .model-settings-heading > div,
  .cross-validation-setting > div {
    display: grid;
    gap: 4px;
  }

  strong,
  span {
    display: block;
  }

  strong {
    color: #244d55;
    font-size: 14px;
  }

  span {
    color: #647b80;
    font-size: 13px;
    line-height: 1.5;
  }
}

.wide-form-item {
  grid-column: 1 / -1;
}

.model-settings,
.cross-validation-setting {
  margin-bottom: 18px;
  padding: 16px;
  border: 1px solid #d6e6e3;
  border-radius: 8px;
  background: #fbfcfc;
}

.comparison-model-settings {
  margin-bottom: 18px;
  padding: 4px 16px;
  border: 1px solid #d6e6e3;
  border-radius: 8px;
  background: #fbfcfc;

  :deep(.el-collapse) {
    border: 0;
  }

  :deep(.el-collapse-item__header) {
    color: #294f56;
    background: transparent;
    font-size: 13px;
    font-weight: 650;
  }

  :deep(.el-collapse-item__wrap) {
    background: transparent;
  }
}

.model-settings-heading {
  margin-bottom: 14px;

  > div {
    display: grid;
    gap: 3px;
  }

  strong {
    color: #244d55;
    font-size: 14px;
  }

  span {
    color: #647b80;
    font-size: 13px;
  }
}

.hyperparameter-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.hyperparameter-field {
  display: grid;
  align-content: start;
  gap: 7px;
  min-width: 0;
  padding: 12px;
  border: 1px solid #e1e8e7;
  border-radius: 7px;
  background: #fff;

  > span {
    color: #294f56;
    font-size: 13px;
    font-weight: 650;
  }

  > small {
    color: #718589;
    line-height: 1.4;
    font-size: 11px;
  }

  :deep(.el-input-number),
  :deep(.el-select) {
    width: 100%;
  }
}

.cross-validation-setting {
  > div {
    display: grid;
    gap: 3px;
  }

  strong {
    color: #244d55;
    font-size: 14px;
  }

  > div span {
    color: #647b80;
    font-size: 13px;
  }

  > label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #526970;
    font-size: 13px;

    :deep(.el-input-number) {
      width: 112px;
    }
  }
}

.validation-metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;

  article {
    display: grid;
    gap: 5px;
    padding: 14px;
    border: 1px solid #d6e6e3;
    border-radius: 7px;
    background: #f6f8f8;
  }

  span,
  small {
    color: #61797e;
    font-size: 12px;
  }

  strong {
    color: #287453;
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
    font-size: 19px;
  }
}

.application-inference-card {
  margin-top: 24px;
}

.inference-requirements {
  display: grid;
  grid-template-columns: minmax(150px, 0.35fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
  margin: 20px 0;
  padding: 16px;
  border: 1px solid #d6e6e3;
  border-radius: 8px;
  background: #f6f8f8;

  > span {
    color: #526970;
    font-size: 14px;
    font-weight: 650;
  }
}

.inference-actions {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  margin-bottom: 18px;
}

.inference-summary-grid {
  margin-top: 24px;
}

.filter-range-inputs {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: 8px;

  :deep(.el-input-number) {
    width: 100%;
  }
}

.time-series-mapping-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.time-series-parameter-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));

  :deep(.el-input-number) {
    width: 100%;
  }
}

.file-picker {
  display: flex;
  align-items: center;
  width: 100%;
  min-height: 46px;
  overflow: hidden;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  cursor: pointer;

  &.disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
  }

  .file-button {
    align-self: stretch;
    display: flex;
    align-items: center;
    padding: 0 18px;
    color: #fff;
    background: #334155;
  }

  .file-name {
    padding: 0 16px;
    overflow: hidden;
    color: #64748b;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

.message-block {
  margin-bottom: 20px;
}

.actions {
  display: flex;
  gap: 12px;
  padding-top: 4px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 24px 0 16px;
}

.summary-card {
  display: flex;
  flex-direction: column;
  min-height: 112px;
  padding: 18px;
  border: 1px solid #dbe3ec;
  border-radius: 10px;
  background: #fff;

  span,
  small {
    color: #64748b;
    font-size: 13px;
  }

  strong {
    margin-top: 8px;
    color: #166534;
    font-size: 28px;
    font-weight: 700;
  }

  &.warning strong {
    color: #c2410c;
  }
}

.result-card {
  .result-heading {
    h2 {
      color: #166534;
      font-size: 20px;
      font-weight: 650;
    }

    p {
      color: #64748b;
      font-size: 13px;
    }
  }
}

.warning-list {
  display: grid;
  gap: 8px;
}

.result-meta {
  display: grid;
  grid-template-columns: minmax(220px, 0.8fr) minmax(320px, 1.2fr);
  gap: 16px;
  margin-bottom: 18px;
  padding: 16px;
  border-radius: 8px;
  background: #f8fafc;

  > div {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  span {
    color: #64748b;
    font-size: 12px;
  }

  strong {
    color: #1f2937;
  }
}

.selected-column-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.result-meta.regression-meta {
  grid-template-columns: 0.7fr 1.3fr 1fr;
}

.equation-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 18px;
  padding: 16px;
  border: 1px solid #dbe3ec;
  border-radius: 8px;
  background: #fbfdff;

  span {
    color: #64748b;
    font-size: 12px;
  }

  code {
    overflow-x: auto;
    color: #0f5132;
    font-size: 14px;
    line-height: 1.7;
    white-space: nowrap;
  }
}

.result-section {
  max-width: 100%;
  min-width: 0;
  margin-top: 24px;

  h3 {
    margin: 3px 0 10px;
    color: #1f2937;
    font-size: 18px;
    font-weight: 650;
  }
}

.table-wrap {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow-x: auto;
  contain: inline-size;

  :deep(.el-table) {
    min-width: 900px;
  }
}

.result-card :deep(.el-card__body) {
  min-width: 0;
}

.artifact-row {
  margin-top: 24px;
  padding: 16px;
  border-radius: 8px;
  background: #f8fafc;

  > div {
    display: flex;
    flex-direction: column;
  }

  span {
    color: #64748b;
    font-size: 13px;
  }
}

.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  color: #607c80;
  font-size: 13px;

  span {
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }
}

.legend-error-bar {
  width: 24px;
  height: 16px;
  overflow: visible;

  line {
    stroke: #287fba;
    stroke-width: 1.6;
  }

  circle {
    fill: #287fba;
  }
}

.chart-note {
  margin: -2px 0 12px;
  color: #607c80;
  font-size: 13px;
}

.time-series-chart-wrap,
.anomaly-chart-wrap {
  width: 100%;
  overflow-x: auto;
  border: 1px solid #d6e6e3;
  border-radius: 9px;
  background: #fff;
}

.time-series-chart,
.anomaly-chart {
  display: block;
  width: 100%;
  min-width: 620px;
  height: auto;
  font-family: 'IBM Plex Sans', Arial, sans-serif;
}

.anomaly-chart-section {
  margin-top: 26px;
}

.anomaly-chart-heading {
  align-items: flex-end;
}

.anomaly-chart-actions {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  gap: 8px;
}

.anomaly-contamination-control {
  display: flex;
  align-items: center;
  width: 100%;
  gap: 10px;

  strong {
    color: #294f56;
    font-size: 13px;
    font-weight: 650;
  }
}

.anomaly-contamination-input {
  width: 150px;
}

.paper-reproduction-alert {
  margin: -2px 0 12px;
}

.paper-method-note {
  margin-top: 0;
}

.chart-grid-line {
  stroke: #dbe5e7;
  stroke-width: 1;
  stroke-dasharray: 4 5;
}

.chart-axis {
  stroke: #244c54;
  stroke-width: 1.4;
}

.chart-error-bar,
.chart-error-cap {
  stroke: #287fba;
  stroke-width: 1.6;
}

.chart-point {
  fill: #287fba;
  stroke: #fff;
  stroke-width: 1.2;
}

.chart-tick {
  fill: #607c80;
  font-size: 12px;
}

.chart-label {
  fill: #244c54;
  font-size: 13px;
  font-weight: 650;
}

/* Shared alpine daylight system for the data-mining workspace. */
.data-mining-workbench {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  min-height: calc(100vh - 72px);
  overflow-x: clip;
  color: #244c54;
  background: #f3f5f6;
  font-size: 15px;
  line-height: 1.55;
}

.data-mining-page {
  width: 100%;
  max-width: 1180px;
  min-width: 0;
  margin: 0;
  padding: 30px 30px 60px;
  background: #f3f5f6;
  justify-self: center;
}

.data-mining-page > *,
.page-heading > div,
.form-grid,
.form-grid > *,
.time-series-mapping-grid,
.time-series-parameter-grid,
.feature-guide,
.feature-guide > *,
.file-picker,
.file-copy,
.result-section,
.result-meta,
.artifact-row {
  min-width: 0;
  max-width: 100%;
}

.page-heading {
  margin-bottom: 17px;

  h1 {
    margin: 5px 0 7px;
    color: #173f47;
    line-height: 1.15;
    font-size: clamp(34px, 3vw, 46px);
    letter-spacing: -0.035em;
  }

  .eyebrow {
    color: #d86149;
    font-size: 12px;
    letter-spacing: 0.14em;
  }

  .intro {
    color: #617d82;
    font-size: 15px;
    overflow-wrap: anywhere;
  }
}

.service-status {
  margin-top: 7px;
  border-color: #cfe2df;
  color: #4d6d72;
  background: #fff;

  &.online .status-dot {
    background: #67c996;
    box-shadow: 0 0 0 4px rgb(103 201 150 / 14%);
  }
}

.workflow-card,
.result-card {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  border: 1px solid #dfe5e6;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 12px 30px rgb(31 56 62 / 6%);
}

.workflow-card :deep(.el-card__body) {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  padding: 20px;
}

.form-grid {
  :deep(.el-form-item),
  :deep(.el-form-item__content),
  :deep(.el-select) {
    min-width: 0;
    max-width: 100%;
  }
}

:deep(.el-form-item__label) {
  padding-bottom: 7px;
  color: #4e6f74;
  line-height: 1.3;
  font-size: 14px;
  font-weight: 600;
}

:deep(.el-select__wrapper) {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  min-height: 42px;
  border: 1px solid #c7dcda;
  border-radius: 6px;
  background: #fff;
  box-shadow: none;

  .el-select__selected-item,
  .el-select__placeholder,
  .el-select__caret {
    color: #244d55;
  }

  &:hover {
    border-color: #78afb0;
  }

  &.is-focused {
    border-color: #df6e55;
    box-shadow: 0 0 0 2px rgb(223 110 85 / 10%);
  }
}

.feature-guide,
.preprocessing-panel {
  border-color: #c9dfdb;
  border-radius: 9px;
}

.feature-guide {
  background: #edf7f5;
}

.preprocessing-panel {
  background: #fff;
}

.feature-guide-heading h2,
.preprocessing-panel .section-heading h3,
.result-section h3 {
  color: #173f47;
}

.guide-kicker {
  color: #5c8588;
  letter-spacing: 0.12em;
}

.status-message,
.field-help,
.selection-tools {
  color: #607c80;
}

.status-message {
  font-size: 15px;
  overflow-wrap: anywhere;
}

.field-help,
.selection-tools {
  font-size: 14px;
}

.format-tags,
.output-list,
.selected-column-tags,
.rail-outputs {
  :deep(.el-tag),
  > span {
    border-color: #bfddd8;
    color: #317a7d;
    background: #f8fcfb;
  }
}

.file-picker {
  align-items: center;
  justify-content: center;
  gap: 17px;
  min-height: 112px;
  padding: 18px 24px;
  border: 1px dashed #86b8b5;
  border-radius: 9px;
  color: #294f56;
  background: #fff;
  transition:
    border-color 0.2s ease,
    background-color 0.2s ease;

  &:hover,
  &:focus-within {
    border-color: #4f9fa0;
    background: #f0f8f6;
  }

  .upload-icon {
    flex: 0 0 auto;
    color: #4f9fa0;
    font-size: 38px;
  }

  .file-copy {
    display: grid;
    gap: 3px;
    min-width: 0;
  }

  .file-button {
    align-self: auto;
    display: block;
    padding: 0;
    color: #173f47;
    background: transparent;
    font-size: 16px;
    font-weight: 650;
  }

  small {
    color: #6b8589;
    font-size: 12px;
  }

  .file-name {
    padding: 0;
    overflow: hidden;
    color: #197e83;
    font-size: 12px;
    font-style: normal;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

.actions {
  justify-content: flex-end;

  :deep(.el-button--primary) {
    min-width: 178px;
    min-height: 44px;
    border-color: #d86149;
    border-radius: 6px;
    background: #d86149;
    font-weight: 650;

    &:hover,
    &:focus-visible {
      border-color: #e1745d;
      background: #e1745d;
    }

    &.is-disabled {
      border-color: #c7d5d3;
      background: #b7c6c4;
    }
  }
}

.summary-card,
.result-meta,
.equation-card,
.artifact-row {
  border-color: #d6e6e3;
  background: #fff;
}

.summary-card {
  box-shadow: 0 10px 26px rgb(38 91 91 / 5%);

  strong {
    color: #287453;
  }
}

.result-card .result-heading h2 {
  color: #287453;
}

.result-meta,
.artifact-row {
  background: #f6f8f8;
}

:deep(.el-table) {
  --el-table-bg-color: #fff;
  --el-table-tr-bg-color: #fff;
  --el-table-header-bg-color: #e8f3f1;
  --el-table-row-hover-bg-color: #f2f8f7;
  --el-table-border-color: #d5e5e2;
  --el-table-header-text-color: #4f6f74;
  --el-table-text-color: #294f56;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.data-mining-workbench code,
.data-mining-workbench .file-name {
  font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

.insight-rail {
  position: sticky;
  top: 72px;
  align-self: start;
  height: calc(100vh - 72px);
  overflow-y: auto;
  border-left: 1px solid #e1e7e8;
  background: #fff;
  scrollbar-width: thin;
  scrollbar-color: #9cbab7 transparent;
}

.insight-section {
  padding: 32px 26px 30px;
  border-bottom: 1px solid #d3e5e1;

  h2 {
    margin: 7px 0 15px;
    color: #173f47;
    line-height: 1.45;
    font-size: 16px;
    font-weight: 650;
  }

  > p:not(.insight-kicker) {
    color: #607c80;
    line-height: 1.7;
    font-size: 12px;
  }
}

.insight-kicker {
  margin: 0;
  color: #5c8588;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.13em;
}

.pipeline-list {
  display: grid;
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;

  li {
    display: grid;
    grid-template-columns: 36px 1fr;
    align-items: center;
    gap: 12px;
    min-height: 62px;
    padding: 10px 12px;
    border: 1px solid #d2e5e1;
    border-radius: 8px;
    background: rgb(255 255 255 / 66%);
  }

  .el-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    color: #318b8e;
    background: #dff0ed;
    font-size: 17px;
  }

  div {
    display: grid;
    gap: 2px;
  }

  strong {
    color: #244c54;
    font-size: 13px;
  }

  span {
    color: #6a8588;
    line-height: 1.45;
    font-size: 11px;
  }
}

.rail-outputs {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 18px;

  span {
    padding: 5px 8px;
    border: 1px solid #bfddd8;
    border-radius: 999px;
    font-size: 10px;
  }
}

@media (max-width: 1360px) {
  .data-mining-workbench {
    grid-template-columns: minmax(0, 1fr) 300px;
  }

  .data-mining-page {
    padding-right: 26px;
    padding-left: 26px;
  }

  .insight-section {
    padding-right: 22px;
    padding-left: 22px;
  }
}

@media (max-width: 1180px) {
  .data-mining-workbench {
    grid-template-columns: minmax(0, 1fr);
  }

  .insight-rail {
    position: static;
    grid-column: 1;
    display: block;
    height: auto;
    border-top: 1px solid #d7e7e4;
    border-left: 0;
  }

  .insight-section {
    border-right: 1px solid #d3e5e1;
  }
}

@media (max-width: 820px) {
  .data-mining-workbench {
    grid-template-columns: minmax(0, 1fr);
  }

  .insight-rail {
    grid-column: 1;
    grid-template-columns: 1fr;
  }

  .insight-section {
    border-right: 0;
  }
}

@media (max-width: 760px) {
  .data-mining-page {
    width: 100%;
    padding: 24px 16px 36px;
  }

  .page-heading {
    flex-direction: column;
    gap: 12px;

    h1 {
      font-size: 30px;
    }
  }

  .feature-guide-heading,
  .section-heading,
  .result-heading {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .feature-guide {
    width: 100%;
    max-width: 100%;
    padding: 16px;
    overflow: hidden;
  }

  .form-grid,
  .time-series-mapping-grid,
  .time-series-parameter-grid,
  .hyperparameter-grid,
  .validation-metric-grid,
  .summary-grid,
  .result-meta,
  .result-meta.regression-meta {
    grid-template-columns: 1fr;
  }

  .file-picker,
  .artifact-row,
  .inference-actions {
    align-items: stretch;
    flex-direction: column;
  }

  .inference-actions,
  .inference-requirements {
    grid-template-columns: 1fr;
  }

  .supervised-mode-picker,
  .model-settings-heading,
  .cross-validation-setting {
    align-items: stretch;
    flex-direction: column;
  }

  .cross-validation-setting > label {
    justify-content: space-between;
  }

  .file-picker .file-copy {
    text-align: center;
  }

  .desktop-data-table {
    display: none;
  }
}
</style>
