<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import {
  getDataMiningCatalog,
  preprocessDataset,
  profileDataset,
  runClassification,
  runClustering,
  runRegression,
  type ClassificationResponse,
  type ClusteringResponse,
  type DataPreprocessingResponse,
  type DataMiningFeatureItem,
  type DatasetProfileResponse,
  type MissingValueStrategy,
  type RegressionResponse
} from '@/api/data-mining'
import { artifactUrl, getHealth } from '@/api/online'

type ServiceState = 'checking' | 'online' | 'offline'

const serviceState = ref<ServiceState>('checking')
const loadingCatalog = ref(true)
const running = ref(false)
const inspectingColumns = ref(false)
const features = ref<DataMiningFeatureItem[]>([])
const selectedFeature = ref('')
const datasetFile = ref<File | null>(null)
const result = ref<DatasetProfileResponse | null>(null)
const columnInspection = ref<DatasetProfileResponse | null>(null)
const preprocessingResult = ref<DataPreprocessingResponse | null>(null)
const regressionResult = ref<RegressionResponse | null>(null)
const classificationResult = ref<ClassificationResponse | null>(null)
const clusteringResult = ref<ClusteringResponse | null>(null)
const selectedColumns = ref<string[]>([])
const missingStrategy = ref<MissingValueStrategy>('keep')
const regressionTarget = ref('')
const regressionFeatures = ref<string[]>([])
const regressionTestSize = ref(0.2)
const classificationTarget = ref('')
const classificationFeatures = ref<string[]>([])
const classificationTestSize = ref(0.2)
const clusteringFeatures = ref<string[]>([])
const clusterCount = ref(3)
const errorMessage = ref('')

const missingStrategyOptions: Array<{
  value: MissingValueStrategy
  label: string
  description: string
}> = [
  {
    value: 'keep',
    label: 'Keep missing values',
    description: 'Do not replace or remove missing values.'
  },
  {
    value: 'drop_rows',
    label: 'Drop incomplete rows',
    description: 'Remove rows containing a missing value in any selected column.'
  },
  {
    value: 'fill_mean',
    label: 'Fill numeric columns with mean',
    description: 'Fill numeric missing values with the mean of each column.'
  },
  {
    value: 'fill_median',
    label: 'Fill numeric columns with median',
    description: 'Fill numeric missing values with the median of each column.'
  },
  {
    value: 'fill_mode',
    label: 'Fill each column with mode',
    description: 'Fill missing values with the most frequent non-empty value in each column.'
  }
]

const currentFeature = computed(() =>
  features.value.find((feature) => feature.name === selectedFeature.value)
)
const currentFeatureIsVerified = computed(() => currentFeature.value?.status === 'verified')
const isPreprocessing = computed(() => selectedFeature.value === 'data_preprocessing')
const isRegression = computed(() => selectedFeature.value === 'regression')
const isClassification = computed(() => selectedFeature.value === 'classification')
const isClustering = computed(() => selectedFeature.value === 'clustering')
const requiresColumnInspection = computed(
  () => isPreprocessing.value || isRegression.value || isClassification.value || isClustering.value
)
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
const clusteringPreviewColumns = computed(() =>
  Object.keys(clusteringResult.value?.preview[0] || {})
)
const numericColumns = computed(
  () =>
    columnInspection.value?.columns
      .filter((column) => column.data_type === 'number')
      .map((column) => column.name) || []
)
const regressionFeatureOptions = computed(() =>
  numericColumns.value.filter((column) => column !== regressionTarget.value)
)
const classificationTargetColumns = computed(
  () => columnInspection.value?.columns.map((column) => column.name) || []
)
const classificationFeatureOptions = computed(() =>
  numericColumns.value.filter((column) => column !== classificationTarget.value)
)
const selectedStrategy = computed(() =>
  missingStrategyOptions.find((option) => option.value === missingStrategy.value)
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
      !inspectingColumns.value
    )
  }
  if (isClassification.value) {
    return (
      Boolean(classificationTarget.value) &&
      classificationFeatures.value.length > 0 &&
      !inspectingColumns.value
    )
  }
  if (isClustering.value) {
    return clusteringFeatures.value.length > 0 && !inspectingColumns.value
  }
  return selectedFeature.value === 'dataset_profile'
})
const runButtonLabel = computed(() => {
  if (running.value) {
    if (isPreprocessing.value) return 'Processing…'
    if (isRegression.value || isClassification.value) return 'Training…'
    if (isClustering.value) return 'Clustering…'
    return 'Analyzing…'
  }
  if (!currentFeatureIsVerified.value) return '该功能暂不可运行'
  if (isPreprocessing.value) return 'Run preprocessing'
  if (isRegression.value) return 'Run regression'
  if (isClassification.value) return 'Run classification'
  if (isClustering.value) return 'Run clustering'
  return 'Analyze dataset'
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
  if (requiresColumnInspection.value && datasetFile.value) {
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
onMounted(loadPage)

async function loadPage() {
  serviceState.value = 'checking'
  loadingCatalog.value = true
  errorMessage.value = ''
  try {
    await getHealth()
    serviceState.value = 'online'
    const catalog = await getDataMiningCatalog()
    features.value = catalog.features
    selectedFeature.value =
      features.value.find((feature) => feature.status === 'verified')?.name ||
      features.value[0]?.name ||
      ''
  } catch (error) {
    serviceState.value = 'offline'
    errorMessage.value =
      error instanceof Error ? error.message : 'Cannot connect to the backend service.'
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
  errorMessage.value = ''

  if (file && !/\.(xlsx|csv)$/i.test(file.name)) {
    datasetFile.value = null
    input.value = ''
    errorMessage.value = 'Data Mining currently supports .xlsx and .csv files.'
    return
  }
  datasetFile.value = file
  if (file && requiresColumnInspection.value) {
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
  errorMessage.value = ''
  try {
    columnInspection.value = await profileDataset(datasetFile.value)
    if (isPreprocessing.value) {
      selectedColumns.value = columnInspection.value.columns.map((column) => column.name)
    } else if (isRegression.value) {
      const detectedNumericColumns = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      regressionTarget.value = detectedNumericColumns[detectedNumericColumns.length - 1] || ''
      regressionFeatures.value = detectedNumericColumns.slice(0, -1)
      if (detectedNumericColumns.length < 2) {
        errorMessage.value =
          'Regression requires at least two numeric columns: one target and one feature.'
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
        errorMessage.value =
          'Classification requires a target column and at least one numeric feature column.'
      }
    } else if (isClustering.value) {
      clusteringFeatures.value = columnInspection.value.columns
        .filter((column) => column.data_type === 'number')
        .map((column) => column.name)
      if (clusteringFeatures.value.length === 0) {
        errorMessage.value = 'Clustering requires at least one numeric feature column.'
      }
    }
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : 'Could not inspect dataset columns.'
  } finally {
    inspectingColumns.value = false
  }
}

async function submitJob() {
  if (!canRun.value || !datasetFile.value) return

  running.value = true
  clearResult()
  errorMessage.value = ''
  try {
    if (isPreprocessing.value) {
      preprocessingResult.value = await preprocessDataset(
        datasetFile.value,
        selectedColumns.value,
        missingStrategy.value
      )
      ElMessage.success('Data preprocessing completed')
    } else if (isRegression.value) {
      regressionResult.value = await runRegression(
        datasetFile.value,
        regressionTarget.value,
        regressionFeatures.value,
        regressionTestSize.value
      )
      ElMessage.success('Linear regression completed')
    } else if (isClassification.value) {
      classificationResult.value = await runClassification(
        datasetFile.value,
        classificationTarget.value,
        classificationFeatures.value,
        classificationTestSize.value
      )
      ElMessage.success('Logistic classification completed')
    } else if (isClustering.value) {
      clusteringResult.value = await runClustering(
        datasetFile.value,
        clusteringFeatures.value,
        clusterCount.value
      )
      ElMessage.success('K-means clustering completed')
    } else {
      result.value = await profileDataset(datasetFile.value)
      ElMessage.success('Dataset profile completed')
    }
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : 'The Data Mining operation failed.'
    ElMessage.error('Data Mining operation failed')
  } finally {
    running.value = false
  }
}

function clearResult() {
  result.value = null
  preprocessingResult.value = null
  regressionResult.value = null
  classificationResult.value = null
  clusteringResult.value = null
}

function selectAllColumns() {
  selectedColumns.value = columnInspection.value?.columns.map((column) => column.name) || []
}

function clearSelectedColumns() {
  selectedColumns.value = []
}

function formatLabel(value: string) {
  return value.replace(/_/g, ' ')
}

function statusLabel(status: 'verified' | 'testing') {
  return status === 'verified' ? '已验证' : '测试中'
}

function statusType(status: 'verified' | 'testing') {
  return status === 'verified' ? 'success' : 'warning'
}

function formatPercent(value: number) {
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

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<template>
  <main class="data-mining-page">
    <section class="page-heading">
      <div>
        <p class="eyebrow">GEOCHEMISTRY π ONLINE</p>
        <h1>Data mining</h1>
        <p class="intro">
          Upload a dataset, inspect its structure and quality, then continue to preprocessing and
          modeling.
        </p>
      </div>
      <div class="service-status" :class="serviceState">
        <span class="status-dot"></span>
        <span v-if="serviceState === 'checking'">Checking service</span>
        <span v-else-if="serviceState === 'online'">Backend online</span>
        <span v-else>Backend offline</span>
      </div>
    </section>

    <el-card v-loading="loadingCatalog" class="workflow-card" shadow="never">
      <el-form label-position="top">
        <div class="form-grid">
          <el-form-item label="Function">
            <el-select
              v-model="selectedFeature"
              placeholder="Select a function"
              :disabled="running"
            >
              <el-option
                v-for="feature in features"
                :key="feature.name"
                :label="feature.description"
                :value="feature.name"
              >
                <div class="feature-option">
                  <span>{{ feature.description }}</span>
                  <el-tag :type="statusType(feature.status)" size="small" effect="plain">
                    {{ statusLabel(feature.status) }}
                  </el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="Supported data">
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
              <p class="guide-kicker">功能状态与输出说明</p>
              <h2>{{ currentFeature.description }}</h2>
            </div>
            <el-tag :type="statusType(currentFeature.status)" effect="dark">
              {{ statusLabel(currentFeature.status) }}
            </el-tag>
          </div>

          <p class="status-message">{{ currentFeature.status_message }}</p>
          <div v-if="currentFeature.outputs.length" class="output-list">
            <strong>当前输出</strong>
            <el-tag
              v-for="output in currentFeature.outputs"
              :key="output"
              type="info"
              effect="plain"
            >
              {{ output }}
            </el-tag>
          </div>

          <el-alert
            v-if="!currentFeatureIsVerified"
            title="该功能仍在接入中，完成输入和结果验证后开放运行。"
            type="warning"
            :closable="false"
            show-icon
          />
        </section>

        <el-form-item label="Dataset (.xlsx or .csv)">
          <label class="file-picker" :class="{ disabled: running || !currentFeatureIsVerified }">
            <input
              type="file"
              accept=".xlsx,.csv"
              :disabled="running || !currentFeatureIsVerified"
              @change="onFileChange"
            />
            <span class="file-button">Choose dataset</span>
            <span class="file-name">{{ datasetFile?.name || 'No file selected' }}</span>
          </label>
        </el-form-item>

        <section v-if="isPreprocessing" v-loading="inspectingColumns" class="preprocessing-panel">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">PREPROCESSING CONFIGURATION</p>
              <h3>Select output columns and a missing-value rule</h3>
            </div>
            <el-tag v-if="columnInspection" type="success" effect="plain">
              {{ columnInspection.columns.length }} columns detected
            </el-tag>
          </div>

          <template v-if="columnInspection">
            <el-form-item label="Columns to keep">
              <div class="column-selection">
                <div class="selection-tools">
                  <span>
                    {{ selectedColumns.length }} of {{ columnInspection.columns.length }} selected
                  </span>
                  <div>
                    <el-button link type="primary" :disabled="running" @click="selectAllColumns">
                      Select all
                    </el-button>
                    <el-button link :disabled="running" @click="clearSelectedColumns">
                      Clear
                    </el-button>
                  </div>
                </div>
                <el-select
                  v-model="selectedColumns"
                  multiple
                  filterable
                  collapse-tags
                  collapse-tags-tooltip
                  placeholder="Select at least one column"
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
                      <small> {{ column.data_type }} · {{ column.missing }} missing </small>
                    </div>
                  </el-option>
                </el-select>
              </div>
            </el-form-item>

            <el-form-item label="Missing-value handling">
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
            title="Choose a dataset first. Its columns will be detected automatically."
            type="info"
            :closable="false"
            show-icon
          />

          <el-alert
            title="The uploaded source file is never modified. A new UTF-8 CSV file and a JSON processing record will be generated."
            type="info"
            :closable="false"
            show-icon
          />
        </section>

        <section v-if="isRegression" v-loading="inspectingColumns" class="preprocessing-panel">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">REGRESSION CONFIGURATION</p>
              <h3>Choose a numeric target and one or more numeric features</h3>
            </div>
            <el-tag v-if="columnInspection" type="success" effect="plain">
              {{ numericColumns.length }} numeric columns detected
            </el-tag>
          </div>

          <template v-if="columnInspection">
            <div class="form-grid">
              <el-form-item label="Model">
                <el-input model-value="Linear regression" disabled />
              </el-form-item>

              <el-form-item label="Test dataset size">
                <el-select v-model="regressionTestSize" :disabled="running">
                  <el-option label="20% (recommended)" :value="0.2" />
                  <el-option label="25%" :value="0.25" />
                  <el-option label="30%" :value="0.3" />
                  <el-option label="40%" :value="0.4" />
                </el-select>
              </el-form-item>
            </div>

            <el-form-item label="Target column">
              <el-select
                v-model="regressionTarget"
                filterable
                placeholder="Select the value to predict"
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
                The target is the numeric value the model will learn to predict.
              </p>
            </el-form-item>

            <el-form-item label="Feature columns">
              <el-select
                v-model="regressionFeatures"
                multiple
                filterable
                collapse-tags
                collapse-tags-tooltip
                placeholder="Select at least one numeric feature"
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
                The target column is automatically excluded from the feature list.
              </p>
            </el-form-item>
          </template>

          <el-alert
            v-else-if="!datasetFile"
            title="Choose a dataset first. Numeric columns will be detected automatically."
            type="info"
            :closable="false"
            show-icon
          />

          <el-alert
            title="Rows containing missing or infinite values in the selected columns are removed before training. At least 10 complete rows are required. The split uses random state 42 for reproducibility."
            type="info"
            :closable="false"
            show-icon
          />
        </section>

        <section v-if="isClassification" v-loading="inspectingColumns" class="preprocessing-panel">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CLASSIFICATION CONFIGURATION</p>
              <h3>Choose a label column and one or more numeric features</h3>
            </div>
            <el-tag v-if="columnInspection" type="success" effect="plain">
              {{ numericColumns.length }} numeric features available
            </el-tag>
          </div>

          <template v-if="columnInspection">
            <div class="form-grid">
              <el-form-item label="Model">
                <el-input model-value="Standardized logistic regression" disabled />
              </el-form-item>

              <el-form-item label="Test dataset size">
                <el-select v-model="classificationTestSize" :disabled="running">
                  <el-option label="20% (recommended)" :value="0.2" />
                  <el-option label="25%" :value="0.25" />
                  <el-option label="30%" :value="0.3" />
                  <el-option label="40%" :value="0.4" />
                </el-select>
              </el-form-item>
            </div>

            <el-form-item label="Target class column">
              <el-select
                v-model="classificationTarget"
                filterable
                placeholder="Select the label to predict"
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
                Text and numeric class labels are supported; missing labels are removed.
              </p>
            </el-form-item>

            <el-form-item label="Numeric feature columns">
              <el-select
                v-model="classificationFeatures"
                multiple
                filterable
                collapse-tags
                collapse-tags-tooltip
                placeholder="Select at least one numeric feature"
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
                Features are standardized automatically before the classifier is fitted.
              </p>
            </el-form-item>
          </template>

          <el-alert
            v-else-if="!datasetFile"
            title="Choose a dataset first. Label and numeric feature columns will be detected automatically."
            type="info"
            :closable="false"
            show-icon
          />

          <el-alert
            title="Incomplete rows are removed before training. At least 12 complete rows and two rows per class are required. The split is stratified and uses random state 42."
            type="info"
            :closable="false"
            show-icon
          />
        </section>

        <section v-if="isClustering" v-loading="inspectingColumns" class="preprocessing-panel">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CLUSTERING CONFIGURATION</p>
              <h3>Choose numeric features and the number of clusters</h3>
            </div>
            <el-tag v-if="columnInspection" type="success" effect="plain">
              {{ numericColumns.length }} numeric columns detected
            </el-tag>
          </div>

          <template v-if="columnInspection">
            <div class="form-grid">
              <el-form-item label="Model">
                <el-input model-value="Standardized K-means" disabled />
              </el-form-item>

              <el-form-item label="Number of clusters">
                <el-select v-model="clusterCount" :disabled="running">
                  <el-option
                    v-for="count in 9"
                    :key="count + 1"
                    :label="String(count + 1)"
                    :value="count + 1"
                  />
                </el-select>
              </el-form-item>
            </div>

            <el-form-item label="Numeric feature columns">
              <el-select
                v-model="clusteringFeatures"
                multiple
                filterable
                collapse-tags
                collapse-tags-tooltip
                placeholder="Select at least one numeric feature"
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
                Features are standardized for fitting; reported cluster centers use the original
                units.
              </p>
            </el-form-item>
          </template>

          <el-alert
            v-else-if="!datasetFile"
            title="Choose a dataset first. Numeric feature columns will be detected automatically."
            type="info"
            :closable="false"
            show-icon
          />

          <el-alert
            title="Incomplete rows are removed before clustering. At least max(10, 2 × cluster count) complete rows are required. K-means uses random state 42."
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
            Retry connection
          </el-button>
        </div>
      </el-form>
    </el-card>

    <template v-if="result">
      <section class="summary-grid">
        <article class="summary-card">
          <span>Rows</span>
          <strong>{{ formatNumber(result.summary.rows) }}</strong>
        </article>
        <article class="summary-card">
          <span>Columns</span>
          <strong>{{ formatNumber(result.summary.columns) }}</strong>
        </article>
        <article class="summary-card" :class="{ warning: result.summary.missing_cells > 0 }">
          <span>Missing cells</span>
          <strong>{{ formatPercent(result.summary.missing_rate) }}</strong>
          <small>{{ formatNumber(result.summary.missing_cells) }} cells</small>
        </article>
        <article class="summary-card" :class="{ warning: result.summary.duplicate_rows > 0 }">
          <span>Duplicate rows</span>
          <strong>{{ formatNumber(result.summary.duplicate_rows) }}</strong>
        </article>
      </section>

      <el-card class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>Dataset profile completed</h2>
              <p>{{ result.source_filename }} · Job ID: {{ result.job_id }}</p>
            </div>
            <el-tag type="success">SUCCESS</el-tag>
          </div>
        </template>

        <div class="warning-list">
          <el-alert
            v-for="warning in result.warnings"
            :key="warning"
            :title="warning"
            :type="warning.startsWith('未发现') ? 'success' : 'warning'"
            :closable="false"
            show-icon
          />
        </div>

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">COLUMN PROFILE</p>
              <h3>Column quality and descriptive statistics</h3>
            </div>
          </div>
          <div class="table-wrap">
            <el-table :data="result.columns" border size="small">
              <el-table-column prop="name" label="Column" min-width="140" fixed />
              <el-table-column prop="data_type" label="Type" min-width="95" />
              <el-table-column prop="non_null" label="Non-null" min-width="90" />
              <el-table-column prop="missing" label="Missing" min-width="85" />
              <el-table-column label="Missing rate" min-width="105">
                <template #default="scope">{{ formatPercent(scope.row.missing_rate) }}</template>
              </el-table-column>
              <el-table-column prop="unique" label="Unique" min-width="85" />
              <el-table-column label="Minimum" min-width="105">
                <template #default="scope">{{ formatNumber(scope.row.minimum) }}</template>
              </el-table-column>
              <el-table-column label="Maximum" min-width="105">
                <template #default="scope">{{ formatNumber(scope.row.maximum) }}</template>
              </el-table-column>
              <el-table-column label="Mean" min-width="105">
                <template #default="scope">{{ formatNumber(scope.row.mean) }}</template>
              </el-table-column>
              <el-table-column label="Std. dev." min-width="105">
                <template #default="scope">{{
                  formatNumber(scope.row.standard_deviation)
                }}</template>
              </el-table-column>
              <el-table-column label="Sample values" min-width="200">
                <template #default="scope">
                  {{ scope.row.sample_values.map(formatCell).join(' · ') || '—' }}
                </template>
              </el-table-column>
            </el-table>
          </div>
        </section>

        <section v-if="result.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">DATA PREVIEW</p>
              <h3>First {{ result.preview.length }} rows</h3>
            </div>
          </div>
          <div class="table-wrap">
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
        </section>

        <div v-for="artifact in result.artifacts" :key="artifact.download_url" class="artifact-row">
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
            Download JSON report
          </el-button>
        </div>
      </el-card>
    </template>

    <template v-if="preprocessingResult">
      <section class="summary-grid">
        <article class="summary-card">
          <span>Rows</span>
          <strong>{{ formatNumber(preprocessingResult.summary.processed_rows) }}</strong>
          <small>
            {{ formatNumber(preprocessingResult.summary.original_rows) }} original ·
            {{ formatNumber(preprocessingResult.summary.removed_rows) }} removed
          </small>
        </article>
        <article class="summary-card">
          <span>Columns</span>
          <strong>{{ formatNumber(preprocessingResult.summary.processed_columns) }}</strong>
          <small>
            {{ formatNumber(preprocessingResult.summary.original_columns) }} original ·
            {{ formatNumber(preprocessingResult.summary.removed_columns) }} removed
          </small>
        </article>
        <article
          class="summary-card"
          :class="{ warning: preprocessingResult.summary.processed_missing_cells > 0 }"
        >
          <span>Remaining missing cells</span>
          <strong>{{ formatNumber(preprocessingResult.summary.processed_missing_cells) }}</strong>
          <small>
            {{ formatNumber(preprocessingResult.summary.original_missing_cells) }} before processing
          </small>
        </article>
        <article class="summary-card">
          <span>Filled cells</span>
          <strong>{{ formatNumber(preprocessingResult.summary.filled_cells) }}</strong>
          <small>{{ selectedStrategy?.label }}</small>
        </article>
      </section>

      <el-card class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>Data preprocessing completed</h2>
              <p>
                {{ preprocessingResult.source_filename }} · Job ID:
                {{ preprocessingResult.job_id }}
              </p>
            </div>
            <el-tag type="success">SUCCESS</el-tag>
          </div>
        </template>

        <div class="result-meta">
          <div>
            <span>Missing-value rule</span>
            <strong>{{
              selectedStrategy?.label || formatLabel(preprocessingResult.missing_strategy)
            }}</strong>
          </div>
          <div>
            <span>Selected columns</span>
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
            :title="warning"
            :type="warning.includes('没有缺失单元格') ? 'success' : 'warning'"
            :closable="false"
            show-icon
          />
        </div>

        <section v-if="preprocessingResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">PROCESSED DATA PREVIEW</p>
              <h3>First {{ preprocessingResult.preview.length }} rows</h3>
            </div>
          </div>
          <div class="table-wrap">
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
                ? 'Download processed CSV'
                : 'Download processing record'
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
          <small>Test dataset</small>
        </article>
        <article class="summary-card">
          <span>Mean absolute error</span>
          <strong>
            {{ formatNumber(regressionResult.metrics.mean_absolute_error, 6) }}
          </strong>
          <small>Lower is better</small>
        </article>
        <article class="summary-card">
          <span>Root mean squared error</span>
          <strong>
            {{ formatNumber(regressionResult.metrics.root_mean_squared_error, 6) }}
          </strong>
          <small>Lower is better</small>
        </article>
        <article class="summary-card">
          <span>Train / test rows</span>
          <strong>
            {{ regressionResult.summary.train_rows }} /
            {{ regressionResult.summary.test_rows }}
          </strong>
          <small> {{ regressionResult.summary.dropped_rows }} incomplete rows removed </small>
        </article>
      </section>

      <el-card class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>Linear regression completed</h2>
              <p>
                {{ regressionResult.source_filename }} · Job ID:
                {{ regressionResult.job_id }}
              </p>
            </div>
            <el-tag type="success">SUCCESS</el-tag>
          </div>
        </template>

        <div class="result-meta regression-meta">
          <div>
            <span>Target column</span>
            <strong>{{ regressionResult.target_column }}</strong>
          </div>
          <div>
            <span>Feature columns</span>
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
            <span>Reproducible split</span>
            <strong>
              {{ formatPercent(regressionResult.test_size) }} test · random state
              {{ regressionResult.random_state }}
            </strong>
          </div>
        </div>

        <div class="equation-card">
          <span>Fitted equation</span>
          <code>{{ regressionResult.equation }}</code>
        </div>

        <div class="warning-list">
          <el-alert
            v-for="warning in regressionResult.warnings"
            :key="warning"
            :title="warning"
            :type="warning.includes('均可用于回归') ? 'success' : 'warning'"
            :closable="false"
            show-icon
          />
        </div>

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">MODEL COEFFICIENTS</p>
              <h3>Feature effects in the fitted linear model</h3>
            </div>
            <el-tag type="info" effect="plain">
              Intercept: {{ formatNumber(regressionResult.intercept, 6) }}
            </el-tag>
          </div>
          <el-table :data="regressionResult.coefficients" border size="small">
            <el-table-column prop="feature" label="Feature" min-width="180" />
            <el-table-column label="Coefficient" min-width="160">
              <template #default="scope">
                {{ formatNumber(scope.row.coefficient, 8) }}
              </template>
            </el-table-column>
          </el-table>
        </section>

        <section v-if="regressionResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">TEST PREDICTIONS</p>
              <h3>Actual values, predictions and residuals</h3>
            </div>
          </div>
          <div class="table-wrap">
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
                ? 'Download predictions CSV'
                : 'Download regression report'
            }}
          </el-button>
        </div>
      </el-card>
    </template>

    <template v-if="classificationResult">
      <section class="summary-grid">
        <article class="summary-card">
          <span>Accuracy</span>
          <strong>{{ formatPercent(classificationResult.metrics.accuracy) }}</strong>
          <small>Test dataset</small>
        </article>
        <article class="summary-card">
          <span>Macro F1</span>
          <strong>{{ formatPercent(classificationResult.metrics.f1_macro) }}</strong>
          <small>Equal weight per class</small>
        </article>
        <article class="summary-card">
          <span>Macro precision / recall</span>
          <strong>
            {{ formatPercent(classificationResult.metrics.precision_macro) }} /
            {{ formatPercent(classificationResult.metrics.recall_macro) }}
          </strong>
          <small>Test dataset</small>
        </article>
        <article class="summary-card">
          <span>Train / test rows</span>
          <strong>
            {{ classificationResult.summary.train_rows }} /
            {{ classificationResult.summary.test_rows }}
          </strong>
          <small> {{ classificationResult.summary.dropped_rows }} incomplete rows removed </small>
        </article>
      </section>

      <el-card class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>Logistic classification completed</h2>
              <p>
                {{ classificationResult.source_filename }} · Job ID:
                {{ classificationResult.job_id }}
              </p>
            </div>
            <el-tag type="success">SUCCESS</el-tag>
          </div>
        </template>

        <div class="result-meta regression-meta">
          <div>
            <span>Target class</span>
            <strong>{{ classificationResult.target_column }}</strong>
          </div>
          <div>
            <span>Feature columns</span>
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
            <span>Classes and split</span>
            <strong>
              {{ classificationResult.classes.join(' · ') }} ·
              {{ formatPercent(classificationResult.test_size) }} test · random state
              {{ classificationResult.random_state }}
            </strong>
          </div>
        </div>

        <div class="warning-list">
          <el-alert
            v-for="warning in classificationResult.warnings"
            :key="warning"
            :title="warning"
            :type="warning.includes('均可用于分类') ? 'success' : 'warning'"
            :closable="false"
            show-icon
          />
        </div>

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CONFUSION MATRIX</p>
              <h3>Actual and predicted class counts</h3>
            </div>
          </div>
          <el-table :data="classificationResult.confusion_matrix" border size="small">
            <el-table-column prop="actual_class" label="Actual class" min-width="160" />
            <el-table-column prop="predicted_class" label="Predicted class" min-width="160" />
            <el-table-column prop="count" label="Rows" min-width="100" />
          </el-table>
        </section>

        <section v-if="classificationResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">TEST PREDICTIONS</p>
              <h3>Actual class, predicted class and correctness</h3>
            </div>
          </div>
          <div class="table-wrap">
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
                ? 'Download predictions CSV'
                : 'Download classification report'
            }}
          </el-button>
        </div>
      </el-card>
    </template>

    <template v-if="clusteringResult">
      <section class="summary-grid">
        <article class="summary-card">
          <span>Silhouette score</span>
          <strong>{{ formatNumber(clusteringResult.metrics.silhouette_score, 6) }}</strong>
          <small>Higher is better</small>
        </article>
        <article class="summary-card">
          <span>Davies–Bouldin score</span>
          <strong>{{ formatNumber(clusteringResult.metrics.davies_bouldin_score, 6) }}</strong>
          <small>Lower is better</small>
        </article>
        <article class="summary-card">
          <span>Calinski–Harabasz score</span>
          <strong>
            {{ formatNumber(clusteringResult.metrics.calinski_harabasz_score, 3) }}
          </strong>
          <small>Higher is better</small>
        </article>
        <article class="summary-card">
          <span>Usable rows</span>
          <strong>{{ formatNumber(clusteringResult.summary.usable_rows) }}</strong>
          <small> {{ clusteringResult.summary.dropped_rows }} incomplete rows removed </small>
        </article>
      </section>

      <el-card class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>K-means clustering completed</h2>
              <p>
                {{ clusteringResult.source_filename }} · Job ID:
                {{ clusteringResult.job_id }}
              </p>
            </div>
            <el-tag type="success">SUCCESS</el-tag>
          </div>
        </template>

        <div class="result-meta regression-meta">
          <div>
            <span>Clusters</span>
            <strong>{{ clusteringResult.cluster_count }}</strong>
          </div>
          <div>
            <span>Feature columns</span>
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
            <span>Reproducibility</span>
            <strong>Standardized input · random state {{ clusteringResult.random_state }}</strong>
          </div>
        </div>

        <div class="warning-list">
          <el-alert
            v-for="warning in clusteringResult.warnings"
            :key="warning"
            :title="warning"
            :type="warning.includes('均可用于聚类') ? 'success' : 'warning'"
            :closable="false"
            show-icon
          />
        </div>

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CLUSTER SIZES</p>
              <h3>Rows assigned to each cluster</h3>
            </div>
          </div>
          <el-table :data="clusteringResult.cluster_sizes" border size="small">
            <el-table-column prop="cluster" label="Cluster" min-width="130" />
            <el-table-column prop="rows" label="Rows" min-width="130" />
          </el-table>
        </section>

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CLUSTER CENTERS</p>
              <h3>Centers converted back to original feature units</h3>
            </div>
          </div>
          <div class="table-wrap">
            <el-table :data="clusteringResult.cluster_centers" border size="small">
              <el-table-column prop="cluster" label="Cluster" min-width="100" fixed />
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
        </section>

        <section v-if="clusteringResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">CLUSTER ASSIGNMENTS</p>
              <h3>First {{ clusteringResult.preview.length }} usable rows</h3>
            </div>
          </div>
          <div class="table-wrap">
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
                ? 'Download assignments CSV'
                : 'Download clustering report'
            }}
          </el-button>
        </div>
      </el-card>
    </template>
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

@media (max-width: 760px) {
  .data-mining-page {
    width: min(100% - 24px, 1080px);
    padding-top: 28px;
  }

  .page-heading {
    flex-direction: column;
    gap: 12px;

    h1 {
      font-size: 30px;
    }
  }

  .form-grid,
  .summary-grid,
  .result-meta,
  .result-meta.regression-meta {
    grid-template-columns: 1fr;
  }

  .file-picker,
  .artifact-row {
    align-items: stretch;
    flex-direction: column;
  }

  .file-picker .file-button,
  .file-picker .file-name {
    min-height: 42px;
    justify-content: center;
  }

  .table-wrap :deep(.el-table) {
    min-width: 100%;
  }

  .table-wrap {
    overflow: hidden;
  }
}
</style>
