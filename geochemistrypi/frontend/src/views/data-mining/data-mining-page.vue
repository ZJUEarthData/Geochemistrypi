<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import WorkspaceSidebar from '@/components/workspace-sidebar.vue'

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
import { apiText, dataMiningFeatureDescription, t, warningIsSuccess } from '@/i18n'

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
    if (isPreprocessing.value) return t('Processing…', '正在处理…')
    if (isRegression.value || isClassification.value) return t('Training…', '正在训练…')
    if (isClustering.value) return t('Clustering…', '正在聚类…')
    return t('Analyzing…', '正在分析…')
  }
  if (!currentFeatureIsVerified.value)
    return t('This function is not available yet', '该功能暂不可运行')
  if (isPreprocessing.value) return t('Run preprocessing', '运行预处理')
  if (isRegression.value) return t('Run regression', '运行回归')
  if (isClassification.value) return t('Run classification', '运行分类')
  if (isClustering.value) return t('Run clustering', '运行聚类')
  return t('Analyze dataset', '分析数据集')
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
    }
  } catch (error) {
    errorMessage.value =
      error instanceof Error
        ? error.message
        : t('Could not inspect dataset columns.', '无法检测数据集列。')
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
      ElMessage.success(t('Data preprocessing completed', '数据预处理完成'))
    } else if (isRegression.value) {
      regressionResult.value = await runRegression(
        datasetFile.value,
        regressionTarget.value,
        regressionFeatures.value,
        regressionTestSize.value
      )
      ElMessage.success(t('Linear regression completed', '线性回归完成'))
    } else if (isClassification.value) {
      classificationResult.value = await runClassification(
        datasetFile.value,
        classificationTarget.value,
        classificationFeatures.value,
        classificationTestSize.value
      )
      ElMessage.success(t('Logistic classification completed', '逻辑分类完成'))
    } else if (isClustering.value) {
      clusteringResult.value = await runClustering(
        datasetFile.value,
        clusteringFeatures.value,
        clusterCount.value
      )
      ElMessage.success(t('K-means clustering completed', 'K-means 聚类完成'))
    } else {
      result.value = await profileDataset(datasetFile.value)
      ElMessage.success(t('Dataset profile completed', '数据集概览完成'))
    }
  } catch (error) {
    errorMessage.value =
      error instanceof Error
        ? error.message
        : t('The Data Mining operation failed.', '数据挖掘操作失败。')
    ElMessage.error(t('Data Mining operation failed', '数据挖掘操作失败'))
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
  return status === 'verified' ? t('Verified', '已验证') : t('Testing', '测试中')
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
  <main class="data-mining-workbench">
    <WorkspaceSidebar active="data-mining" />

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
        <span v-if="serviceState === 'checking'">{{ t('Checking service', '正在检查服务') }}</span>
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
                  <span>{{ dataMiningFeatureDescription(feature.name, feature.description) }}</span>
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
                {{ dataMiningFeatureDescription(currentFeature.name, currentFeature.description) }}
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
              <small>{{ t('Drop an XLSX or CSV file here, or click to browse.', '拖放 XLSX 或 CSV 文件到此处，或点击浏览。') }}</small>
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
                  t('Select output columns and a missing-value rule', '选择输出列和缺失值处理规则')
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
            <div class="form-grid">
              <el-form-item :label="t('Model', '模型')">
                <el-input :model-value="t('Linear regression', '线性回归')" disabled />
              </el-form-item>

              <el-form-item :label="t('Test dataset size', '测试集比例')">
                <el-select v-model="regressionTestSize" :disabled="running">
                  <el-option :label="t('20% (recommended)', '20%（推荐）')" :value="0.2" />
                  <el-option label="25%" :value="0.25" />
                  <el-option label="30%" :value="0.3" />
                  <el-option label="40%" :value="0.4" />
                </el-select>
              </el-form-item>
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

        <section v-if="isClassification" v-loading="inspectingColumns" class="preprocessing-panel">
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
            <div class="form-grid">
              <el-form-item :label="t('Model', '模型')">
                <el-input
                  :model-value="t('Standardized logistic regression', '标准化逻辑回归')"
                  disabled
                />
              </el-form-item>

              <el-form-item :label="t('Test dataset size', '测试集比例')">
                <el-select v-model="classificationTestSize" :disabled="running">
                  <el-option :label="t('20% (recommended)', '20%（推荐）')" :value="0.2" />
                  <el-option label="25%" :value="0.25" />
                  <el-option label="30%" :value="0.3" />
                  <el-option label="40%" :value="0.4" />
                </el-select>
              </el-form-item>
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
                {{ t('Choose numeric features and the number of clusters', '选择数值特征和簇数') }}
              </h3>
            </div>
            <el-tag v-if="columnInspection" type="success" effect="plain">
              {{ numericColumns.length }} {{ t('numeric columns detected', '个数值列已识别') }}
            </el-tag>
          </div>

          <template v-if="columnInspection">
            <div class="form-grid">
              <el-form-item :label="t('Model', '模型')">
                <el-input :model-value="t('Standardized K-means', '标准化 K-means')" disabled />
              </el-form-item>

              <el-form-item :label="t('Number of clusters', '簇数')">
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
          <div class="table-wrap">
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
        </section>

        <section v-if="result.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">{{ t('DATA PREVIEW', '数据预览') }}</p>
              <h3>{{ t('First', '前') }} {{ result.preview.length }} {{ t('rows', '行') }}</h3>
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
              <h2>{{ t('Linear regression completed', '线性回归完成') }}</h2>
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

        <div class="equation-card">
          <span>{{ t('Fitted equation', '拟合方程') }}</span>
          <code>{{ regressionResult.equation }}</code>
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

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">{{ t('MODEL COEFFICIENTS', '模型系数') }}</p>
              <h3>
                {{ t('Feature effects in the fitted linear model', '特征在拟合线性模型中的影响') }}
              </h3>
            </div>
            <el-tag type="info" effect="plain">
              {{ t('Intercept', '截距') }}: {{ formatNumber(regressionResult.intercept, 6) }}
            </el-tag>
          </div>
          <el-table :data="regressionResult.coefficients" border size="small">
            <el-table-column prop="feature" :label="t('Feature', '特征')" min-width="180" />
            <el-table-column :label="t('Coefficient', '系数')" min-width="160">
              <template #default="scope">
                {{ formatNumber(scope.row.coefficient, 8) }}
              </template>
            </el-table-column>
          </el-table>
        </section>

        <section v-if="regressionResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">{{ t('TEST PREDICTIONS', '测试集预测') }}</p>
              <h3>{{ t('Actual values, predictions and residuals', '实际值、预测值和残差') }}</h3>
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
                ? t('Download predictions CSV', '下载预测 CSV')
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
              <h2>{{ t('Logistic classification completed', '逻辑分类完成') }}</h2>
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

        <section class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">{{ t('CONFUSION MATRIX', '混淆矩阵') }}</p>
              <h3>{{ t('Actual and predicted class counts', '实际类别与预测类别计数') }}</h3>
            </div>
          </div>
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
        </section>

        <section v-if="classificationResult.preview.length" class="result-section">
          <div class="section-heading">
            <div>
              <p class="guide-kicker">{{ t('TEST PREDICTIONS', '测试集预测') }}</p>
              <h3>
                {{
                  t('Actual class, predicted class and correctness', '实际类别、预测类别和判定结果')
                }}
              </h3>
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
                ? t('Download predictions CSV', '下载预测 CSV')
                : t('Download classification report', '下载分类报告')
            }}
          </el-button>
        </div>
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
              <h2>{{ t('K-means clustering completed', 'K-means 聚类完成') }}</h2>
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
          <el-table :data="clusteringResult.cluster_sizes" border size="small">
            <el-table-column prop="cluster" :label="t('Cluster', '簇')" min-width="130" />
            <el-table-column prop="rows" :label="t('Rows', '行数')" min-width="130" />
          </el-table>
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
          <div class="table-wrap">
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
                ? t('Download assignments CSV', '下载聚类分配 CSV')
                : t('Download clustering report', '下载聚类报告')
            }}
          </el-button>
        </div>
      </el-card>
    </template>
    </section>

    <aside class="insight-rail">
      <section class="insight-section">
        <p class="insight-kicker">{{ t('ANALYSIS PIPELINE', '分析流程') }}</p>
        <h2>{{ t('From field data to evidence', '从野外数据到可验证证据') }}</h2>
        <ol class="pipeline-list">
          <li>
            <el-icon><UploadFilled /></el-icon>
            <div>
              <strong>{{ t('Upload', '上传') }}</strong>
              <span>{{ t('XLSX or CSV field dataset', 'XLSX 或 CSV 野外数据集') }}</span>
            </div>
          </li>
          <li>
            <el-icon><Search /></el-icon>
            <div>
              <strong>{{ t('Inspect', '检查') }}</strong>
              <span>{{ t('Types, missing values and ranges', '类型、缺失值与数值范围') }}</span>
            </div>
          </li>
          <li>
            <el-icon><SetUp /></el-icon>
            <div>
              <strong>{{ t('Model', '建模') }}</strong>
              <span>{{ t('Prepare, regress, classify or cluster', '预处理、回归、分类或聚类') }}</span>
            </div>
          </li>
          <li>
            <el-icon><Download /></el-icon>
            <div>
              <strong>{{ t('Export', '导出') }}</strong>
              <span>{{ t('Traceable tables and reports', '可追溯的数据表与报告') }}</span>
            </div>
          </li>
        </ol>
      </section>

      <section v-if="currentFeature" class="insight-section current-insight">
        <p class="insight-kicker">{{ t('CURRENT FUNCTION', '当前功能') }}</p>
        <h2>{{ dataMiningFeatureDescription(currentFeature.name, currentFeature.description) }}</h2>
        <p>{{ apiText(currentFeature.status_message) }}</p>
        <div v-if="currentFeature.outputs.length" class="rail-outputs">
          <span v-for="output in currentFeature.outputs" :key="output">
            {{ apiText(output) }}
          </span>
        </div>
      </section>
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

/* Shared alpine daylight system for the data-mining workspace. */
.data-mining-workbench {
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr) 330px;
  min-height: calc(100vh - 72px);
  color: #244c54;
  background: #f1f8f7;
}

.data-mining-page {
  width: 100%;
  min-width: 0;
  margin: 0;
  padding: 30px 30px 60px;
  background: #f4faf9;
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
  min-width: 0;
  border: 1px solid #d6e6e3;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 14px 36px rgb(38 91 91 / 7%);
}

.workflow-card :deep(.el-card__body) {
  padding: 20px;
}

:deep(.el-form-item__label) {
  padding-bottom: 7px;
  color: #4e6f74;
  line-height: 1.3;
  font-size: 13px;
  font-weight: 600;
}

:deep(.el-select__wrapper) {
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
  background: #edf7f5;
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
  background: #edf7f5;
}

:deep(.el-table) {
  --el-table-bg-color: #fff;
  --el-table-tr-bg-color: #fff;
  --el-table-header-bg-color: #e8f3f1;
  --el-table-row-hover-bg-color: #f2f8f7;
  --el-table-border-color: #d5e5e2;
  --el-table-header-text-color: #4f6f74;
  --el-table-text-color: #294f56;
}

.insight-rail {
  position: sticky;
  top: 72px;
  align-self: start;
  height: calc(100vh - 72px);
  overflow-y: auto;
  border-left: 1px solid #d7e7e4;
  background: #eaf5f2;
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
    grid-template-columns: 210px minmax(0, 1fr) 300px;
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
    grid-template-columns: 190px minmax(0, 1fr);
  }

  .insight-rail {
    position: static;
    grid-column: 2;
    display: grid;
    grid-template-columns: 1fr 1fr;
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
    padding: 28px 20px 42px;
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

  .file-picker .file-copy {
    text-align: center;
  }

  .table-wrap :deep(.el-table) {
    min-width: 100%;
  }

  .table-wrap {
    overflow: hidden;
  }
}
</style>
