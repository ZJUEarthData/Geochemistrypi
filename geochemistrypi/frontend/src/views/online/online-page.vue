<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import FormulaDisplay from '@/components/formula-display.vue'
import RunSummary from '@/components/run-summary.vue'
import TaskProgress from '@/components/task-progress.vue'
import { useTaskTracking } from '@/composables/use-task-tracking'

import { profileDataset, type DatasetProfileResponse } from '@/api/data-mining'

import {
  DEFAULT_MAX_UPLOAD_BYTES,
  artifactUrl,
  getCatalog,
  getHealth,
  runChemicalModel,
  type ArtifactResponse,
  type TaskCatalogItem
} from '@/api/online'
import {
  apiText,
  chemicalMethodDescription,
  chemicalMethodFormula,
  chemicalMethodStatus,
  t
} from '@/i18n'

type ServiceState = 'checking' | 'online' | 'offline'

const serviceState = ref<ServiceState>('checking')
const softwareVersion = ref('')
const maxUploadBytes = ref(DEFAULT_MAX_UPLOAD_BYTES)
const taskTimeoutMinutes = ref(30)
const maxConcurrentTasks = ref(1)
const loadingCatalog = ref(true)
const inspectingDataset = ref(false)
const running = ref(false)
const tasks = ref<TaskCatalogItem[]>([])
const selectedTask = ref('')
const selectedMethod = ref('')
const selectedElement = ref('')
const datasetFile = ref<File | null>(null)
const datasetProfile = ref<DatasetProfileResponse | null>(null)
const artifacts = ref<ArtifactResponse[]>([])
const jobId = ref('')
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

const resourceLimitNote = computed(() =>
  t(
    `Maximum file size: ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB. The site runs ${maxConcurrentTasks.value} calculation at a time; additional jobs wait in the queue. A running job stops after ${taskTimeoutMinutes.value} minutes.`,
    `文件大小不得超过 ${Math.round(maxUploadBytes.value / 1024 / 1024)} MB。全站同时只运行 ${maxConcurrentTasks.value} 个计算任务，其他任务排队等待；任务开始运行 ${taskTimeoutMinutes.value} 分钟后将自动停止。`
  )
)

const availableTasks = computed(() => tasks.value.filter((task) => task.available))
const unavailableTasks = computed(() => tasks.value.filter((task) => !task.available))
const currentTask = computed(() => tasks.value.find((task) => task.name === selectedTask.value))
const availableMethods = computed(() => currentTask.value?.methods || [])
const currentMethod = computed(() =>
  availableMethods.value.find((method) => method.name === selectedMethod.value)
)
const availableElements = computed(() => currentMethod.value?.elements || [])
const requiredColumns = computed(() => currentMethod.value?.required_columns || [])
const currentMethodIsVerified = computed(() => currentMethod.value?.status === 'verified')
const canRun = computed(
  () =>
    serviceState.value === 'online' &&
    Boolean(selectedTask.value) &&
    Boolean(selectedMethod.value) &&
    Boolean(selectedElement.value) &&
    Boolean(datasetFile.value) &&
    currentMethodIsVerified.value &&
    !inspectingDataset.value &&
    !running.value
)

const runSummaryStatus = computed(() => {
  if (serviceState.value === 'checking') return t('Checking service', '正在检查服务')
  if (serviceState.value === 'offline') return t('Backend offline', '后端离线')
  if (inspectingDataset.value) return t('Inspecting dataset', '正在检查数据集')
  if (taskStatus.value?.status === 'queued') return t('Waiting in queue', '正在排队')
  if (taskStatus.value?.status === 'cancelling') return t('Cancelling', '正在取消')
  if (taskStatus.value?.status === 'cancelled') return t('Cancelled', '已取消')
  if (running.value) return t('Calculating', '正在计算')
  if (jobId.value) return t('Completed', '已完成')
  if (errorMessage.value) return t('Needs attention', '需要检查')
  if (datasetFile.value) return t('Ready to run', '可开始运行')
  return t('Waiting for dataset', '等待数据集')
})

const runSummaryTone = computed(() => {
  if (serviceState.value === 'offline' || errorMessage.value) return 'danger' as const
  if (running.value || inspectingDataset.value || serviceState.value === 'checking')
    return 'info' as const
  if (jobId.value) return 'success' as const
  if (datasetFile.value) return 'warning' as const
  return 'neutral' as const
})

const runSummaryMethod = computed(() =>
  currentMethod.value
    ? chemicalMethodDescription(currentMethod.value.name, currentMethod.value.description)
    : ''
)

const runSummaryParameters = computed(() => [
  `${t('Task', '任务')}: ${selectedTask.value ? formatLabel(selectedTask.value) : '—'}`,
  `${t('Element', '元素')}: ${selectedElement.value || '—'}`
])

watch(selectedTask, () => {
  selectedMethod.value =
    availableMethods.value.find((method) => method.status === 'verified')?.name ||
    availableMethods.value[0]?.name ||
    ''
  clearResult()
})

watch(selectedMethod, () => {
  selectedElement.value = availableElements.value[0] || ''
  clearResult()
})

watch(selectedElement, clearResult)

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
    const catalog = await getCatalog()
    tasks.value = catalog.tasks
    selectedTask.value =
      availableTasks.value.find((task) =>
        task.methods.some((method) => method.status === 'verified')
      )?.name ||
      availableTasks.value[0]?.name ||
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
  datasetProfile.value = null
  errorMessage.value = ''

  if (file && !/\.(xlsx|csv)$/i.test(file.name)) {
    datasetFile.value = null
    input.value = ''
    errorMessage.value = t(
      'Chemical Modeling supports .xlsx and .csv files.',
      '化学建模支持 .xlsx 和 .csv 文件。'
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

  if (!file) return

  inspectingDataset.value = true
  const trackingId = beginTask('Dataset profile')
  try {
    datasetProfile.value = await profileDataset(file, trackingId)
  } catch (error) {
    if (!cancelledByUser.value) {
      errorMessage.value =
        error instanceof Error
          ? error.message
          : t('Could not inspect the selected dataset.', '无法检查所选数据集。')
    }
  } finally {
    await finishTask()
    inspectingDataset.value = false
  }
}

async function submitJob() {
  if (!canRun.value || !datasetFile.value) return

  running.value = true
  clearResult()
  errorMessage.value = ''
  const trackingId = beginTask(`Chemical modeling: ${selectedMethod.value}`)
  try {
    const result = await runChemicalModel(
      selectedTask.value,
      selectedMethod.value,
      selectedElement.value,
      datasetFile.value,
      trackingId
    )
    artifacts.value = result.artifacts
    jobId.value = result.job_id
    ElMessage.success(t('Calculation completed', '计算完成'))
  } catch (error) {
    if (cancelledByUser.value) {
      ElMessage.info(t('Task cancelled', '任务已取消'))
    } else {
      errorMessage.value =
        error instanceof Error ? error.message : t('Calculation failed.', '计算失败。')
      ElMessage.error(t('Calculation failed', '计算失败'))
    }
  } finally {
    await finishTask()
    running.value = false
  }
}

function clearResult() {
  artifacts.value = []
  jobId.value = ''
}

function formatLabel(value: string) {
  const taskLabels: Record<string, [string, string]> = {
    algo_equilibrium: ['Equilibrium', '化学平衡'],
    algo_fractionation: ['Isotope fractionation', '同位素分馏'],
    algo_kinetic: ['Kinetics', '动力学'],
    algo_solubility: ['Solubility', '溶解度'],
    algo_thermodynamic: ['Thermodynamics', '热力学'],
    algo_transport: ['Transport', '传输']
  }
  const labels = taskLabels[value]
  return labels ? t(labels[0], labels[1]) : value.replace(/^algo_/, '').replace(/_/g, ' ')
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function methodStatusLabel(status: 'verified' | 'testing') {
  return status === 'verified' ? t('Verified', '已验证') : t('Testing', '测试中')
}

function methodStatusType(status: 'verified' | 'testing') {
  return status === 'verified' ? 'success' : 'warning'
}

function dataTypeLabel(dataType: string) {
  if (dataType === 'number') return t('Number', '数值')
  if (dataType === 'integer') return t('Integer', '整数')
  if (dataType === 'text') return t('Text', '文本')
  return dataType
}

function rangeLabel(minimum: number | null, exclusiveMinimum: boolean) {
  if (minimum === null) return '—'
  return `${exclusiveMinimum ? '>' : '≥'} ${minimum}`
}
</script>

<template>
  <main class="online-workbench">
    <section class="online-page">
      <section class="page-heading">
        <div>
          <p class="eyebrow">GEOCHEMISTRY π ONLINE</p>
          <h1>{{ t('Chemical modeling', '化学建模') }}</h1>
          <p class="intro">
            {{
              t(
                'Select a model, upload an Excel or CSV file, and download the calculated result.',
                '选择模型，上传 Excel 或 CSV 文件，然后下载计算结果。'
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

      <el-card v-loading="loadingCatalog" class="calculation-card" shadow="never">
        <el-form label-position="top">
          <div class="form-grid">
            <el-form-item :label="t('Task', '任务')">
              <el-select
                v-model="selectedTask"
                :placeholder="t('Select a task', '选择任务')"
                :disabled="running"
              >
                <el-option
                  v-for="task in availableTasks"
                  :key="task.name"
                  :label="formatLabel(task.name)"
                  :value="task.name"
                />
              </el-select>
            </el-form-item>

            <el-form-item :label="t('Method', '方法')">
              <el-select
                v-model="selectedMethod"
                :placeholder="t('Select a method', '选择方法')"
                :disabled="running"
              >
                <el-option
                  v-for="method in availableMethods"
                  :key="method.name"
                  :label="chemicalMethodDescription(method.name, method.description)"
                  :value="method.name"
                >
                  <div class="method-option">
                    <span>{{ chemicalMethodDescription(method.name, method.description) }}</span>
                    <el-tag :type="methodStatusType(method.status)" size="small" effect="plain">
                      {{ methodStatusLabel(method.status) }}
                    </el-tag>
                  </div>
                </el-option>
              </el-select>
            </el-form-item>

            <el-form-item :label="t('Element', '元素')">
              <el-select
                v-model="selectedElement"
                :placeholder="t('Select an element', '选择元素')"
                :disabled="running"
              >
                <el-option
                  v-for="element in availableElements"
                  :key="element"
                  :label="element"
                  :value="element"
                />
              </el-select>
            </el-form-item>
          </div>

          <section v-if="currentMethod" class="method-guide">
            <div class="method-guide-heading">
              <div>
                <p class="guide-kicker">
                  {{ t('METHOD STATUS AND INPUT GUIDE', '算法状态与输入说明') }}
                </p>
                <h2>
                  {{ chemicalMethodDescription(currentMethod.name, currentMethod.description) }}
                </h2>
              </div>
              <el-tag :type="methodStatusType(currentMethod.status)" effect="dark">
                {{ methodStatusLabel(currentMethod.status) }}
              </el-tag>
            </div>

            <p class="status-message">
              {{ chemicalMethodStatus(currentMethod.name, currentMethod.status_message) }}
            </p>

            <div v-if="currentMethod.formula" class="formula-row">
              <span>{{ t('Formula', '计算公式') }}</span>
              <FormulaDisplay
                :method="currentMethod.name"
                :fallback="chemicalMethodFormula(currentMethod.name, currentMethod.formula)"
              />
            </div>

            <div v-if="currentMethod.input_columns.length" class="input-table-wrap">
              <el-table :data="currentMethod.input_columns" border size="small">
                <el-table-column prop="name" :label="t('Column', '列名')" min-width="70">
                  <template #default="scope">
                    <code>{{ scope.row.name }}</code>
                    <span v-if="scope.row.required" class="required-mark">{{
                      t('Required', '必填')
                    }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="t('Meaning', '含义')" min-width="96">
                  <template #default="scope">{{ apiText(scope.row.label) }}</template>
                </el-table-column>
                <el-table-column :label="t('Description', '说明')" min-width="150">
                  <template #default="scope">{{ apiText(scope.row.description) }}</template>
                </el-table-column>
                <el-table-column :label="t('Type', '类型')" min-width="56">
                  <template #default="scope">{{ dataTypeLabel(scope.row.data_type) }}</template>
                </el-table-column>
                <el-table-column :label="t('Valid range', '有效范围')" min-width="68">
                  <template #default="scope">
                    {{ rangeLabel(scope.row.minimum, scope.row.exclusive_minimum) }}
                  </template>
                </el-table-column>
                <el-table-column :label="t('Unit', '单位要求')" min-width="110">
                  <template #default="scope">{{ apiText(scope.row.unit) }}</template>
                </el-table-column>
                <el-table-column prop="example" :label="t('Example', '示例')" min-width="56" />
              </el-table>
            </div>

            <div v-if="currentMethod.input_columns.length" class="input-field-cards">
              <article v-for="column in currentMethod.input_columns" :key="column.name">
                <header>
                  <code>{{ column.name }}</code>
                  <span v-if="column.required" class="required-mark">{{
                    t('Required', '必填')
                  }}</span>
                </header>
                <dl>
                  <div>
                    <dt>{{ t('Meaning', '含义') }}</dt>
                    <dd>{{ apiText(column.label) }}</dd>
                  </div>
                  <div>
                    <dt>{{ t('Description', '说明') }}</dt>
                    <dd>{{ apiText(column.description) }}</dd>
                  </div>
                  <div>
                    <dt>{{ t('Type', '类型') }}</dt>
                    <dd>{{ dataTypeLabel(column.data_type) }}</dd>
                  </div>
                  <div>
                    <dt>{{ t('Valid range', '有效范围') }}</dt>
                    <dd>{{ rangeLabel(column.minimum, column.exclusive_minimum) }}</dd>
                  </div>
                  <div>
                    <dt>{{ t('Unit', '单位要求') }}</dt>
                    <dd>{{ apiText(column.unit) }}</dd>
                  </div>
                  <div>
                    <dt>{{ t('Example', '示例') }}</dt>
                    <dd class="mono">{{ column.example }}</dd>
                  </div>
                </dl>
              </article>
            </div>

            <el-alert
              v-else
              :title="
                t(
                  'Input-column and unit guidance for this method is being prepared.',
                  '该方法的输入列和单位说明正在整理。'
                )
              "
              type="info"
              :closable="false"
              show-icon
            />

            <el-alert
              v-if="!currentMethodIsVerified"
              :title="
                t(
                  'Methods under testing are shown for reference and cannot run until validation is complete.',
                  '测试中的方法只展示说明，完成验证前不能执行计算。'
                )
              "
              type="warning"
              :closable="false"
              show-icon
            />
          </section>

          <div v-if="requiredColumns.length && currentMethodIsVerified" class="column-hint">
            <strong>{{ t('Required data columns', '数据文件必填列') }}</strong>
            <code v-for="column in requiredColumns" :key="column">{{ column }}</code>
          </div>

          <el-form-item class="dataset-field" :label="t('Upload dataset', '上传数据集')">
            <label class="file-picker" :class="{ disabled: running || !currentMethodIsVerified }">
              <input
                type="file"
                accept=".xlsx,.csv"
                :disabled="running || !currentMethodIsVerified"
                @change="onFileChange"
              />
              <el-icon class="upload-icon"><UploadFilled /></el-icon>
              <span class="upload-copy">
                <strong>{{ t('Upload dataset', '上传数据集') }}</strong>
                <small>{{
                  t(
                    'Drag and drop an XLSX or CSV file here, or click to browse.',
                    '拖放 XLSX 或 CSV 文件到此处，或点击浏览。'
                  )
                }}</small>
                <small>{{ resourceLimitNote }}</small>
                <em>{{ datasetFile?.name || t('No file selected', '未选择文件') }}</em>
              </span>
            </label>
          </el-form-item>

          <el-alert
            v-if="errorMessage"
            class="message-block"
            :title="errorMessage"
            type="error"
            :closable="false"
            show-icon
          />

          <TaskProgress
            v-if="taskStatus && (running || inspectingDataset)"
            :task="taskStatus"
            :cancelling="cancellingTask"
            @cancel="cancelCurrentTask"
          />

          <div class="actions">
            <el-button
              type="primary"
              size="large"
              :loading="running"
              :disabled="!canRun"
              @click="submitJob"
            >
              {{
                running
                  ? t('Calculating…', '正在计算…')
                  : currentMethodIsVerified
                    ? t('Start calculation', '开始计算')
                    : t('This method is not available yet', '该方法暂不可计算')
              }}
            </el-button>
            <el-button v-if="serviceState === 'offline'" size="large" @click="loadPage">
              {{ t('Retry connection', '重新连接') }}
            </el-button>
          </div>
        </el-form>
      </el-card>

      <el-card v-if="artifacts.length" class="result-card" shadow="never">
        <template #header>
          <div class="result-heading">
            <div>
              <h2>{{ t('Calculation completed', '计算完成') }}</h2>
              <p>{{ t('Job ID', '任务 ID') }}: {{ jobId }}</p>
            </div>
            <el-tag type="success">{{ t('SUCCESS', '成功') }}</el-tag>
          </div>
        </template>

        <div v-for="artifact in artifacts" :key="artifact.download_url" class="artifact-row">
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
            {{ t('Download result', '下载结果') }}
          </el-button>
        </div>
      </el-card>

      <el-alert
        v-if="unavailableTasks.length"
        class="unavailable-note"
        type="warning"
        :closable="false"
        show-icon
      >
        <template #title>{{
          t('Some algorithms are temporarily unavailable', '部分算法暂时不可用')
        }}</template>
        <p v-for="task in unavailableTasks" :key="task.name">
          <strong>{{ formatLabel(task.name) }}:</strong> {{ apiText(task.error) }}
        </p>
      </el-alert>
    </section>

    <aside class="context-rail">
      <RunSummary
        :file-name="datasetFile?.name"
        :rows="datasetProfile?.summary.rows"
        :columns="datasetProfile?.summary.columns"
        :missing-cells="datasetProfile?.summary.missing_cells"
        :method="runSummaryMethod"
        :parameters="runSummaryParameters"
        :status="runSummaryStatus"
        :status-tone="runSummaryTone"
        :job-id="jobId || taskId"
        :software-version="softwareVersion"
      />
    </aside>
  </main>
</template>

<style lang="scss" scoped>
.online-page {
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

.calculation-card,
.result-card {
  border-color: #e2e8f0;
  border-radius: 12px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;

  :deep(.el-select) {
    width: 100%;
  }
}

.method-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.method-guide {
  margin-bottom: 22px;
  padding: 18px;
  border: 1px solid #dbe3ec;
  border-radius: 10px;
  background: #fbfdff;
}

.method-guide-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;

  h2 {
    margin: 2px 0 0;
    color: #1f2937;
    font-size: 19px;
    font-weight: 650;
  }

  .guide-kicker {
    margin: 0;
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
  }
}

.status-message {
  margin: 12px 0;
  color: #475569;
  line-height: 1.7;
}

.formula-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: start;
  gap: 12px;
  margin-bottom: 14px;

  > span {
    padding-top: 11px;
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
  }
}

.input-table-wrap {
  margin-bottom: 14px;
  overflow-x: auto;

  :deep(.el-table) {
    min-width: 650px;
  }

  code {
    color: #b9380b;
    font-weight: 650;
  }

  .required-mark {
    display: block;
    margin-top: 3px;
    color: #dc2626;
    font-size: 11px;
  }
}

.input-field-cards {
  display: none;
}

.input-notes {
  margin: 12px 0 0;
  padding-left: 20px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.8;

  & + .el-alert {
    margin-top: 12px;
  }
}

.column-hint {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0 0 22px;
  padding: 12px 14px;
  border-radius: 8px;
  color: #475569;
  background: #f8fafc;

  strong {
    margin-right: 4px;
  }

  code {
    padding: 2px 7px;
    border: 1px solid #dbe3ec;
    border-radius: 4px;
    color: #b9380b;
    background: #fff;
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

.result-card {
  margin-top: 24px;

  .result-heading,
  .artifact-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
  }

  h2 {
    color: #166534;
    font-size: 20px;
    font-weight: 650;
  }

  p,
  span {
    color: #64748b;
    font-size: 13px;
  }

  .artifact-row > div {
    display: flex;
    flex-direction: column;
  }
}

.unavailable-note {
  margin-top: 24px;

  p {
    margin-top: 4px;
  }
}

@media (max-width: 760px) {
  .online-page {
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

  .form-grid {
    grid-template-columns: 1fr;
    gap: 0;
  }

  .file-picker,
  .result-card .artifact-row {
    align-items: stretch;
    flex-direction: column;
  }

  .file-picker .file-button,
  .file-picker .file-name {
    min-height: 42px;
    justify-content: center;
  }
}

/* Alpine daylight theme: calm geochemical analysis with a fresh field-note palette. */
.online-workbench {
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

.online-page {
  width: 100%;
  max-width: 1180px;
  min-width: 0;
  margin: 0;
  padding: 30px 30px 56px;
  background: #f3f5f6;
  justify-self: center;
}

.online-page > *,
.page-heading > div,
.form-grid,
.form-grid > *,
.method-guide,
.formula-row,
.formula-row > *,
.input-field-cards,
.file-picker,
.upload-copy {
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
    font-weight: 650;
    letter-spacing: -0.035em;
  }

  .eyebrow {
    color: #d86149;
    font-size: 12px;
    letter-spacing: 0.14em;
  }

  .intro {
    max-width: 680px;
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
    background: #67d0a2;
    box-shadow: 0 0 0 4px rgb(103 208 162 / 14%);
  }
}

.calculation-card,
.result-card {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  border: 1px solid #dfe5e6;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 12px 30px rgb(31 56 62 / 6%);

  :deep(.el-card__body) {
    width: 100%;
    max-width: 100%;
    min-width: 0;
    padding: 22px;
  }
}

.form-grid {
  gap: 18px;

  :deep(.el-form-item),
  :deep(.el-form-item__content),
  :deep(.el-select) {
    min-width: 0;
    max-width: 100%;
  }

  :deep(.el-form-item) {
    margin-bottom: 0;
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
  border-radius: 5px;
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

.method-guide {
  margin: 2px 0 16px;
  padding: 20px;
  border-color: #c9dfdb;
  border-radius: 9px;
  background: #f4f8f7;
  box-shadow: none;
}

.method-guide-heading {
  h2 {
    color: #173f47;
    font-size: 22px;
  }

  .guide-kicker {
    color: #5c8588;
    letter-spacing: 0.12em;
  }

  :deep(.el-tag--success) {
    border-color: #9fd4ba;
    color: #287453;
    background: #edf8f2;
  }
}

.status-message {
  max-width: 720px;
  margin: 10px 0 14px;
  color: #5f797d;
  font-size: 15px;
  overflow-wrap: anywhere;
}

.formula-row {
  gap: 14px;

  > span {
    color: #55767a;
  }
}

.input-table-wrap {
  border: 1px solid #d5e5e2;
  border-radius: 3px;

  :deep(.el-table) {
    --el-table-bg-color: transparent;
    --el-table-tr-bg-color: transparent;
    --el-table-header-bg-color: #e8f3f1;
    --el-table-row-hover-bg-color: #f2f8f7;
    --el-table-border-color: #d5e5e2;
    --el-table-header-text-color: #4f6f74;
    --el-table-text-color: #294f56;
    min-width: 650px;
    background: transparent;
    font-size: 13px;
    font-variant-numeric: tabular-nums;

    &::before,
    .el-table__inner-wrapper::before {
      background: #d5e5e2;
    }

    th.el-table__cell,
    td.el-table__cell {
      padding: 7px 0;
      background: transparent;
    }

    .cell {
      padding-right: 6px;
      padding-left: 6px;
    }
  }

  code {
    color: #197e83;
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
  }

  .required-mark {
    color: #d86149;
  }
}

.input-notes {
  padding-left: 18px;
  color: #607c80;
  line-height: 1.7;
  list-style: disc;
}

.column-hint {
  gap: 9px;
  margin-bottom: 14px;
  color: #557479;
  background: #eaf5f2;

  code {
    border-color: #bcdad6;
    color: #197e83;
    background: #f8fcfb;
  }
}

.dataset-field {
  margin-top: 8px;
}

.file-picker {
  justify-content: center;
  gap: 18px;
  min-height: 112px;
  padding: 18px 24px;
  border: 1px dashed #86b8b5;
  border-radius: 7px;
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

  .upload-copy {
    display: grid;
    gap: 3px;
    min-width: 0;

    strong {
      color: #173f47;
      font-size: 16px;
      font-weight: 650;
    }

    small {
      color: #6b8589;
      font-size: 12px;
    }

    em {
      overflow: hidden;
      color: #197e83;
      font-size: 12px;
      font-style: normal;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }
}

.message-block,
.unavailable-note {
  border-color: #edc6bc;
  background: #fff4f1;
}

.actions {
  justify-content: flex-end;
  padding-top: 4px;

  :deep(.el-button--primary) {
    min-width: 188px;
    height: 46px;
    border-color: #ee6b52;
    border-radius: 5px;
    background: #d95f4b;
    font-weight: 650;

    &:hover,
    &:focus-visible {
      border-color: #f17d67;
      background: #e56a54;
    }

    &.is-disabled {
      border-color: #c8d6d4;
      background: #b8c8c6;
      opacity: 0.6;
    }
  }
}

.result-card {
  margin-top: 22px;
  padding: 18px;
  border: 1px solid #c7dfd8;
  border-radius: 7px;
  background: #fff;

  h2 {
    color: #287453;
  }

  p,
  span {
    color: #607c80;
  }
}

.context-rail {
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

.context-section {
  padding: 34px 26px 30px;
  border-bottom: 1px solid #d3e5e1;

  h2 {
    margin: 7px 0 12px;
    color: #173f47;
    line-height: 1.45;
    font-size: 16px;
    font-weight: 620;
  }
}

.context-kicker {
  margin: 0;
  color: #5c8588;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.13em;
}

.method-summary,
.method-tips {
  color: #607c80;
  line-height: 1.7;
  font-size: 12px;
}

.method-tips {
  margin-top: 24px;

  h3 {
    margin-bottom: 10px;
    color: #244c54;
    font-size: 13px;
    font-weight: 650;
  }

  ul {
    display: grid;
    gap: 9px;
    padding-left: 17px;
    list-style: disc;
  }
}

@media (max-width: 1360px) {
  .online-workbench {
    grid-template-columns: minmax(0, 1fr) 300px;
  }

  .online-page {
    padding-right: 26px;
    padding-left: 26px;
  }

  .context-section {
    padding-right: 22px;
    padding-left: 22px;
  }
}

@media (max-width: 1180px) {
  .online-workbench {
    grid-template-columns: minmax(0, 1fr);
  }

  .context-rail {
    position: static;
    grid-column: 1;
    display: block;
    height: auto;
    border-top: 1px solid rgb(151 208 214 / 22%);
    border-left: 0;
  }

  .context-section {
    border-right: 1px solid rgb(151 208 214 / 18%);
  }
}

@media (max-width: 820px) {
  .online-workbench {
    grid-template-columns: minmax(0, 1fr);
  }

  .online-page {
    width: 100%;
    padding: 28px 20px 42px;
  }

  .context-rail {
    grid-column: 1;
    grid-template-columns: 1fr;
  }

  .context-section {
    border-right: 0;
  }
}

@media (max-width: 560px) {
  .online-page {
    padding: 24px 16px 36px;
  }

  .form-grid {
    grid-template-columns: 1fr;
    gap: 0;
  }

  .page-heading h1 {
    font-size: 34px;
  }

  .method-guide {
    width: 100%;
    max-width: 100%;
    padding: 16px;
    overflow: hidden;
  }

  .method-guide-heading {
    flex-wrap: wrap;
  }

  .formula-row {
    grid-template-columns: minmax(0, 1fr);

    > span {
      padding-top: 0;
    }
  }

  .input-table-wrap {
    display: none;
  }

  .input-field-cards {
    display: grid;
    gap: 12px;
    margin-bottom: 14px;

    article {
      width: 100%;
      max-width: 100%;
      min-width: 0;
      padding: 14px;
      border: 1px solid #d8e3e2;
      border-radius: 7px;
      background: #fff;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid #e7ecec;
    }

    code,
    .mono {
      color: #197e83;
      font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
      font-variant-numeric: tabular-nums;
    }

    .required-mark {
      color: #b74735;
      font-size: 12px;
      font-weight: 650;
    }

    dl {
      display: grid;
      gap: 10px;
      margin: 12px 0 0;
    }

    dl > div {
      display: grid;
      grid-template-columns: minmax(82px, 0.38fr) minmax(0, 1fr);
      gap: 10px;
    }

    dt {
      color: #6a7f84;
      font-size: 13px;
      font-weight: 620;
    }

    dd {
      min-width: 0;
      margin: 0;
      overflow-wrap: anywhere;
      color: #294f56;
      font-size: 14px;
      line-height: 1.5;
    }
  }

  .file-picker {
    align-items: center;
    flex-direction: column;
    text-align: center;
  }

  .actions :deep(.el-button--primary) {
    width: 100%;
  }
}
</style>
