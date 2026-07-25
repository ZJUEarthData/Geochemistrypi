<script lang="ts" setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import {
  artifactUrl,
  getCatalog,
  getHealth,
  runChemicalModel,
  type ArtifactResponse,
  type TaskCatalogItem
} from '@/api/online'

type ServiceState = 'checking' | 'online' | 'offline'

const serviceState = ref<ServiceState>('checking')
const loadingCatalog = ref(true)
const running = ref(false)
const tasks = ref<TaskCatalogItem[]>([])
const selectedTask = ref('')
const selectedMethod = ref('')
const selectedElement = ref('')
const datasetFile = ref<File | null>(null)
const artifacts = ref<ArtifactResponse[]>([])
const jobId = ref('')
const errorMessage = ref('')

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
    !running.value
)

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
    await getHealth()
    serviceState.value = 'online'
    const catalog = await getCatalog()
    tasks.value = catalog.tasks
    selectedTask.value =
      availableTasks.value.find((task) => task.methods.some((method) => method.status === 'verified'))?.name ||
      availableTasks.value[0]?.name ||
      ''
  } catch (error) {
    serviceState.value = 'offline'
    errorMessage.value = error instanceof Error ? error.message : 'Cannot connect to the backend service.'
  } finally {
    loadingCatalog.value = false
  }
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] || null
  clearResult()
  errorMessage.value = ''

  if (file && !file.name.toLowerCase().endsWith('.xlsx')) {
    datasetFile.value = null
    input.value = ''
    errorMessage.value = 'The first Online version only supports .xlsx files.'
    return
  }
  datasetFile.value = file
}

async function submitJob() {
  if (!canRun.value || !datasetFile.value) return

  running.value = true
  clearResult()
  errorMessage.value = ''
  try {
    const result = await runChemicalModel(
      selectedTask.value,
      selectedMethod.value,
      selectedElement.value,
      datasetFile.value
    )
    artifacts.value = result.artifacts
    jobId.value = result.job_id
    ElMessage.success('Calculation completed')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Calculation failed.'
    ElMessage.error('Calculation failed')
  } finally {
    running.value = false
  }
}

function clearResult() {
  artifacts.value = []
  jobId.value = ''
}

function formatLabel(value: string) {
  return value.replace(/^algo_/, '').replace(/_/g, ' ')
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function methodStatusLabel(status: 'verified' | 'testing') {
  return status === 'verified' ? '已验证' : '测试中'
}

function methodStatusType(status: 'verified' | 'testing') {
  return status === 'verified' ? 'success' : 'warning'
}

function dataTypeLabel(dataType: string) {
  if (dataType === 'number') return '数值'
  if (dataType === 'integer') return '整数'
  return dataType
}

function rangeLabel(minimum: number | null, exclusiveMinimum: boolean) {
  if (minimum === null) return '—'
  return `${exclusiveMinimum ? '>' : '≥'} ${minimum}`
}
</script>

<template>
  <main class="online-page">
    <section class="page-heading">
      <div>
        <p class="eyebrow">GEOCHEMISTRY π ONLINE</p>
        <h1>Chemical modeling</h1>
        <p class="intro">Select a model, upload an Excel workbook, and download the calculated result.</p>
      </div>
      <div class="service-status" :class="serviceState">
        <span class="status-dot"></span>
        <span v-if="serviceState === 'checking'">Checking service</span>
        <span v-else-if="serviceState === 'online'">Backend online</span>
        <span v-else>Backend offline</span>
      </div>
    </section>

    <el-card v-loading="loadingCatalog" class="calculation-card" shadow="never">
      <el-form label-position="top">
        <div class="form-grid">
          <el-form-item label="Task">
            <el-select v-model="selectedTask" placeholder="Select a task" :disabled="running">
              <el-option
                v-for="task in availableTasks"
                :key="task.name"
                :label="formatLabel(task.name)"
                :value="task.name"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="Method">
            <el-select v-model="selectedMethod" placeholder="Select a method" :disabled="running">
              <el-option
                v-for="method in availableMethods"
                :key="method.name"
                :label="method.description || formatLabel(method.name)"
                :value="method.name"
              >
                <div class="method-option">
                  <span>{{ method.description || formatLabel(method.name) }}</span>
                  <el-tag :type="methodStatusType(method.status)" size="small" effect="plain">
                    {{ methodStatusLabel(method.status) }}
                  </el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="Element">
            <el-select v-model="selectedElement" placeholder="Select an element" :disabled="running">
              <el-option v-for="element in availableElements" :key="element" :label="element" :value="element" />
            </el-select>
          </el-form-item>
        </div>

        <section v-if="currentMethod" class="method-guide">
          <div class="method-guide-heading">
            <div>
              <p class="guide-kicker">算法状态与输入说明</p>
              <h2>{{ currentMethod.description || formatLabel(currentMethod.name) }}</h2>
            </div>
            <el-tag :type="methodStatusType(currentMethod.status)" effect="dark">
              {{ methodStatusLabel(currentMethod.status) }}
            </el-tag>
          </div>

          <p class="status-message">{{ currentMethod.status_message }}</p>

          <div v-if="currentMethod.formula" class="formula-row">
            <span>计算公式</span>
            <code>{{ currentMethod.formula }}</code>
          </div>

          <div v-if="currentMethod.input_columns.length" class="input-table-wrap">
            <el-table :data="currentMethod.input_columns" border size="small">
              <el-table-column prop="name" label="列名" min-width="90">
                <template #default="scope">
                  <code>{{ scope.row.name }}</code>
                  <span v-if="scope.row.required" class="required-mark">必填</span>
                </template>
              </el-table-column>
              <el-table-column prop="label" label="含义" min-width="130" />
              <el-table-column prop="description" label="说明" min-width="220" />
              <el-table-column label="类型" min-width="75">
                <template #default="scope">{{ dataTypeLabel(scope.row.data_type) }}</template>
              </el-table-column>
              <el-table-column label="有效范围" min-width="90">
                <template #default="scope">
                  {{ rangeLabel(scope.row.minimum, scope.row.exclusive_minimum) }}
                </template>
              </el-table-column>
              <el-table-column prop="unit" label="单位要求" min-width="180" />
              <el-table-column prop="example" label="示例" min-width="80" />
            </el-table>
          </div>

          <el-alert
            v-else
            title="该方法的输入列和单位说明正在整理。"
            type="info"
            :closable="false"
            show-icon
          />

          <ul v-if="currentMethod.input_notes.length" class="input-notes">
            <li v-for="note in currentMethod.input_notes" :key="note">{{ note }}</li>
          </ul>

          <el-alert
            v-if="!currentMethodIsVerified"
            title="测试中的方法只展示说明，完成验证前不能执行计算。"
            type="warning"
            :closable="false"
            show-icon
          />
        </section>

        <div v-if="requiredColumns.length && currentMethodIsVerified" class="column-hint">
          <strong>Excel 必填列</strong>
          <code v-for="column in requiredColumns" :key="column">{{ column }}</code>
        </div>

        <el-form-item label="Dataset (.xlsx)">
          <label class="file-picker" :class="{ disabled: running || !currentMethodIsVerified }">
            <input
              type="file"
              accept=".xlsx"
              :disabled="running || !currentMethodIsVerified"
              @change="onFileChange"
            />
            <span class="file-button">Choose Excel file</span>
            <span class="file-name">{{ datasetFile?.name || 'No file selected' }}</span>
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

        <div class="actions">
          <el-button type="primary" size="large" :loading="running" :disabled="!canRun" @click="submitJob">
            {{ running ? 'Calculating…' : currentMethodIsVerified ? 'Start calculation' : '该方法暂不可计算' }}
          </el-button>
          <el-button v-if="serviceState === 'offline'" size="large" @click="loadPage">Retry connection</el-button>
        </div>
      </el-form>
    </el-card>

    <el-card v-if="artifacts.length" class="result-card" shadow="never">
      <template #header>
        <div class="result-heading">
          <div>
            <h2>Calculation completed</h2>
            <p>Job ID: {{ jobId }}</p>
          </div>
          <el-tag type="success">SUCCESS</el-tag>
        </div>
      </template>

      <div v-for="artifact in artifacts" :key="artifact.download_url" class="artifact-row">
        <div>
          <strong>{{ artifact.name }}</strong>
          <span>{{ formatBytes(artifact.size_bytes) }}</span>
        </div>
        <el-button type="success" plain tag="a" :href="artifactUrl(artifact.download_url)" download>
          Download result
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
      <template #title>Some algorithms are temporarily unavailable</template>
      <p v-for="task in unavailableTasks" :key="task.name">
        <strong>{{ formatLabel(task.name) }}:</strong> {{ task.error }}
      </p>
    </el-alert>
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
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;

  span {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
  }

  code {
    padding: 5px 9px;
    border-radius: 5px;
    color: #9a3412;
    background: #fff3ed;
  }
}

.input-table-wrap {
  margin-bottom: 14px;
  overflow-x: auto;

  :deep(.el-table) {
    min-width: 850px;
  }

  code {
    color: #b9380b;
    font-weight: 650;
  }

  .required-mark {
    margin-left: 6px;
    color: #dc2626;
    font-size: 11px;
  }
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
</style>
