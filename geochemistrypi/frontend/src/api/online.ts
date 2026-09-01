export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
export const DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

export const FRONTEND_IDENTITY = {
  instanceId: __GEOCHEMISTRYPI_INSTANCE_ID__,
  sourceRevision: __GEOCHEMISTRYPI_SOURCE_REVISION__,
  buildId: __GEOCHEMISTRYPI_BUILD_ID__
}

export interface HealthResponse {
  status: string
  service: string
  version: string
  instance_id: string
  source_revision: string
  build_id: string
  max_upload_bytes: number
  task_timeout_seconds: number
  max_concurrent_tasks: number
}

export type MethodStatus = 'verified' | 'testing'

export interface InputColumnItem {
  name: string
  label: string
  description: string
  data_type: string
  unit: string
  example: number | string
  required: boolean
  minimum: number | null
  exclusive_minimum: boolean
}

export interface MethodCatalogItem {
  name: string
  description: string
  elements: string[]
  status: MethodStatus
  status_message: string
  formula: string | null
  input_columns: InputColumnItem[]
  input_notes: string[]
  required_columns: string[]
}

export interface TaskCatalogItem {
  name: string
  available: boolean
  methods: MethodCatalogItem[]
  error: string | null
}

export interface CatalogResponse {
  tasks: TaskCatalogItem[]
}

export interface ArtifactResponse {
  name: string
  download_url: string
  size_bytes: number
}

export interface RunResponse {
  job_id: string
  status: string
  message: string
  artifacts: ArtifactResponse[]
}

export type OnlineTaskState =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'timed_out'
  | 'cancelled'

export interface OnlineTaskStatus {
  task_id: string
  label: string
  status: OnlineTaskState
  progress: number
  queue_position: number | null
  submitted_at: string
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number
  timeout_seconds: number
  cancellable: boolean
  message: string
  error: string | null
}

export function createTaskId(): string {
  return crypto.randomUUID()
}

export function taskHeaders(taskId?: string): HeadersInit | undefined {
  return taskId ? { 'X-Task-ID': taskId } : undefined
}

export async function getTaskStatus(taskId: string): Promise<OnlineTaskStatus> {
  const response = await fetch(`${API_BASE_URL}/api/tasks/${taskId}`)
  return parseResponse<OnlineTaskStatus>(response)
}

export async function cancelTask(taskId: string): Promise<OnlineTaskStatus> {
  const response = await fetch(`${API_BASE_URL}/api/tasks/${taskId}/cancel`, { method: 'POST' })
  return parseResponse<OnlineTaskStatus>(response)
}

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string' ? detail : JSON.stringify(detail || payload || response.statusText)
    throw new Error(message)
  }
  return payload as T
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`)
  const health = await parseResponse<HealthResponse>(response)
  if (
    health.instance_id !== FRONTEND_IDENTITY.instanceId ||
    health.build_id !== FRONTEND_IDENTITY.buildId
  ) {
    throw new Error(
      'Frontend/backend version mismatch. Run start-online.cmd again to start the matching services.'
    )
  }
  return health
}

export async function getCatalog(): Promise<CatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chemical-modeling/catalog`)
  return parseResponse<CatalogResponse>(response)
}

export async function runChemicalModel(
  task: string,
  method: string,
  element: string,
  dataset: File,
  taskId?: string
): Promise<RunResponse> {
  const form = new FormData()
  form.append('task', task)
  form.append('method', method)
  form.append('element', element)
  form.append('dataset', dataset)

  const response = await fetch(`${API_BASE_URL}/api/chemical-modeling/run`, {
    method: 'POST',
    headers: taskHeaders(taskId),
    body: form
  })
  return parseResponse<RunResponse>(response)
}

export function artifactUrl(downloadUrl: string): string {
  return `${API_BASE_URL}${downloadUrl}`
}
