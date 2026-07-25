export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export interface HealthResponse {
  status: string
  service: string
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
  return parseResponse<HealthResponse>(response)
}

export async function getCatalog(): Promise<CatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chemical-modeling/catalog`)
  return parseResponse<CatalogResponse>(response)
}

export async function runChemicalModel(
  task: string,
  method: string,
  element: string,
  dataset: File
): Promise<RunResponse> {
  const form = new FormData()
  form.append('task', task)
  form.append('method', method)
  form.append('element', element)
  form.append('dataset', dataset)

  const response = await fetch(`${API_BASE_URL}/api/chemical-modeling/run`, {
    method: 'POST',
    body: form
  })
  return parseResponse<RunResponse>(response)
}

export function artifactUrl(downloadUrl: string): string {
  return `${API_BASE_URL}${downloadUrl}`
}
