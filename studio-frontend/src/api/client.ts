const BASE_URL = '/api/v1'

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  // Runs
  getRuns: () => request<import('./types').Run[]>('/runs'),
  getRun: (id: string) => request<import('./types').RunDetail>(`/runs/${id}`),
  getRunMetrics: (id: string) => request<import('./types').Metrics>(`/runs/${id}/metrics`),
  getRunConfig: (id: string) => request<Record<string, unknown>>(`/runs/${id}/config`),
  getRunCheckpoints: (id: string) => request<import('./types').Checkpoint[]>(`/runs/${id}/checkpoints`),
  deleteRun: (id: string) => request<{ status: string }>(`/runs/${id}`, { method: 'DELETE' }),

  // Models
  getModels: () => request<import('./types').Model[]>('/models'),
  getSupportedArchitectures: () => request<string[]>('/models/supported'),

  // Datasets
  getDatasets: () => request<import('./types').Dataset[]>('/datasets'),
  getDataset: (fp: string) => request<import('./types').Dataset>(`/datasets/${fp}`),
  deleteDataset: (fp: string) => request<{ status: string }>(`/datasets/${fp}`, { method: 'DELETE' }),

  // Training
  startTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string; status: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  stopTraining: (trackId: string) =>
    request<{ status: string }>(`/training/${trackId}/stop`, { method: 'POST' }),
  getActiveTraining: () => request<import('./types').ActiveTraining[]>('/training/active'),

  // Inference
  generate: (body: Record<string, unknown>) =>
    request<import('./types').GenerationResult>('/inference/generate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getInferenceStatus: () => request<{ loaded_model: unknown }>('/inference/status'),
}

// V2 API client
const V2_BASE = '/api/v2'

async function requestV2<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${V2_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail || res.statusText)
  }
  return res.json()
}

export const apiV2 = {
  // Recipes
  getRecipes: () => requestV2<import('./types').Recipe[]>('/recipes'),
  getRecipe: (id: string) => requestV2<import('./types').Recipe>(`/recipes/${id}`),
  resolveRecipe: (id: string, body: Record<string, unknown>) =>
    requestV2<Record<string, unknown>>(`/recipes/${id}/resolve`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // Memory
  getHardware: () => requestV2<import('./types').HardwareInfo>('/memory/hardware'),
  estimateMemory: (body: Record<string, unknown>) =>
    requestV2<import('./types').MemoryEstimateResult>('/memory/estimate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getCompatibleModels: () => requestV2<import('./types').CompatibleModel[]>('/memory/compatible-models'),

  // Queue
  getQueue: () => requestV2<import('./types').QueueJob[]>('/queue'),
  submitJob: (config: Record<string, unknown>) =>
    requestV2<import('./types').QueueJob>('/queue/submit', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  cancelJob: (id: string) =>
    requestV2<import('./types').QueueJob>(`/queue/${id}/cancel`, { method: 'POST' }),
  promoteJob: (id: string) =>
    requestV2<import('./types').QueueJob>(`/queue/${id}/promote`, { method: 'POST' }),
  getQueueStats: () => requestV2<import('./types').QueueStats>('/queue/stats'),

  // Models
  getModelLibrary: () => requestV2<import('./types').LibraryModel[]>('/models/library'),

  // Schema
  getConfigSchema: () => requestV2<Record<string, unknown>>('/schema'),
  getTrainingParamsSchema: () => requestV2<Record<string, unknown>>('/schema/training-params'),

  // Data Library
  getDataCatalog: () => requestV2<import('./types').CatalogDataset[]>('/data/catalog'),
  getDownloadedDatasets: () => requestV2<import('./types').DownloadedDataset[]>('/data/datasets'),
  downloadDataset: (catalogId: string, maxSamples?: number) =>
    requestV2<import('./types').DownloadedDataset>('/data/download', {
      method: 'POST',
      body: JSON.stringify({ catalog_id: catalogId, max_samples: maxSamples }),
    }),
  getDatasetDetail: (name: string) => requestV2<import('./types').DownloadedDataset>(`/data/datasets/${name}`),
  getDatasetSamples: (name: string, n: number = 5) =>
    requestV2<Record<string, unknown>[]>(`/data/datasets/${name}/samples?n=${n}`),
  deleteDownloadedDataset: (name: string) =>
    requestV2<{ status: string }>(`/data/datasets/${name}`, { method: 'DELETE' }),
}

export { ApiError }
