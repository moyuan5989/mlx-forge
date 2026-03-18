// API response types matching the MLX Forge Studio backend

export interface Run {
  id: string
  path: string
  status: 'completed' | 'running' | 'stopped' | 'unknown'
  model: string | null
  current_step: number
  num_iters: number
  latest_train_loss: number | null
  latest_val_loss: number | null
}

export interface RunDetail extends Run {
  config?: Record<string, unknown>
  manifest?: Record<string, unknown>
  environment?: Record<string, unknown>
}

export interface TrainMetric {
  event: 'train'
  step: number
  train_loss: number
  learning_rate: number
  tokens_per_second: number
  trained_tokens: number
  peak_memory_gb: number
  timestamp: string
}

export interface EvalMetric {
  event: 'eval'
  step: number
  val_loss: number
  timestamp: string
}

export interface Metrics {
  train: TrainMetric[]
  eval: EvalMetric[]
}

export interface CheckpointState {
  schema_version: number
  step: number
  epoch: number
  trained_tokens: number
  best_val_loss: number
  learning_rate: number
  rng_seed: number
}

export interface Checkpoint {
  name: string
  path: string
  is_best: boolean
  state?: CheckpointState
}

export interface Model {
  id: string
  model_id: string
  path: string
  architecture: string
  supported: boolean
  size_gb: number | null
  config: Record<string, unknown>
}

export interface Dataset {
  fingerprint: string
  path: string
  num_samples: number
  total_tokens: number
  min_length: number
  mean_length: number
  max_length: number
  format: string
  model?: string
  template_hash?: string
  tokenizer_hash?: string
  data_hash?: string
  source_path?: string
  model_id?: string
  created_at?: string
}

export interface ActiveTraining {
  track_id: string
  run_id: string
  pid: number
  started_at: string
  config?: Record<string, unknown>
}

export interface GenerationResult {
  text: string
  num_tokens: number
  tokens_per_second: number
  finish_reason: 'stop' | 'length'
}

// WebSocket message types
export interface WsMetricMessage {
  type: 'metric'
  data: TrainMetric | EvalMetric
}

export interface WsTokenMessage {
  type: 'token'
  text: string
}

export interface WsDoneMessage {
  type: 'done'
  stats: { num_tokens: number }
}

export interface WsErrorMessage {
  type: 'error'
  detail: string
}

export interface WsStoppedMessage {
  type: 'stopped'
}

export type WsTrainingMessage = WsMetricMessage | WsErrorMessage | WsStoppedMessage
export type WsInferenceMessage = WsTokenMessage | WsDoneMessage | WsErrorMessage

// V2 Types

export interface Recipe {
  id: string
  name: string
  description: string
  category: 'sft' | 'dpo'
  training_type: 'sft' | 'dpo'
  data_format: 'chat' | 'completions' | 'text' | 'preference'
  recommended_models: string[]
  config_template: Record<string, unknown>
  auto_rules: string[]
  icon: string
}

export interface HardwareInfo {
  total_memory_gb: number
  training_budget_gb: number
  chip_name: string
}

export interface MemoryBarSegment {
  label: string
  gb: number
  color: string
}

export interface MemoryEstimateResult {
  base_weights_gb: number
  lora_overhead_gb: number
  optimizer_state_gb: number
  peak_activations_gb: number
  mlx_overhead_gb: number
  total_gb: number
  budget_gb: number
  fits: boolean
  bar_segments: MemoryBarSegment[]
}

export interface CompatibleModel {
  model_id: string
  display_name: string
  num_params_b: number
  fp16: { total_gb: number; fits: boolean }
  qlora_4bit: { total_gb: number; fits: boolean }
  fit_level: 'comfortable' | 'tight' | 'unlikely'
  downloaded: boolean
}

export interface LibraryModel {
  model_id: string
  display_name: string
  num_params_b: number
  architecture: string
  hidden_dim: number
  num_layers: number
  vocab_size: number
  downloaded: boolean
  fp16: { total_gb: number; fits: boolean }
  qlora_4bit: { total_gb: number; fits: boolean }
  fit_level: 'comfortable' | 'tight' | 'unlikely'
}

export interface Adapter {
  run_id: string
  model: string
  status: string
  checkpoint: string
  path: string
  label: string
}

export interface QueueJob {
  id: string
  config: Record<string, unknown>
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: number
  started_at: number | null
  completed_at: number | null
  run_id: string | null
  track_id: string | null
  error: string | null
  position: number
}

export interface QueueStats {
  queued: number
  running: number
  completed: number
  failed: number
  cancelled: number
  max_concurrent: number
}

// ── Data Library Types ──

export interface CatalogDataset {
  id: string
  source: string
  display_name: string
  category: string
  format: string
  description: string
  license: string
  total_samples: number
  avg_tokens: number
  tags: string[]
  downloaded: boolean
}

export interface DownloadedDataset {
  id: string
  display_name: string
  format: string
  num_samples: number
  source: string
  category: string
  license: string
  tags: string[]
  origin: string
  path: string
  downloaded: boolean
}
