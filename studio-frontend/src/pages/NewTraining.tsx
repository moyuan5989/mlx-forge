import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  ChevronRight,
  ChevronLeft,
  Play,
  Download,
  AlertTriangle,
} from 'lucide-react'
import { cn, formatMemory } from '../lib/utils'
import { useCompatibleModels, useMemoryEstimate, useHardware } from '../hooks/useMemory'
import { useSubmitJob } from '../hooks/useQueue'
import { useDownloadedDatasets } from '../hooks/useDatasets'
import MemoryBar from '../components/shared/MemoryBar'
import type { CompatibleModel, MemoryEstimateResult, DownloadedDataset } from '../api/types'

const STEPS = ['Model', 'Data', 'Config', 'Review'] as const

const TRAINING_TYPES = [
  { value: 'sft', label: 'SFT', description: 'Supervised fine-tuning on chat, instruction, or text data' },
  { value: 'dpo', label: 'DPO', description: 'Direct Preference Optimization with chosen/rejected pairs' },
  { value: 'mlm', label: 'MLM', description: 'Masked Language Modeling for encoder models (BERT, RoBERTa)' },
  { value: 'seq2seq', label: 'Seq2Seq', description: 'Sequence-to-sequence for encoder-decoder models (T5, BART)' },
  { value: 'grpo', label: 'GRPO', description: 'Group Relative Policy Optimization with reward functions' },
  { value: 'orpo', label: 'ORPO', description: 'Odds Ratio Preference Optimization' },
  { value: 'kto', label: 'KTO', description: 'Kahneman-Tversky Optimization with binary feedback' },
  { value: 'simpo', label: 'SimPO', description: 'Simple Preference Optimization (reference-free)' },
] as const

type TrainingType = typeof TRAINING_TYPES[number]['value']

const DATA_FORMAT_HINTS: Record<TrainingType, string> = {
  sft: 'JSONL with "messages" (chat), "prompt"+"completion", or "text" field',
  dpo: 'JSONL with "chosen" and "rejected" fields (arrays of messages)',
  mlm: 'JSONL with "text" field — masking is applied automatically during training',
  seq2seq: 'JSONL with "input" and "target" fields',
  grpo: 'JSONL with "messages" field (chat format)',
  orpo: 'JSONL with "chosen" and "rejected" fields (arrays of messages)',
  kto: 'JSONL with "text" and "label" (0 or 1) fields',
  simpo: 'JSONL with "chosen" and "rejected" fields (arrays of messages)',
}

const DEFAULT_CONFIGS: Record<TrainingType, Record<string, unknown>> = {
  sft: { learning_rate: 2e-5, batch_size: 4, num_iters: 1000 },
  dpo: { learning_rate: 5e-6, batch_size: 2, num_iters: 500, dpo_beta: 0.1 },
  mlm: { learning_rate: 5e-5, batch_size: 8, num_iters: 1000, mlm_probability: 0.15 },
  seq2seq: { learning_rate: 3e-5, batch_size: 4, num_iters: 1000 },
  grpo: { learning_rate: 1e-5, batch_size: 2, num_iters: 500 },
  orpo: { learning_rate: 5e-6, batch_size: 2, num_iters: 500 },
  kto: { learning_rate: 5e-6, batch_size: 4, num_iters: 500 },
  simpo: { learning_rate: 5e-6, batch_size: 2, num_iters: 500 },
}

export default function NewTraining() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [currentStep, setCurrentStep] = useState(0)

  // Wizard state
  const [selectedModel, setSelectedModel] = useState('')
  const [trainingType, setTrainingType] = useState<TrainingType>('sft')
  const [dataSource, setDataSource] = useState<'downloaded' | 'custom'>('downloaded')
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [trainPath, setTrainPath] = useState('')
  const [validPath, setValidPath] = useState('')
  const [advancedMode, setAdvancedMode] = useState(false)
  const [configOverrides, setConfigOverrides] = useState<Record<string, unknown>>({})

  // Pre-select model from ?model= query param
  useEffect(() => {
    const modelParam = searchParams.get('model')
    if (modelParam && !selectedModel) {
      setSelectedModel(modelParam)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Data
  const { data: models } = useCompatibleModels()
  const { data: hardware } = useHardware()
  const { data: downloadedDatasets } = useDownloadedDatasets()
  const memEstimate = useMemoryEstimate()
  const submitJob = useSubmitJob()

  // Re-estimate memory when model or config changes
  const defaults = DEFAULT_CONFIGS[trainingType]
  const memBatchSize = (configOverrides['batch_size'] as number) || (defaults.batch_size as number) || 4
  const memGradCkpt = (configOverrides['gradient_checkpointing'] as boolean) || false
  const memQuantBits = configOverrides['quantization_bits'] as number | undefined

  useEffect(() => {
    if (selectedModel) {
      memEstimate.mutate({
        model_id: selectedModel,
        quantization_bits: memQuantBits,
        lora_rank: 16,
        batch_size: memBatchSize,
        max_seq_length: 2048,
        gradient_checkpointing: memGradCkpt,
      })
    }
  }, [selectedModel, memBatchSize, memGradCkpt, memQuantBits]) // eslint-disable-line react-hooks/exhaustive-deps

  const canNext = () => {
    switch (currentStep) {
      case 0: return selectedModel !== ''
      case 1: return dataSource === 'downloaded' ? selectedDataset !== '' : trainPath !== '' && validPath !== ''
      case 2: return true
      case 3: return true
      default: return false
    }
  }

  const buildConfig = (): Record<string, unknown> => {
    let resolvedTrainPath = trainPath
    let resolvedValidPath = validPath
    if (dataSource === 'downloaded' && selectedDataset) {
      const ds = downloadedDatasets?.find((d) => d.id === selectedDataset)
      if (ds) {
        resolvedTrainPath = ds.path
        resolvedValidPath = ds.path
      }
    }

    return {
      schema_version: 1,
      model: { path: selectedModel },
      adapter: {
        method: 'lora',
        preset: 'attention-qv',
        rank: 16,
        scale: 20.0,
        ...(memQuantBits ? {} : {}),
      },
      data: {
        train: resolvedTrainPath,
        valid: resolvedValidPath,
        max_seq_length: 2048,
      },
      training: {
        training_type: trainingType,
        ...defaults,
        ...configOverrides,
      },
    }
  }

  const handleSubmit = async () => {
    try {
      const config = buildConfig()
      await submitJob.mutateAsync(config)
      navigate('/experiments')
    } catch {
      // Error handled by mutation state
    }
  }

  const handleExportConfig = () => {
    const config = buildConfig()
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'mlxforge-config.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-heading">New Training</h2>
        <p className="text-sm text-caption mt-1">
          Set up and start a new fine-tuning run
        </p>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-2">
        {STEPS.map((step, i) => (
          <div key={step} className="flex items-center">
            <button
              onClick={() => i < currentStep && setCurrentStep(i)}
              className={cn(
                'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-colors',
                i === currentStep
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                  : i < currentStep
                    ? 'text-body hover:text-label cursor-pointer'
                    : 'text-muted cursor-default'
              )}
            >
              <span className={cn(
                'h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold',
                i === currentStep ? 'bg-indigo-500 text-white' :
                i < currentStep ? 'bg-surface-muted text-label' :
                'bg-surface-input text-muted'
              )}>
                {i + 1}
              </span>
              {step}
            </button>
            {i < STEPS.length - 1 && (
              <ChevronRight className="h-3 w-3 text-default mx-1" />
            )}
          </div>
        ))}
      </div>

      {/* Step content */}
      <div className="rounded-lg border border-subtle bg-surface-overlay p-6 min-h-[400px]">
        {currentStep === 0 && (
          <StepModel
            models={models || []}
            selected={selectedModel}
            onSelect={setSelectedModel}
            memoryEstimate={memEstimate.data || null}
            hardware={hardware || null}
          />
        )}
        {currentStep === 1 && (
          <StepData
            dataSource={dataSource}
            onDataSourceChange={setDataSource}
            downloadedDatasets={downloadedDatasets || []}
            selectedDataset={selectedDataset}
            onDatasetSelect={setSelectedDataset}
            trainPath={trainPath}
            validPath={validPath}
            onTrainChange={setTrainPath}
            onValidChange={setValidPath}
            trainingType={trainingType}
          />
        )}
        {currentStep === 2 && (
          <StepConfig
            trainingType={trainingType}
            onTrainingTypeChange={setTrainingType}
            defaults={defaults}
            advanced={advancedMode}
            onAdvancedToggle={setAdvancedMode}
            overrides={configOverrides}
            onOverridesChange={setConfigOverrides}
            memoryEstimate={memEstimate.data || null}
          />
        )}
        {currentStep === 3 && (
          <StepReview
            config={buildConfig()}
            memoryEstimate={memEstimate.data || null}
            hardware={hardware || null}
          />
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
          disabled={currentStep === 0}
          className="flex items-center gap-1 px-4 py-2 text-sm text-body hover:text-label disabled:opacity-30 disabled:cursor-default"
        >
          <ChevronLeft className="h-4 w-4" /> Back
        </button>
        <div className="flex gap-2">
          {currentStep === STEPS.length - 1 ? (
            <>
              <button
                onClick={handleExportConfig}
                className="flex items-center gap-1 rounded-md border border-default px-4 py-2 text-sm text-label hover:bg-surface-hover"
              >
                <Download className="h-4 w-4" /> Export Config
              </button>
              <button
                onClick={handleSubmit}
                disabled={submitJob.isPending}
                className="flex items-center gap-1 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {submitJob.isPending ? 'Starting...' : 'Start Training'}
              </button>
            </>
          ) : (
            <button
              onClick={() => setCurrentStep((s) => Math.min(STEPS.length - 1, s + 1))}
              disabled={!canNext()}
              className="flex items-center gap-1 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-default"
            >
              Next <ChevronRight className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {submitJob.isError && (
        <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 flex items-center gap-2 text-sm text-red-400">
          <AlertTriangle className="h-4 w-4 flex-shrink-0" />
          {(submitJob.error as Error).message || 'Failed to start training'}
        </div>
      )}
    </div>
  )
}

// Step 1: Select Model
function StepModel({ models, selected, onSelect, memoryEstimate, hardware }: {
  models: CompatibleModel[]
  selected: string
  onSelect: (m: string) => void
  memoryEstimate: MemoryEstimateResult | null
  hardware: { total_memory_gb: number; training_budget_gb: number; chip_name: string } | null
}) {
  const sortedModels = [...models].sort((a, b) => {
    if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1
    return a.num_params_b - b.num_params_b
  })

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-heading">Select a Model</h3>
        {hardware && (
          <p className="text-sm text-caption">
            {hardware.chip_name} - {formatMemory(hardware.total_memory_gb)} unified memory,{' '}
            {formatMemory(hardware.training_budget_gb)} available for training
            <span className="text-muted"> (75% — the rest is reserved for macOS)</span>
          </p>
        )}
      </div>

      <div className="space-y-2 max-h-[280px] overflow-y-auto">
        {sortedModels.map((model) => {
          const isSelected = selected === model.model_id
          const canFit = model.fp16.fits || model.qlora_4bit.fits

          return (
            <button
              key={model.model_id}
              onClick={() => canFit && onSelect(model.model_id)}
              disabled={!canFit}
              className={cn(
                'w-full text-left rounded-lg border p-3 transition-all',
                isSelected
                  ? 'border-indigo-500 bg-indigo-500/10'
                  : canFit
                    ? 'border-subtle bg-surface-card hover:border-default'
                    : 'border-subtle/50 bg-surface-overlay opacity-40 cursor-not-allowed'
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={cn('font-medium text-sm', isSelected ? 'text-indigo-300' : 'text-label')}>
                    {model.display_name}
                  </span>
                  {model.downloaded && (
                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-blue-500/20 text-blue-400">
                      Downloaded
                    </span>
                  )}
                  {model.fit_level === 'comfortable' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-emerald-500/20 text-emerald-400">
                      Good fit
                    </span>
                  )}
                  {model.fit_level === 'tight' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-amber-500/20 text-amber-400">
                      Tight fit
                    </span>
                  )}
                </div>
                <span className="text-xs text-caption">
                  {model.num_params_b}B params
                </span>
              </div>
              <div className="flex items-center gap-4 mt-1 text-xs text-caption">
                <span>FP16: {model.fp16.total_gb.toFixed(1)} GB {model.fp16.fits ? '' : '(OOM)'}</span>
                <span>QLoRA 4-bit: {model.qlora_4bit.total_gb.toFixed(1)} GB</span>
              </div>
            </button>
          )
        })}
      </div>

      {memoryEstimate && (
        <div className="pt-2">
          <MemoryBar
            segments={memoryEstimate.bar_segments}
            totalGb={memoryEstimate.total_gb}
            budgetGb={memoryEstimate.budget_gb}
            fits={memoryEstimate.fits}
          />
        </div>
      )}
    </div>
  )
}

// Step 2: Prepare Data
function StepData({ dataSource, onDataSourceChange, downloadedDatasets, selectedDataset, onDatasetSelect, trainPath, validPath, onTrainChange, onValidChange, trainingType }: {
  dataSource: 'downloaded' | 'custom'
  onDataSourceChange: (s: 'downloaded' | 'custom') => void
  downloadedDatasets: DownloadedDataset[]
  selectedDataset: string
  onDatasetSelect: (id: string) => void
  trainPath: string
  validPath: string
  onTrainChange: (p: string) => void
  onValidChange: (p: string) => void
  trainingType: TrainingType
}) {
  const navigate = useNavigate()

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-heading">Prepare Data</h3>
        <p className="text-sm text-caption">
          Select a downloaded dataset or provide custom file paths.
        </p>
      </div>

      {/* Source toggle */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onDataSourceChange('downloaded')}
          className={cn(
            'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
            dataSource === 'downloaded'
              ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
              : 'text-body border border-subtle hover:border-default'
          )}
        >
          Downloaded Datasets
        </button>
        <button
          onClick={() => onDataSourceChange('custom')}
          className={cn(
            'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
            dataSource === 'custom'
              ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
              : 'text-body border border-subtle hover:border-default'
          )}
        >
          Custom File Path
        </button>
      </div>

      {dataSource === 'downloaded' ? (
        <div className="space-y-3">
          {downloadedDatasets.length === 0 ? (
            <div className="rounded-md border border-subtle bg-surface-card p-6 text-center">
              <p className="text-sm text-caption mb-3">No datasets downloaded yet.</p>
              <button
                onClick={() => navigate('/datasets')}
                className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-500"
              >
                <Download className="h-3 w-3" /> Browse Dataset Catalog
              </button>
            </div>
          ) : (
            <div className="space-y-2 max-h-[260px] overflow-y-auto">
              {downloadedDatasets.map((ds) => {
                const isSelected = selectedDataset === ds.id
                return (
                  <button
                    key={ds.id}
                    onClick={() => onDatasetSelect(ds.id)}
                    className={cn(
                      'w-full text-left rounded-lg border p-3 transition-all',
                      isSelected
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : 'border-subtle bg-surface-card hover:border-default'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className={cn('font-medium text-sm', isSelected ? 'text-indigo-300' : 'text-label')}>
                        {ds.display_name}
                      </span>
                      <span className="text-xs text-caption">{ds.num_samples.toLocaleString()} samples</span>
                    </div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-caption">
                      <span>Format: {ds.format}</span>
                      <span className="font-mono text-muted truncate max-w-[300px]">{ds.path}</span>
                    </div>
                  </button>
                )
              })}
            </div>
          )}
          {selectedDataset && (
            <p className="text-xs text-muted">
              Validation split will be auto-created from the training data if no separate validation file exists.
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-label mb-1">Training Data</label>
            <input
              type="text"
              value={trainPath}
              onChange={(e) => onTrainChange(e.target.value)}
              placeholder="/path/to/train.jsonl"
              className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label placeholder-muted focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-label mb-1">Validation Data</label>
            <input
              type="text"
              value={validPath}
              onChange={(e) => onValidChange(e.target.value)}
              placeholder="/path/to/valid.jsonl"
              className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label placeholder-muted focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>
      )}

      <div className="rounded-md bg-surface-card border border-default/50 p-3">
        <p className="text-xs text-body">
          <strong className="text-label">Expected format:</strong>{' '}
          {DATA_FORMAT_HINTS[trainingType]}
        </p>
      </div>
    </div>
  )
}

// Step 3: Configure
function StepConfig({ trainingType, onTrainingTypeChange, defaults, advanced, onAdvancedToggle, overrides, onOverridesChange, memoryEstimate }: {
  trainingType: TrainingType
  onTrainingTypeChange: (t: TrainingType) => void
  defaults: Record<string, unknown>
  advanced: boolean
  onAdvancedToggle: (v: boolean) => void
  overrides: Record<string, unknown>
  onOverridesChange: (o: Record<string, unknown>) => void
  memoryEstimate: MemoryEstimateResult | null
}) {
  const update = (key: string, value: unknown) => {
    onOverridesChange({ ...overrides, [key]: value })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-heading">Configure Training</h3>
          <p className="text-sm text-caption">Choose training method and adjust hyperparameters.</p>
        </div>
        <label className="flex items-center gap-2 text-sm text-body cursor-pointer">
          <input
            type="checkbox"
            checked={advanced}
            onChange={(e) => onAdvancedToggle(e.target.checked)}
            className="rounded border-default"
          />
          Advanced
        </label>
      </div>

      {/* Training type selector */}
      <div>
        <label className="block text-xs font-medium text-body mb-2">Training Type</label>
        <div className="grid grid-cols-4 gap-2">
          {TRAINING_TYPES.filter(t =>
            advanced || ['sft', 'dpo', 'mlm', 'seq2seq'].includes(t.value)
          ).map((t) => (
            <button
              key={t.value}
              onClick={() => onTrainingTypeChange(t.value)}
              className={cn(
                'text-left rounded-md border px-3 py-2 transition-all',
                trainingType === t.value
                  ? 'border-indigo-500 bg-indigo-500/10'
                  : 'border-subtle bg-surface-card hover:border-default'
              )}
            >
              <span className={cn(
                'text-xs font-semibold',
                trainingType === t.value ? 'text-indigo-400' : 'text-label'
              )}>
                {t.label}
              </span>
              <p className="text-[10px] text-caption mt-0.5 leading-tight">{t.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Hyperparameter sliders */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Learning Rate: {(overrides['learning_rate'] as number ?? defaults.learning_rate ?? 2e-5).toExponential(0)}
          </label>
          <input
            type="range"
            min={-7}
            max={-3}
            step={1}
            value={Math.log10((overrides['learning_rate'] as number) || (defaults.learning_rate as number) || 2e-5)}
            onChange={(e) => update('learning_rate', Math.pow(10, Number(e.target.value)))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Training Steps: {(overrides['num_iters'] as number) || (defaults.num_iters as number) || 1000}
          </label>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={(overrides['num_iters'] as number) || (defaults.num_iters as number) || 1000}
            onChange={(e) => update('num_iters', Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Batch Size: {(overrides['batch_size'] as number) || (defaults.batch_size as number) || 4}
          </label>
          <input
            type="range"
            min={1}
            max={32}
            step={1}
            value={(overrides['batch_size'] as number) || (defaults.batch_size as number) || 4}
            onChange={(e) => update('batch_size', Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Advanced mode: more fields */}
      {advanced && (
        <div className="space-y-3 pt-2 border-t border-subtle">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-body mb-1">Optimizer</label>
              <select
                value={(overrides['optimizer'] as string) || 'adam'}
                onChange={(e) => update('optimizer', e.target.value)}
                className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label"
              >
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="sgd">SGD</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-body mb-1">Max Grad Norm</label>
              <input
                type="number"
                value={(overrides['max_grad_norm'] as number) ?? 1.0}
                onChange={(e) => update('max_grad_norm', Number(e.target.value))}
                className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label"
                step={0.1}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-body mb-1">LoRA Rank</label>
              <select
                value={(overrides['lora_rank'] as number) || 16}
                onChange={(e) => update('lora_rank', Number(e.target.value))}
                className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label"
              >
                <option value={4}>4</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={64}>64</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-body mb-1">Max Seq Length</label>
              <select
                value={(overrides['max_seq_length'] as number) || 2048}
                onChange={(e) => update('max_seq_length', Number(e.target.value))}
                className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label"
              >
                <option value={512}>512</option>
                <option value={1024}>1024</option>
                <option value={2048}>2048</option>
                <option value={4096}>4096</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-body cursor-pointer">
              <input
                type="checkbox"
                checked={(overrides['gradient_checkpointing'] as boolean) ?? false}
                onChange={(e) => update('gradient_checkpointing', e.target.checked)}
                className="rounded border-default"
              />
              Gradient Checkpointing
            </label>
          </div>
        </div>
      )}

      {/* Memory bar */}
      {memoryEstimate && (
        <div className="pt-2">
          <MemoryBar
            segments={memoryEstimate.bar_segments}
            totalGb={memoryEstimate.total_gb}
            budgetGb={memoryEstimate.budget_gb}
            fits={memoryEstimate.fits}
          />
        </div>
      )}
    </div>
  )
}

// Step 4: Review
function StepReview({ config, memoryEstimate, hardware }: {
  config: Record<string, unknown>
  memoryEstimate: MemoryEstimateResult | null
  hardware: { total_memory_gb: number; training_budget_gb: number; chip_name: string } | null
}) {
  const model = config.model as Record<string, unknown>
  const data = config.data as Record<string, unknown>
  const training = config.training as Record<string, unknown>

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-heading">Review & Start</h3>

      <div className="rounded-md border border-subtle divide-y divide-subtle">
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Model</span>
          <span className="text-sm font-mono text-label">{(model?.path as string) || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Training Type</span>
          <span className="text-sm text-label uppercase">{(training?.training_type as string) || 'sft'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Training Data</span>
          <span className="text-sm font-mono text-body truncate max-w-[300px]">{(data?.train as string) || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Validation Data</span>
          <span className="text-sm font-mono text-body truncate max-w-[300px]">{(data?.valid as string) || '-'}</span>
        </div>
        {Object.entries(training || {}).filter(([k]) => k !== 'training_type').map(([key, value]) => (
          <div key={key} className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-caption">{key}</span>
            <span className="text-sm font-mono text-label">
              {typeof value === 'number' ? (value < 0.01 ? value.toExponential(0) : String(value)) : String(value)}
            </span>
          </div>
        ))}
      </div>

      {memoryEstimate && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-label">Memory Estimate</h4>
          <MemoryBar
            segments={memoryEstimate.bar_segments}
            totalGb={memoryEstimate.total_gb}
            budgetGb={memoryEstimate.budget_gb}
            fits={memoryEstimate.fits}
          />
          {!memoryEstimate.fits && (
            <div className="rounded-md bg-red-500/10 border border-red-500/30 p-2 flex items-center gap-2 text-xs text-red-400">
              <AlertTriangle className="h-3 w-3" />
              Estimated memory exceeds budget. Consider enabling QLoRA or reducing batch size.
            </div>
          )}
        </div>
      )}

      {hardware && (
        <p className="text-xs text-muted">
          Hardware: {hardware.chip_name} ({formatMemory(hardware.total_memory_gb)})
        </p>
      )}
    </div>
  )
}
