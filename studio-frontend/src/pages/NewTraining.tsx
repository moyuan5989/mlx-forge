import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  MessageSquare,
  ClipboardCheck,
  Pen,
  ThumbsUp,
  ChevronRight,
  ChevronLeft,
  Play,
  Download,
  AlertTriangle,
} from 'lucide-react'
import { cn, formatMemory } from '../lib/utils'
import { useRecipes, useResolveRecipe } from '../hooks/useRecipes'
import { useCompatibleModels, useMemoryEstimate, useHardware } from '../hooks/useMemory'
import { useSubmitJob } from '../hooks/useQueue'
import MemoryBar from '../components/shared/MemoryBar'
import type { Recipe, CompatibleModel, MemoryEstimateResult } from '../api/types'

const STEPS = ['Task', 'Model', 'Data', 'Config', 'Review'] as const

const recipeIcons: Record<string, typeof MessageSquare> = {
  MessageSquare,
  ClipboardCheck,
  Pen,
  ThumbsUp,
}

export default function NewTraining() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [currentStep, setCurrentStep] = useState(0)

  // Wizard state
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [selectedModel, setSelectedModel] = useState('')

  // Pre-select model from ?model= query param
  useEffect(() => {
    const modelParam = searchParams.get('model')
    if (modelParam && !selectedModel) {
      setSelectedModel(modelParam)
      setCurrentStep(1)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const [trainPath, setTrainPath] = useState('')
  const [validPath, setValidPath] = useState('')
  const [advancedMode, setAdvancedMode] = useState(false)
  const [configOverrides, setConfigOverrides] = useState<Record<string, unknown>>({})

  // Data
  const { data: recipes } = useRecipes()
  const { data: models } = useCompatibleModels()
  const { data: hardware } = useHardware()
  const memEstimate = useMemoryEstimate()
  const submitJob = useSubmitJob()
  const resolveRecipe = useResolveRecipe()

  // Re-estimate memory when model or config changes
  const memBatchSize = (configOverrides['batch_size'] as number) || 4
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
      case 0: return selectedRecipe != null
      case 1: return selectedModel !== ''
      case 2: return trainPath !== '' && validPath !== ''
      case 3: return true
      case 4: return true
      default: return false
    }
  }

  const resolveConfig = async () => {
    if (!selectedRecipe) return null
    return resolveRecipe.mutateAsync({
      recipeId: selectedRecipe.id,
      body: {
        model_id: selectedModel,
        train_path: trainPath,
        valid_path: validPath,
        overrides: configOverrides,
      },
    })
  }

  const handleSubmit = async () => {
    try {
      const config = await resolveConfig()
      if (!config) return
      await submitJob.mutateAsync(config)
      navigate('/queue')
    } catch {
      // Error handled by mutation state
    }
  }

  const handleExportConfig = async () => {
    try {
      const config = await resolveConfig()
      if (!config) return
      const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'lmforge-config.json'
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      // Error handled by mutation state
    }
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
          <StepTask
            recipes={recipes || []}
            selected={selectedRecipe}
            onSelect={setSelectedRecipe}
          />
        )}
        {currentStep === 1 && (
          <StepModel
            models={models || []}
            selected={selectedModel}
            onSelect={setSelectedModel}
            memoryEstimate={memEstimate.data || null}
            hardware={hardware || null}
            recipe={selectedRecipe}
          />
        )}
        {currentStep === 2 && (
          <StepData
            trainPath={trainPath}
            validPath={validPath}
            onTrainChange={setTrainPath}
            onValidChange={setValidPath}
            recipe={selectedRecipe}
          />
        )}
        {currentStep === 3 && (
          <StepConfig
            recipe={selectedRecipe}
            advanced={advancedMode}
            onAdvancedToggle={setAdvancedMode}
            overrides={configOverrides}
            onOverridesChange={setConfigOverrides}
            memoryEstimate={memEstimate.data || null}
          />
        )}
        {currentStep === 4 && (
          <StepReview
            recipe={selectedRecipe}
            model={selectedModel}
            trainPath={trainPath}
            validPath={validPath}
            overrides={configOverrides}
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
                disabled={resolveRecipe.isPending}
                className="flex items-center gap-1 rounded-md border border-default px-4 py-2 text-sm text-label hover:bg-surface-hover disabled:opacity-50"
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

// Step 1: Choose Task
function StepTask({ recipes, selected, onSelect }: {
  recipes: Recipe[]
  selected: Recipe | null
  onSelect: (r: Recipe) => void
}) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-heading">What do you want to train?</h3>
        <p className="text-sm text-caption">Choose a recipe to get started with optimized defaults.</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {recipes.map((recipe) => {
          const Icon = recipeIcons[recipe.icon] || MessageSquare
          const isSelected = selected?.id === recipe.id
          return (
            <button
              key={recipe.id}
              onClick={() => onSelect(recipe)}
              className={cn(
                'text-left rounded-lg border p-4 transition-all',
                isSelected
                  ? 'border-indigo-500 bg-indigo-500/10'
                  : 'border-subtle bg-surface-card hover:border-default'
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon className={cn('h-5 w-5', isSelected ? 'text-indigo-400' : 'text-caption')} />
                <span className={cn('font-medium', isSelected ? 'text-indigo-300' : 'text-label')}>
                  {recipe.name}
                </span>
              </div>
              <p className="text-xs text-caption leading-relaxed">{recipe.description}</p>
              <span className={cn(
                'inline-block mt-2 px-2 py-0.5 text-[10px] font-medium rounded-full uppercase tracking-wide',
                recipe.category === 'dpo'
                  ? 'bg-purple-500/20 text-purple-400'
                  : 'bg-blue-500/20 text-blue-400'
              )}>
                {recipe.training_type}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// Step 2: Select Model
function StepModel({ models, selected, onSelect, memoryEstimate, hardware, recipe }: {
  models: CompatibleModel[]
  selected: string
  onSelect: (m: string) => void
  memoryEstimate: MemoryEstimateResult | null
  hardware: { total_memory_gb: number; training_budget_gb: number; chip_name: string } | null
  recipe: Recipe | null
}) {
  const recommended = recipe?.recommended_models || []
  const sortedModels = [...models].sort((a, b) => {
    const aRec = recommended.includes(a.model_id) ? -1 : 0
    const bRec = recommended.includes(b.model_id) ? -1 : 0
    if (aRec !== bRec) return aRec - bRec
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
          const isRecommended = recommended.includes(model.model_id)
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
                  {isRecommended && (
                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-emerald-500/20 text-emerald-400">
                      Recommended
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

// Step 3: Prepare Data
function StepData({ trainPath, validPath, onTrainChange, onValidChange, recipe }: {
  trainPath: string
  validPath: string
  onTrainChange: (p: string) => void
  onValidChange: (p: string) => void
  recipe: Recipe | null
}) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-heading">Prepare Data</h3>
        <p className="text-sm text-caption">
          Provide paths to your JSONL training and validation files.
          {recipe && ` Expected format: ${recipe.data_format}`}
        </p>
      </div>

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

      {recipe && (
        <div className="rounded-md bg-surface-card border border-default/50 p-3">
          <p className="text-xs text-body">
            <strong className="text-label">Expected format:</strong>{' '}
            {recipe.data_format === 'chat' && 'JSONL with "messages" field (array of {role, content})'}
            {recipe.data_format === 'completions' && 'JSONL with "prompt" and "completion" fields'}
            {recipe.data_format === 'text' && 'JSONL with "text" field'}
            {recipe.data_format === 'preference' && 'JSONL with "chosen" and "rejected" fields (arrays of messages)'}
          </p>
        </div>
      )}
    </div>
  )
}

// Step 4: Configure
function StepConfig({ recipe, advanced, onAdvancedToggle, overrides, onOverridesChange, memoryEstimate }: {
  recipe: Recipe | null
  advanced: boolean
  onAdvancedToggle: (v: boolean) => void
  overrides: Record<string, unknown>
  onOverridesChange: (o: Record<string, unknown>) => void
  memoryEstimate: MemoryEstimateResult | null
}) {
  const template = recipe?.config_template?.training as Record<string, unknown> | undefined

  const update = (key: string, value: unknown) => {
    onOverridesChange({ ...overrides, [key]: value })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-heading">Configure Training</h3>
          <p className="text-sm text-caption">Adjust settings or use recipe defaults.</p>
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

      {/* Simple mode: key sliders */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Learning Rate: {(overrides['learning_rate'] as number ?? template?.learning_rate ?? 2e-5).toExponential(0)}
          </label>
          <input
            type="range"
            min={-7}
            max={-3}
            step={1}
            value={Math.log10((overrides['learning_rate'] as number) || (template?.learning_rate as number) || 2e-5)}
            onChange={(e) => update('learning_rate', Math.pow(10, Number(e.target.value)))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Training Steps: {(overrides['num_iters'] as number) || (template?.num_iters as number) || 1000}
          </label>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={(overrides['num_iters'] as number) || (template?.num_iters as number) || 1000}
            onChange={(e) => update('num_iters', Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-body mb-1">
            Batch Size: {(overrides['batch_size'] as number) || (template?.batch_size as number) || 4}
          </label>
          <input
            type="range"
            min={1}
            max={32}
            step={1}
            value={(overrides['batch_size'] as number) || (template?.batch_size as number) || 4}
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
                value={(overrides['optimizer'] as string) || (template?.optimizer as string) || 'adam'}
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
                value={(overrides['max_grad_norm'] as number) ?? (template?.max_grad_norm as number) ?? 1.0}
                onChange={(e) => update('max_grad_norm', Number(e.target.value))}
                className="w-full rounded-md border border-default bg-surface-input px-3 py-2 text-sm text-label"
                step={0.1}
              />
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

// Step 5: Review
function StepReview({ recipe, model, trainPath, validPath, overrides, memoryEstimate, hardware }: {
  recipe: Recipe | null
  model: string
  trainPath: string
  validPath: string
  overrides: Record<string, unknown>
  memoryEstimate: MemoryEstimateResult | null
  hardware: { total_memory_gb: number; training_budget_gb: number; chip_name: string } | null
}) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-heading">Review & Start</h3>

      <div className="rounded-md border border-subtle divide-y divide-subtle">
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Recipe</span>
          <span className="text-sm text-label">{recipe?.name || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Model</span>
          <span className="text-sm font-mono text-label">{model || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Training Data</span>
          <span className="text-sm font-mono text-body">{trainPath || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Validation Data</span>
          <span className="text-sm font-mono text-body">{validPath || '-'}</span>
        </div>
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-sm text-caption">Training Type</span>
          <span className="text-sm text-label uppercase">{recipe?.training_type || 'SFT'}</span>
        </div>
        {Object.entries(overrides).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-caption">{key}</span>
            <span className="text-sm font-mono text-label">{String(value)}</span>
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
