import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Cpu, Download, Play } from 'lucide-react'
import { useModelLibrary } from '../hooks/useModels'
import { useHardware } from '../hooks/useMemory'
import { cn, formatMemory } from '../lib/utils'
import type { LibraryModel } from '../api/types'

type FitFilter = 'all' | 'fits' | 'recommended'
type SortKey = 'size' | 'name'

const ARCH_OPTIONS = ['all', 'qwen3', 'qwen2', 'llama', 'gemma2', 'gemma3', 'phi3', 'phi4'] as const

export default function Models() {
  const { data: models, isLoading } = useModelLibrary()
  const { data: hardware } = useHardware()
  const navigate = useNavigate()

  const [archFilter, setArchFilter] = useState<string>('all')
  const [fitFilter, setFitFilter] = useState<FitFilter>('all')
  const [sortKey, setSortKey] = useState<SortKey>('size')

  const filtered = useMemo(() => {
    if (!models) return []
    let result = [...models]

    if (archFilter !== 'all') {
      result = result.filter((m) => m.architecture === archFilter)
    }
    if (fitFilter === 'fits') {
      result = result.filter((m) => m.fp16.fits || m.qlora_4bit.fits)
    } else if (fitFilter === 'recommended') {
      result = result.filter((m) => m.recommended)
    }

    if (sortKey === 'name') {
      result.sort((a, b) => a.display_name.localeCompare(b.display_name))
    }
    // 'size' is default sort from backend (already sorted by num_params_b)

    return result
  }, [models, archFilter, fitFilter, sortKey])

  if (isLoading) return <p className="text-caption text-sm">Loading model library...</p>

  return (
    <div className="space-y-4">
      {/* Header + Hardware Banner */}
      <div>
        <h2 className="text-2xl font-bold text-heading">Model Library</h2>
        {hardware && (
          <p className="text-sm text-caption mt-1">
            <Cpu className="inline h-3.5 w-3.5 mr-1 -mt-0.5" />
            {hardware.chip_name} &mdash; {formatMemory(hardware.total_memory_gb)} total,{' '}
            {formatMemory(hardware.training_budget_gb)} training budget
          </p>
        )}
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={archFilter}
          onChange={(e) => setArchFilter(e.target.value)}
          className="rounded-md border border-default bg-surface-input px-2.5 py-1.5 text-xs text-label"
        >
          {ARCH_OPTIONS.map((a) => (
            <option key={a} value={a}>
              {a === 'all' ? 'All Architectures' : a}
            </option>
          ))}
        </select>

        <select
          value={fitFilter}
          onChange={(e) => setFitFilter(e.target.value as FitFilter)}
          className="rounded-md border border-default bg-surface-input px-2.5 py-1.5 text-xs text-label"
        >
          <option value="all">All Models</option>
          <option value="fits">Fits on Device</option>
          <option value="recommended">Recommended</option>
        </select>

        <select
          value={sortKey}
          onChange={(e) => setSortKey(e.target.value as SortKey)}
          className="rounded-md border border-default bg-surface-input px-2.5 py-1.5 text-xs text-label"
        >
          <option value="size">Sort by Size</option>
          <option value="name">Sort by Name</option>
        </select>

        <span className="text-xs text-muted ml-auto">
          {filtered.length} of {models?.length ?? 0} models
        </span>
      </div>

      {/* Card Grid */}
      {filtered.length === 0 ? (
        <p className="text-sm text-caption">No models match the current filters.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((model) => (
            <ModelCard key={model.model_id} model={model} onTrain={() => navigate(`/new?model=${encodeURIComponent(model.model_id)}`)} />
          ))}
        </div>
      )}
    </div>
  )
}

function ModelCard({ model, onTrain }: { model: LibraryModel; onTrain: () => void }) {
  const canFit = model.fp16.fits || model.qlora_4bit.fits

  return (
    <div
      className={cn(
        'rounded-lg border p-4 flex flex-col',
        canFit
          ? 'border-subtle bg-surface-card shadow-[var(--shadow-card)]'
          : 'border-subtle/50 bg-surface-overlay opacity-60'
      )}
    >
      {/* Title row */}
      <div className="flex items-start justify-between mb-1">
        <h3 className="text-sm font-medium text-label">{model.display_name}</h3>
        <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
          {model.recommended && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-emerald-500/20 text-emerald-400">
              Recommended
            </span>
          )}
          {model.downloaded && (
            <span className="flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] font-medium rounded bg-blue-500/20 text-blue-400">
              <Download className="h-2.5 w-2.5" /> Downloaded
            </span>
          )}
        </div>
      </div>

      {/* Model ID */}
      <p className="text-[11px] font-mono text-caption mb-2">{model.model_id}</p>

      {/* Badges */}
      <div className="flex items-center gap-2 mb-3">
        <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
          {model.architecture}
        </span>
        <span className="text-xs text-caption">{model.num_params_b}B params</span>
      </div>

      {/* Memory estimates */}
      <div className="text-xs text-caption space-y-1 mb-3">
        <p className="flex items-center gap-1.5">
          <FitDot fits={model.fp16.fits} />
          FP16: {model.fp16.total_gb.toFixed(1)} GB
        </p>
        <p className="flex items-center gap-1.5">
          <FitDot fits={model.qlora_4bit.fits} />
          QLoRA 4-bit: {model.qlora_4bit.total_gb.toFixed(1)} GB
        </p>
      </div>

      {/* Arch details */}
      <div className="text-[11px] text-muted space-y-0.5 mb-3">
        <p>Layers: {model.num_layers} &middot; Hidden: {model.hidden_dim} &middot; Vocab: {model.vocab_size.toLocaleString()}</p>
      </div>

      {/* Train button */}
      <div className="mt-auto pt-2">
        <button
          onClick={onTrain}
          disabled={!canFit}
          className={cn(
            'flex items-center justify-center gap-1 w-full rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
            canFit
              ? 'bg-indigo-600 text-white hover:bg-indigo-500'
              : 'bg-surface-input text-muted cursor-not-allowed'
          )}
        >
          <Play className="h-3 w-3" /> Train
        </button>
      </div>
    </div>
  )
}

function FitDot({ fits }: { fits: boolean }) {
  return (
    <span
      className={cn(
        'inline-block h-2 w-2 rounded-full flex-shrink-0',
        fits ? 'bg-emerald-400' : 'bg-red-400'
      )}
    />
  )
}
