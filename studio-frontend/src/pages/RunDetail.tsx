import { useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Star } from 'lucide-react'
import { useRun, useRunMetrics, useRunCheckpoints } from '../hooks/useRuns'
import { useWebSocket } from '../hooks/useWebSocket'
import StatusBadge from '../components/shared/StatusBadge'
import MetricCard from '../components/shared/MetricCard'
import LossChart from '../components/charts/LossChart'
import { formatLoss, formatTokPerSec, formatMemory, formatNumber } from '../lib/utils'
import type { TrainMetric, EvalMetric } from '../api/types'
import { useQueryClient } from '@tanstack/react-query'

export default function RunDetail() {
  const { id } = useParams<{ id: string }>()
  const { data: run, isLoading } = useRun(id!)
  const { data: metrics } = useRunMetrics(id!)
  const { data: checkpoints } = useRunCheckpoints(id!)
  const queryClient = useQueryClient()

  const isRunning = run?.status === 'running'

  // Live metrics streaming for running jobs
  const handleWsMessage = useCallback(
    (msg: unknown) => {
      const m = msg as { type: string; data?: TrainMetric | EvalMetric }
      if (m.type === 'metric' && m.data) {
        // Invalidate metrics query to pick up new data
        queryClient.invalidateQueries({ queryKey: ['runs', id, 'metrics'] })
      }
    },
    [id, queryClient]
  )

  useWebSocket({
    url: `/ws/training/${id}`,
    onMessage: handleWsMessage,
    enabled: isRunning,
  })

  if (isLoading) return <p className="text-caption text-sm">Loading...</p>
  if (!run) return <p className="text-caption text-sm">Run not found.</p>

  const trainMetrics = metrics?.train ?? []
  const evalMetrics = metrics?.eval ?? []
  const lastTrain = trainMetrics[trainMetrics.length - 1]
  const lastEval = evalMetrics[evalMetrics.length - 1]

  // Extract config info
  const config = run.config as Record<string, Record<string, unknown>> | undefined
  const adapterPreset = (config?.adapter as Record<string, unknown>)?.preset as string | undefined
  const adapterTargets = (config?.adapter as Record<string, unknown>)?.targets as string[] | undefined

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link to="/experiments" className="inline-flex items-center gap-1 text-sm text-caption hover:text-label mb-3">
          <ArrowLeft className="h-3 w-3" /> Back to Experiments
        </Link>
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold text-heading">{id}</h2>
          <StatusBadge status={run.status} />
        </div>
        <div className="flex gap-4 text-sm text-caption mt-1">
          {run.model && <span>Model: <span className="text-label font-mono">{run.model}</span></span>}
          {adapterPreset && <span>Preset: <span className="text-label">{adapterPreset}</span></span>}
          {adapterTargets && <span>Targets: <span className="text-label">{adapterTargets.join(', ')}</span></span>}
        </div>
      </div>

      {/* Loss chart */}
      <section className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
        <h3 className="text-sm font-medium text-label mb-3">Loss Curve</h3>
        <LossChart trainMetrics={trainMetrics} evalMetrics={evalMetrics} />
      </section>

      {/* Metrics cards */}
      <div className="grid grid-cols-5 gap-3">
        <MetricCard label="Train Loss" value={formatLoss(lastTrain?.train_loss)} />
        <MetricCard label="Best Val Loss" value={formatLoss(lastEval?.val_loss)} />
        <MetricCard
          label="Throughput"
          value={lastTrain ? formatTokPerSec(lastTrain.tokens_per_second) : '-'}
        />
        <MetricCard
          label="Peak Memory"
          value={lastTrain ? formatMemory(lastTrain.peak_memory_gb) : '-'}
        />
        <MetricCard
          label="Trained Tokens"
          value={lastTrain ? formatNumber(lastTrain.trained_tokens, 0) : '-'}
        />
      </div>

      {/* Config */}
      {run.config && (
        <section className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
          <h3 className="text-sm font-medium text-label mb-3">Configuration</h3>
          <pre className="text-xs text-body font-mono overflow-auto max-h-80 whitespace-pre-wrap">
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </section>
      )}

      {/* Checkpoints */}
      {checkpoints && checkpoints.length > 0 && (
        <section>
          <h3 className="text-sm font-medium text-label mb-3">Checkpoints</h3>
          <div className="rounded-lg border border-subtle overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-subtle text-caption">
                  <th className="text-left px-4 py-2 font-medium">Name</th>
                  <th className="text-right px-4 py-2 font-medium">Step</th>
                  <th className="text-right px-4 py-2 font-medium">Val Loss</th>
                  <th className="text-right px-4 py-2 font-medium">Best</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {checkpoints.map((ckpt) => (
                  <tr key={ckpt.name} className="hover:bg-surface-hover">
                    <td className="px-4 py-2 font-mono text-label">{ckpt.name}</td>
                    <td className="px-4 py-2 text-right text-body">
                      {ckpt.state?.step ?? '-'}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-label">
                      {formatLoss(ckpt.state?.best_val_loss)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {ckpt.is_best && <Star className="h-4 w-4 text-amber-400 inline" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
