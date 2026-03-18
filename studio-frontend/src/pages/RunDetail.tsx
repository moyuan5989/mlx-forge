import { useCallback, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Star, Download, Square } from 'lucide-react'
import { useRun, useRunMetrics, useRunCheckpoints } from '../hooks/useRuns'
import { useWebSocket } from '../hooks/useWebSocket'
import StatusBadge from '../components/shared/StatusBadge'
import MetricCard from '../components/shared/MetricCard'
import LossChart from '../components/charts/LossChart'
import { formatLoss, formatTokPerSec, formatMemory, formatNumber } from '../lib/utils'
import type { TrainMetric, EvalMetric } from '../api/types'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'

export default function RunDetail() {
  const { id } = useParams<{ id: string }>()
  const { data: run, isLoading } = useRun(id!)
  const { data: metrics } = useRunMetrics(id!)
  const { data: checkpoints } = useRunCheckpoints(id!)
  const queryClient = useQueryClient()

  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState<string | null>(null)
  const [stopping, setStopping] = useState(false)
  const [showPushModal, setShowPushModal] = useState(false)
  const [pushRepoId, setPushRepoId] = useState('')
  const [pushAdapterOnly, setPushAdapterOnly] = useState(false)
  const [pushPrivate, setPushPrivate] = useState(false)
  const [pushing, setPushing] = useState(false)
  const [pushResult, setPushResult] = useState<string | null>(null)

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

  async function handleExport() {
    if (!id) return
    setExporting(true)
    setExportResult(null)
    try {
      const res = await api.exportRun(id)
      setExportResult(res.output_dir)
    } catch (e) {
      setExportResult(`Error: ${e instanceof Error ? e.message : 'Export failed'}`)
    } finally {
      setExporting(false)
    }
  }

  async function handlePushToHub() {
    if (!id || !pushRepoId.trim()) return
    setPushing(true)
    setPushResult(null)
    try {
      const res = await api.pushToHub(id, pushRepoId.trim(), {
        adapterOnly: pushAdapterOnly,
        private: pushPrivate,
      })
      setPushResult(res.url)
      setShowPushModal(false)
    } catch (e) {
      setPushResult(`Error: ${e instanceof Error ? e.message : 'Push failed'}`)
    } finally {
      setPushing(false)
    }
  }

  async function handleStop() {
    if (!id || !confirm('Stop the running training job?')) return
    setStopping(true)
    try {
      // Find the track_id from active training
      const active = await api.getActiveTraining()
      const match = active.find((a) => a.run_id === id || a.track_id?.includes(id))
      if (match) {
        await api.stopTraining(match.track_id)
      }
      queryClient.invalidateQueries({ queryKey: ['runs', id] })
    } catch {
      // best effort
    } finally {
      setStopping(false)
    }
  }

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
          <div className="ml-auto flex gap-2">
            {isRunning && (
              <button
                onClick={handleStop}
                disabled={stopping}
                className="inline-flex items-center gap-1.5 rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50 transition-colors"
              >
                <Square className="h-3.5 w-3.5" />
                {stopping ? 'Stopping...' : 'Stop Training'}
              </button>
            )}
            <button
              onClick={handleExport}
              disabled={exporting}
              className="inline-flex items-center gap-1.5 rounded-md border border-default px-3 py-1.5 text-sm font-medium text-label hover:bg-surface-hover disabled:opacity-50 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              {exporting ? 'Exporting...' : 'Export Merged Model'}
            </button>
            <button
              onClick={() => setShowPushModal(!showPushModal)}
              className="inline-flex items-center gap-1.5 rounded-md border border-default px-3 py-1.5 text-sm font-medium text-label hover:bg-surface-hover transition-colors"
            >
              Push to Hub
            </button>
          </div>
        </div>
        {exportResult && (
          <p className={`text-xs mt-1 ${exportResult.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
            {exportResult}
          </p>
        )}
        {pushResult && (
          <p className={`text-xs mt-1 ${pushResult.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
            {pushResult.startsWith('Error') ? pushResult : `Pushed to: ${pushResult}`}
          </p>
        )}
        {showPushModal && (
          <div className="mt-3 rounded-lg border border-subtle bg-surface-card p-4 space-y-3 max-w-md">
            <h4 className="text-sm font-medium text-label">Push to HuggingFace Hub</h4>
            <div>
              <label className="block text-xs text-caption mb-1">Repository ID</label>
              <input
                type="text"
                value={pushRepoId}
                onChange={(e) => setPushRepoId(e.target.value)}
                placeholder="username/model-name"
                className="w-full rounded-md border border-default bg-surface px-3 py-1.5 text-sm text-label placeholder:text-caption focus:outline-none focus:ring-1 focus:ring-accent"
              />
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-1.5 text-xs text-caption">
                <input
                  type="checkbox"
                  checked={pushAdapterOnly}
                  onChange={(e) => setPushAdapterOnly(e.target.checked)}
                  className="rounded border-default"
                />
                Adapter only
              </label>
              <label className="flex items-center gap-1.5 text-xs text-caption">
                <input
                  type="checkbox"
                  checked={pushPrivate}
                  onChange={(e) => setPushPrivate(e.target.checked)}
                  className="rounded border-default"
                />
                Private repo
              </label>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handlePushToHub}
                disabled={pushing || !pushRepoId.trim()}
                className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent/90 disabled:opacity-50 transition-colors"
              >
                {pushing ? 'Pushing...' : 'Push'}
              </button>
              <button
                onClick={() => setShowPushModal(false)}
                className="rounded-md border border-default px-3 py-1.5 text-sm text-caption hover:bg-surface-hover transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
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
