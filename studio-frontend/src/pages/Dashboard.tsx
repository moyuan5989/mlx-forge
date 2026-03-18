import { useState } from 'react'
import { Link } from 'react-router-dom'
import { FlaskConical, Activity, Box, Database, ArrowRight, Square } from 'lucide-react'
import { useRuns } from '../hooks/useRuns'
import { useModels } from '../hooks/useModels'
import { useDatasets } from '../hooks/useDatasets'
import { useActiveTraining } from '../hooks/useTraining'
import StatCard from '../components/shared/StatCard'
import StatusBadge from '../components/shared/StatusBadge'
import { formatLoss, truncate } from '../lib/utils'
import { api } from '../api/client'

export default function Dashboard() {
  const { data: runs, refetch: refetchRuns } = useRuns()
  const { data: models } = useModels()
  const { data: datasets } = useDatasets()
  const { data: active } = useActiveTraining()
  const [stoppingId, setStoppingId] = useState<string | null>(null)

  const totalRuns = runs?.length ?? 0
  const activeCount = active?.length ?? 0
  const modelsCount = models?.length ?? 0
  const datasetsCount = datasets?.length ?? 0

  const runningRuns = runs?.filter((r) => r.status === 'running') ?? []
  const recentRuns = runs?.slice(0, 5) ?? []

  async function handleStop(runId: string, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm('Stop this training job?')) return
    setStoppingId(runId)
    try {
      const activeList = await api.getActiveTraining()
      const match = activeList.find((a) => a.run_id === runId || a.track_id?.includes(runId))
      if (match) {
        await api.stopTraining(match.track_id)
      }
      refetchRuns()
    } catch {
      // best effort
    } finally {
      setStoppingId(null)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-heading">Dashboard</h2>
        <div className="flex gap-2">
          <Link
            to="/new"
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 transition-colors"
          >
            New Training
          </Link>
          <Link
            to="/experiments"
            className="rounded-md border border-default px-4 py-2 text-sm font-medium text-label hover:bg-surface-hover transition-colors"
          >
            Experiments
          </Link>
          <Link
            to="/playground"
            className="rounded-md border border-default px-4 py-2 text-sm font-medium text-label hover:bg-surface-hover transition-colors"
          >
            Playground
          </Link>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard icon={FlaskConical} label="Total Runs" value={totalRuns} />
        <StatCard icon={Activity} label="Active Runs" value={activeCount} />
        <StatCard icon={Box} label="Models Downloaded" value={modelsCount} />
        <StatCard icon={Database} label="Datasets Cached" value={datasetsCount} />
      </div>

      {/* Active training */}
      {runningRuns.length > 0 && (
        <section>
          <h3 className="text-lg font-semibold text-heading mb-3">Active Training</h3>
          <div className="space-y-3">
            {runningRuns.map((run) => {
              const progress = run.num_iters > 0 ? (run.current_step / run.num_iters) * 100 : 0
              return (
                <Link
                  key={run.id}
                  to={`/experiments/${run.id}`}
                  className="block rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4 hover:border-default transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-label">
                      {truncate(run.id, 32)}
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => handleStop(run.id, e)}
                        disabled={stoppingId === run.id}
                        className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors"
                        title="Stop training"
                      >
                        <Square className="h-3 w-3" />
                        {stoppingId === run.id ? 'Stopping...' : 'Stop'}
                      </button>
                      <span className="text-xs text-caption">
                        {run.current_step}/{run.num_iters}
                      </span>
                    </div>
                  </div>
                  <div className="w-full bg-progress-track rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${Math.min(progress, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-caption mt-1">
                    {run.model && truncate(run.model, 40)}
                    {run.latest_train_loss != null && ` \u00b7 loss: ${formatLoss(run.latest_train_loss)}`}
                  </p>
                </Link>
              )
            })}
          </div>
        </section>
      )}

      {/* Recent runs */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-heading">Recent Runs</h3>
          {totalRuns > 5 && (
            <Link to="/experiments" className="text-sm text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
              View all <ArrowRight className="h-3 w-3" />
            </Link>
          )}
        </div>

        {recentRuns.length === 0 ? (
          <p className="text-sm text-caption">No training runs yet.</p>
        ) : (
          <div className="rounded-lg border border-subtle overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-subtle text-caption">
                  <th className="text-left px-4 py-2 font-medium">Run</th>
                  <th className="text-left px-4 py-2 font-medium">Model</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-right px-4 py-2 font-medium">Step</th>
                  <th className="text-right px-4 py-2 font-medium">Train Loss</th>
                  <th className="text-right px-4 py-2 font-medium">Val Loss</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {recentRuns.map((run) => (
                  <tr key={run.id} className="hover:bg-surface-hover">
                    <td className="px-4 py-2">
                      <Link to={`/experiments/${run.id}`} className="text-indigo-400 hover:text-indigo-300">
                        {truncate(run.id, 20)}
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-body font-mono text-xs">
                      {run.model ? truncate(run.model, 30) : '-'}
                    </td>
                    <td className="px-4 py-2">
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="px-4 py-2 text-right text-body">
                      {run.current_step}/{run.num_iters}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-label">
                      {formatLoss(run.latest_train_loss)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-label">
                      {formatLoss(run.latest_val_loss)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
