import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Trash2 } from 'lucide-react'
import { useRuns, useDeleteRun } from '../hooks/useRuns'
import StatusBadge from '../components/shared/StatusBadge'
import { formatLoss, truncate } from '../lib/utils'

export default function Experiments() {
  const { data: runs, isLoading } = useRuns()
  const deleteRun = useDeleteRun()
  const [confirmId, setConfirmId] = useState<string | null>(null)

  function handleDelete(id: string) {
    if (confirmId === id) {
      deleteRun.mutate(id)
      setConfirmId(null)
    } else {
      setConfirmId(id)
    }
  }

  if (isLoading) {
    return <p className="text-caption text-sm">Loading runs...</p>
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-heading">Experiments</h2>

      {!runs || runs.length === 0 ? (
        <p className="text-sm text-caption">No training runs found.</p>
      ) : (
        <div className="rounded-lg border border-subtle overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle text-caption">
                <th className="text-left px-4 py-2 font-medium">Run ID</th>
                <th className="text-left px-4 py-2 font-medium">Model</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-right px-4 py-2 font-medium">Progress</th>
                <th className="text-right px-4 py-2 font-medium">Train Loss</th>
                <th className="text-right px-4 py-2 font-medium">Val Loss</th>
                <th className="text-right px-4 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-subtle">
              {runs.map((run) => (
                <tr key={run.id} className="hover:bg-surface-hover">
                  <td className="px-4 py-2">
                    <Link to={`/experiments/${run.id}`} className="text-indigo-400 hover:text-indigo-300">
                      {truncate(run.id, 24)}
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
                  <td className="px-4 py-2 text-right">
                    <button
                      className="text-caption hover:text-red-400 transition-colors p-1"
                      onClick={() => handleDelete(run.id)}
                      title={confirmId === run.id ? 'Click again to confirm' : 'Delete run'}
                    >
                      <Trash2 className={`h-4 w-4 ${confirmId === run.id ? 'text-red-400' : ''}`} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
