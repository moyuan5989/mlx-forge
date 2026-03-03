import { Link } from 'react-router-dom'
import {
  Play,
  Clock,
  CheckCircle,
  XCircle,
  Ban,
  ArrowUp,
  Trash2,
  Plus,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { useQueue, useQueueStats, useCancelJob, usePromoteJob } from '../hooks/useQueue'
import StatCard from '../components/shared/StatCard'
import type { QueueJob } from '../api/types'

const statusConfig: Record<string, { icon: typeof Play; color: string; label: string }> = {
  queued: { icon: Clock, color: 'text-yellow-400', label: 'Queued' },
  running: { icon: Play, color: 'text-blue-400', label: 'Running' },
  completed: { icon: CheckCircle, color: 'text-emerald-400', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-red-400', label: 'Failed' },
  cancelled: { icon: Ban, color: 'text-caption', label: 'Cancelled' },
}

export default function JobQueue() {
  const { data: jobs } = useQueue()
  const { data: stats } = useQueueStats()
  const cancelJob = useCancelJob()
  const promoteJob = usePromoteJob()

  const activeJobs = jobs?.filter((j) => j.status === 'queued' || j.status === 'running') || []
  const pastJobs = jobs?.filter((j) => j.status !== 'queued' && j.status !== 'running') || []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-heading">Job Queue</h2>
        <Link
          to="/new"
          className="flex items-center gap-1 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
        >
          <Plus className="h-4 w-4" /> New Training
        </Link>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <StatCard icon={Clock} label="Queued" value={stats.queued} />
          <StatCard icon={Play} label="Running" value={stats.running} />
          <StatCard icon={CheckCircle} label="Completed" value={stats.completed} />
          <StatCard icon={XCircle} label="Failed" value={stats.failed} />
        </div>
      )}

      {/* Active Jobs */}
      <section>
        <h3 className="text-lg font-semibold text-heading mb-3">Active</h3>
        {activeJobs.length === 0 ? (
          <p className="text-sm text-caption">No active jobs. Start a new training run.</p>
        ) : (
          <div className="space-y-2">
            {activeJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onCancel={() => cancelJob.mutate(job.id)}
                onPromote={() => promoteJob.mutate(job.id)}
              />
            ))}
          </div>
        )}
      </section>

      {/* Past Jobs */}
      {pastJobs.length > 0 && (
        <section>
          <h3 className="text-lg font-semibold text-heading mb-3">History</h3>
          <div className="space-y-2">
            {pastJobs.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

function JobCard({ job, onCancel, onPromote }: {
  job: QueueJob
  onCancel?: () => void
  onPromote?: () => void
}) {
  const cfg = statusConfig[job.status] || statusConfig.queued
  const Icon = cfg.icon
  const modelPath = (job.config?.model as Record<string, unknown>)?.path as string | undefined

  return (
    <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Icon className={cn('h-4 w-4', cfg.color)} />
          <div>
            <span className="text-sm font-medium text-label">
              Job {job.id}
            </span>
            {modelPath && (
              <span className="text-xs text-caption ml-2 font-mono">{modelPath}</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('text-xs font-medium', cfg.color)}>{cfg.label}</span>
          {job.status === 'queued' && job.position > 0 && onPromote && (
            <button
              onClick={onPromote}
              className="p-1 text-caption hover:text-label"
              title="Move to front"
            >
              <ArrowUp className="h-3.5 w-3.5" />
            </button>
          )}
          {(job.status === 'queued' || job.status === 'running') && onCancel && (
            <button
              onClick={onCancel}
              className="p-1 text-caption hover:text-red-400"
              title="Cancel"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {job.status === 'queued' && (
        <p className="text-xs text-caption mt-1">Position in queue: {job.position + 1}</p>
      )}

      {job.run_id && (
        <p className="text-xs text-caption mt-1">
          Run:{' '}
          <Link to={`/experiments/${job.run_id}`} className="text-indigo-400 hover:text-indigo-300">
            {job.run_id}
          </Link>
        </p>
      )}

      {job.error && (
        <p className="text-xs text-red-400 mt-1">{job.error}</p>
      )}

      {job.completed_at && (
        <p className="text-xs text-muted mt-1">
          Completed {new Date(job.completed_at * 1000).toLocaleTimeString()}
        </p>
      )}
    </div>
  )
}
