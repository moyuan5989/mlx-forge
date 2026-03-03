import { cn } from '../../lib/utils'

const variants: Record<string, string> = {
  completed: 'bg-emerald-500/10 text-emerald-400',
  running: 'bg-blue-500/10 text-blue-400',
  stopped: 'bg-amber-500/10 text-amber-400',
  unknown: 'bg-surface-muted text-body',
}

export default function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
        variants[status] || variants.unknown
      )}
    >
      {status}
    </span>
  )
}
