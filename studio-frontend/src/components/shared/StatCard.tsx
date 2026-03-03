import type { LucideIcon } from 'lucide-react'

interface Props {
  icon: LucideIcon
  label: string
  value: string | number
}

export default function StatCard({ icon: Icon, label, value }: Props) {
  return (
    <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-md bg-surface-muted p-2">
          <Icon className="h-4 w-4 text-body" />
        </div>
        <div>
          <p className="text-2xl font-semibold text-heading">{value}</p>
          <p className="text-xs text-caption">{label}</p>
        </div>
      </div>
    </div>
  )
}
