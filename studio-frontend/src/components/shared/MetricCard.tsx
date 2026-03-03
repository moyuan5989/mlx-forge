interface Props {
  label: string
  value: string | number
  unit?: string
}

export default function MetricCard({ label, value, unit }: Props) {
  return (
    <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-3">
      <p className="text-xs text-caption mb-1">{label}</p>
      <p className="text-lg font-semibold text-heading">
        {value}
        {unit && <span className="text-sm font-normal text-body ml-1">{unit}</span>}
      </p>
    </div>
  )
}
