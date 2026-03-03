import type { MemoryBarSegment } from '../../api/types'
import { cn } from '../../lib/utils'

interface MemoryBarProps {
  segments: MemoryBarSegment[]
  totalGb: number
  budgetGb: number
  fits: boolean
}

const colorValues: Record<string, string> = {
  green: '#10b981',
  blue: '#3b82f6',
  orange: '#fb923c',
  gray: '#71717a',
}

const dotClasses: Record<string, string> = {
  green: 'bg-emerald-500',
  blue: 'bg-blue-500',
  orange: 'bg-orange-400',
  gray: 'bg-zinc-500',
}

export default function MemoryBar({ segments, totalGb, budgetGb, fits }: MemoryBarProps) {
  const maxDisplay = Math.max(budgetGb, totalGb) * 1.1

  // Build a single linear-gradient from segments (no gaps possible)
  const stops: string[] = []
  let cursor = 0
  for (const seg of segments) {
    if (seg.gb <= 0) continue
    const startPct = (cursor / maxDisplay) * 100
    cursor += seg.gb
    const endPct = (cursor / maxDisplay) * 100
    const color = colorValues[seg.color] || '#71717a'
    stops.push(`${color} ${startPct}% ${endPct}%`)
  }
  // Fill the rest with transparent
  const usedPct = (cursor / maxDisplay) * 100
  if (usedPct < 100) {
    stops.push(`transparent ${usedPct}% 100%`)
  }
  const gradient = `linear-gradient(to right, ${stops.join(', ')})`

  return (
    <div className="space-y-2">
      {/* Bar */}
      <div className="relative h-6 bg-surface-input rounded-lg overflow-hidden">
        <div
          className="absolute inset-0 rounded-lg"
          style={{ background: gradient }}
        />

        {/* Budget line */}
        <div
          className={cn(
            'absolute top-0 h-full w-0.5',
            fits ? 'bg-body' : 'bg-red-500'
          )}
          style={{ left: `${(budgetGb / maxDisplay) * 100}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-3 flex-wrap">
          {segments.map((seg) => (
            <span key={seg.label} className="flex items-center gap-1 text-body">
              <span className={cn('inline-block h-2 w-2 rounded-full', dotClasses[seg.color] || 'bg-zinc-500')} />
              {seg.label}: {seg.gb.toFixed(1)} GB
            </span>
          ))}
        </div>
        <span className={cn('font-medium whitespace-nowrap', fits ? 'text-label' : 'text-red-400')}>
          {totalGb.toFixed(1)} / {budgetGb.toFixed(1)} GB
          {!fits && ' (exceeds budget)'}
        </span>
      </div>
    </div>
  )
}
