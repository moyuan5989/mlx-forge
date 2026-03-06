import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { useTheme } from '../../hooks/useTheme'
import type { TrainMetric, EvalMetric } from '../../api/types'

interface Props {
  trainMetrics: TrainMetric[]
  evalMetrics: EvalMetric[]
}

function emaSmooth(values: (number | undefined)[], alpha: number): (number | undefined)[] {
  if (alpha <= 0) return values
  const result: (number | undefined)[] = []
  let prev: number | undefined
  for (const v of values) {
    if (v === undefined) {
      result.push(undefined)
      continue
    }
    if (prev === undefined) {
      prev = v
    } else {
      prev = alpha * prev + (1 - alpha) * v
    }
    result.push(prev)
  }
  return result
}

export default function LossChart({ trainMetrics, evalMetrics }: Props) {
  const { resolvedTheme } = useTheme()
  const [smoothing, setSmoothing] = useState(0.6)

  const chartColors = useMemo(() => {
    const style = getComputedStyle(document.documentElement)
    return {
      grid: style.getPropertyValue('--chart-grid').trim() || '#27272a',
      axis: style.getPropertyValue('--chart-axis').trim() || '#71717a',
      tooltipBg: style.getPropertyValue('--chart-tooltip-bg').trim() || '#18181b',
      tooltipBorder: style.getPropertyValue('--chart-tooltip-border').trim() || '#3f3f46',
    }
  }, [resolvedTheme])

  // Merge into a single array keyed by step
  const byStep = new Map<number, { step: number; train_loss?: number; val_loss?: number }>()

  for (const m of trainMetrics) {
    const entry = byStep.get(m.step) || { step: m.step }
    entry.train_loss = m.train_loss
    byStep.set(m.step, entry)
  }
  for (const m of evalMetrics) {
    const entry = byStep.get(m.step) || { step: m.step }
    entry.val_loss = m.val_loss
    byStep.set(m.step, entry)
  }

  const rawData = Array.from(byStep.values()).sort((a, b) => a.step - b.step)

  const data = useMemo(() => {
    if (smoothing <= 0) return rawData
    const trainValues = rawData.map((d) => d.train_loss)
    const smoothedTrain = emaSmooth(trainValues, smoothing)
    return rawData.map((d, i) => ({
      ...d,
      train_loss_raw: d.train_loss,
      train_loss: smoothedTrain[i],
    }))
  }, [rawData, smoothing])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-caption text-sm">
        No metrics available
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <label className="text-xs text-caption flex items-center gap-2">
          Smoothing
          <input
            type="range"
            min="0"
            max="0.99"
            step="0.01"
            value={smoothing}
            onChange={(e) => setSmoothing(parseFloat(e.target.value))}
            className="w-24 h-1 accent-indigo-500"
          />
          <span className="w-8 text-right font-mono">{smoothing.toFixed(2)}</span>
        </label>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis dataKey="step" stroke={chartColors.axis} tick={{ fontSize: 12 }} />
          <YAxis stroke={chartColors.axis} tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltipBg,
              border: `1px solid ${chartColors.tooltipBorder}`,
              borderRadius: '6px',
              fontSize: '12px',
            }}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          {smoothing > 0 && (
            <Line
              type="monotone"
              dataKey="train_loss_raw"
              stroke="#6366f1"
              strokeWidth={1}
              strokeOpacity={0.25}
              dot={false}
              name="Train Loss (raw)"
              connectNulls
            />
          )}
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
            name={smoothing > 0 ? 'Train Loss (smoothed)' : 'Train Loss'}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="val_loss"
            stroke="#10b981"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ r: 3, fill: '#10b981' }}
            name="Val Loss"
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
