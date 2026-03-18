import { Moon, Sun, Monitor } from 'lucide-react'
import { useTheme, type Theme } from '../hooks/useTheme'
import { cn } from '../lib/utils'

const themeOptions: { value: Theme; label: string; icon: typeof Moon }[] = [
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'system', label: 'System', icon: Monitor },
]

export default function Settings() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-heading">Settings</h2>

      <div className="max-w-lg space-y-6">
        {/* Theme */}
        <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
          <label className="block text-sm font-medium text-label mb-2">Theme</label>
          <div className="inline-flex rounded-lg bg-surface-input p-1 gap-1">
            {themeOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                  theme === value
                    ? 'bg-indigo-600 text-white'
                    : 'text-body hover:text-heading hover:bg-surface-hover'
                )}
                onClick={() => setTheme(value)}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Paths */}
        <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
          <label className="block text-sm font-medium text-label mb-2">Run Directory</label>
          <p className="text-sm text-body font-mono bg-surface-overlay rounded px-3 py-2">
            ~/.mlxforge/runs
          </p>
        </div>

        <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
          <label className="block text-sm font-medium text-label mb-2">API Base URL</label>
          <p className="text-sm text-body font-mono bg-surface-overlay rounded px-3 py-2">
            {window.location.origin}
          </p>
        </div>

        {/* Info */}
        <div className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4">
          <label className="block text-sm font-medium text-label mb-2">About</label>
          <div className="text-sm text-caption space-y-1">
            <p>MLX Forge Studio v0.3.0</p>
            <p>Fine-tune LLMs on your Mac with MLX</p>
          </div>
        </div>
      </div>
    </div>
  )
}
