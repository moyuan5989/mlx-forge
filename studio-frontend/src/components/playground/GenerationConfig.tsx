interface Config {
  model: string
  adapter: string
  temperature: number
  topP: number
  maxTokens: number
}

interface Props {
  config: Config
  onChange: (config: Config) => void
  models: { id: string; model_id: string }[]
  stats?: { numTokens: number; tokPerSec: number } | null
}

export default function GenerationConfig({ config, onChange, models, stats }: Props) {
  function set<K extends keyof Config>(key: K, value: Config[K]) {
    onChange({ ...config, [key]: value })
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-caption mb-1">Model</label>
        <select
          className="w-full bg-surface-input border border-default rounded-md px-3 py-2 text-sm text-heading focus:outline-none focus:ring-1 focus:ring-indigo-500"
          value={config.model}
          onChange={(e) => set('model', e.target.value)}
        >
          <option value="">Select a model...</option>
          {models.map((m) => (
            <option key={m.id} value={m.model_id}>
              {m.model_id}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-xs text-caption mb-1">Adapter Path (optional)</label>
        <input
          type="text"
          className="w-full bg-surface-input border border-default rounded-md px-3 py-2 text-sm text-heading placeholder-caption focus:outline-none focus:ring-1 focus:ring-indigo-500"
          placeholder="/path/to/checkpoint"
          value={config.adapter}
          onChange={(e) => set('adapter', e.target.value)}
        />
      </div>

      <div>
        <label className="block text-xs text-caption mb-1">
          Temperature: {config.temperature.toFixed(1)}
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          className="w-full accent-indigo-500"
          value={config.temperature}
          onChange={(e) => set('temperature', parseFloat(e.target.value))}
        />
      </div>

      <div>
        <label className="block text-xs text-caption mb-1">
          Top-p: {config.topP.toFixed(1)}
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          className="w-full accent-indigo-500"
          value={config.topP}
          onChange={(e) => set('topP', parseFloat(e.target.value))}
        />
      </div>

      <div>
        <label className="block text-xs text-caption mb-1">Max Tokens</label>
        <input
          type="number"
          min="1"
          max="4096"
          className="w-full bg-surface-input border border-default rounded-md px-3 py-2 text-sm text-heading focus:outline-none focus:ring-1 focus:ring-indigo-500"
          value={config.maxTokens}
          onChange={(e) => set('maxTokens', parseInt(e.target.value) || 512)}
        />
      </div>

      {stats && (
        <div className="border-t border-subtle pt-3 space-y-1">
          <p className="text-xs text-caption">
            Tokens: <span className="text-label">{stats.numTokens}</span>
          </p>
          <p className="text-xs text-caption">
            Speed: <span className="text-label">{stats.tokPerSec.toFixed(1)} tok/s</span>
          </p>
        </div>
      )}
    </div>
  )
}
