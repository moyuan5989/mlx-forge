import { useState } from 'react'
import { Download, Trash2, Eye, Search } from 'lucide-react'
import {
  useDataCatalog,
  useDownloadedDatasets,
  useDownloadDataset,
  useDeleteDownloadedDataset,
  useDatasetSamples,
} from '../hooks/useDatasets'
import type { CatalogDataset, DownloadedDataset } from '../api/types'

type Tab = 'catalog' | 'my-datasets'

const CATEGORIES = ['all', 'general', 'conversation', 'code', 'math', 'reasoning', 'safety', 'domain'] as const

export default function Datasets() {
  const [tab, setTab] = useState<Tab>('catalog')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [inspectName, setInspectName] = useState<string | null>(null)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-heading">Data Library</h2>
        <div className="flex gap-1 bg-surface-input rounded-lg p-1">
          <button
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              tab === 'catalog' ? 'bg-surface-muted text-heading' : 'text-body hover:text-label'
            }`}
            onClick={() => setTab('catalog')}
          >
            Catalog
          </button>
          <button
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              tab === 'my-datasets' ? 'bg-surface-muted text-heading' : 'text-body hover:text-label'
            }`}
            onClick={() => setTab('my-datasets')}
          >
            My Datasets
          </button>
        </div>
      </div>

      {tab === 'catalog' ? (
        <CatalogTab
          categoryFilter={categoryFilter}
          setCategoryFilter={setCategoryFilter}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
        />
      ) : (
        <MyDatasetsTab inspectName={inspectName} setInspectName={setInspectName} />
      )}
    </div>
  )
}

function CatalogTab({
  categoryFilter,
  setCategoryFilter,
  searchQuery,
  setSearchQuery,
}: {
  categoryFilter: string
  setCategoryFilter: (c: string) => void
  searchQuery: string
  setSearchQuery: (q: string) => void
}) {
  const { data: catalog, isLoading } = useDataCatalog()
  const downloadDataset = useDownloadDataset()
  const [downloadingId, setDownloadingId] = useState<string | null>(null)

  async function handleDownload(id: string) {
    setDownloadingId(id)
    try {
      await downloadDataset.mutateAsync({ catalogId: id })
    } finally {
      setDownloadingId(null)
    }
  }

  if (isLoading) return <p className="text-caption text-sm">Loading catalog...</p>

  const filtered = (catalog || []).filter((ds: CatalogDataset) => {
    if (categoryFilter !== 'all' && ds.category !== categoryFilter) return false
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      return (
        ds.display_name.toLowerCase().includes(q) ||
        ds.description.toLowerCase().includes(q) ||
        ds.tags.some(t => t.toLowerCase().includes(q))
      )
    }
    return true
  })

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-caption" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search datasets..."
            className="w-full pl-9 pr-3 py-2 bg-surface-input border border-default rounded-lg text-sm text-label placeholder:text-caption focus:outline-none focus:border-indigo-500"
          />
        </div>
        <div className="flex gap-1 flex-wrap">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              className={`px-2.5 py-1 text-xs rounded-full transition-colors ${
                categoryFilter === cat
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                  : 'bg-surface-input text-caption border border-default hover:text-label'
              }`}
              onClick={() => setCategoryFilter(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {filtered.map((ds: CatalogDataset) => (
          <div
            key={ds.id}
            className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4 flex flex-col gap-2"
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-sm font-medium text-label">{ds.display_name}</h3>
                <p className="text-xs text-caption mt-0.5">{ds.source}</p>
              </div>
              {ds.downloaded ? (
                <span className="text-xs text-green-400 bg-green-400/10 px-2 py-0.5 rounded-full">
                  Downloaded
                </span>
              ) : (
                <button
                  className="text-body hover:text-indigo-400 transition-colors p-1 disabled:opacity-50"
                  onClick={() => handleDownload(ds.id)}
                  disabled={downloadingId === ds.id}
                  title="Download"
                >
                  <Download className={`h-4 w-4 ${downloadingId === ds.id ? 'animate-pulse' : ''}`} />
                </button>
              )}
            </div>

            <p className="text-xs text-body leading-relaxed">{ds.description}</p>

            <div className="flex items-center gap-2 flex-wrap">
              <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
                {ds.format}
              </span>
              <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
                {ds.total_samples.toLocaleString()} samples
              </span>
              <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
                {ds.license}
              </span>
            </div>

            <div className="flex gap-1 flex-wrap">
              {ds.tags.map(tag => (
                <span
                  key={tag}
                  className="text-[10px] text-caption bg-surface-muted/50 px-1.5 py-0.5 rounded"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-sm text-caption text-center py-8">
          No datasets match your search.
        </p>
      )}
    </div>
  )
}

function MyDatasetsTab({
  inspectName,
  setInspectName,
}: {
  inspectName: string | null
  setInspectName: (name: string | null) => void
}) {
  const { data: datasets, isLoading } = useDownloadedDatasets()
  const deleteDataset = useDeleteDownloadedDataset()
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  function handleDelete(name: string) {
    if (confirmDelete === name) {
      deleteDataset.mutate(name)
      setConfirmDelete(null)
    } else {
      setConfirmDelete(name)
    }
  }

  if (isLoading) return <p className="text-caption text-sm">Loading datasets...</p>

  if (!datasets || datasets.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-caption text-sm">
          No datasets downloaded yet. Browse the <span className="text-indigo-400">Catalog</span> tab
          or use <code className="bg-surface-input px-1 rounded text-xs">lmforge data download</code>.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((ds: DownloadedDataset) => (
          <div
            key={ds.id}
            className="rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4"
          >
            <div className="flex items-start justify-between mb-1">
              <h3 className="text-sm font-medium text-label">
                {ds.display_name || ds.id}
              </h3>
              <div className="flex gap-1">
                <button
                  className="text-caption hover:text-indigo-400 transition-colors p-1"
                  onClick={() => setInspectName(inspectName === ds.id ? null : ds.id)}
                  title="Inspect"
                >
                  <Eye className="h-4 w-4" />
                </button>
                <button
                  className="text-caption hover:text-red-400 transition-colors p-1"
                  onClick={() => handleDelete(ds.id)}
                  title={confirmDelete === ds.id ? 'Click again to confirm' : 'Delete'}
                >
                  <Trash2
                    className={`h-4 w-4 ${confirmDelete === ds.id ? 'text-red-400' : ''}`}
                  />
                </button>
              </div>
            </div>

            <div className="flex items-center gap-2 mb-2">
              <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
                {ds.format}
              </span>
              <span className="inline-block rounded-full bg-surface-muted px-2 py-0.5 text-xs text-body">
                {ds.origin}
              </span>
            </div>

            <div className="text-xs text-caption space-y-1">
              <p>
                Samples: <span className="text-label">{ds.num_samples.toLocaleString()}</span>
              </p>
              {ds.source && (
                <p className="truncate">
                  Source: <span className="text-body">{ds.source}</span>
                </p>
              )}
            </div>

            {inspectName === ds.id && <SamplePreview name={ds.id} />}
          </div>
        ))}
      </div>
    </div>
  )
}

function SamplePreview({ name }: { name: string }) {
  const { data: samples, isLoading } = useDatasetSamples(name, 3)

  if (isLoading) return <p className="text-xs text-caption mt-2">Loading samples...</p>

  return (
    <div className="mt-3 space-y-2">
      <p className="text-xs text-body font-medium">Sample preview:</p>
      {(samples || []).map((sample, i) => (
        <pre
          key={i}
          className="text-[10px] text-caption bg-surface-overlay rounded p-2 overflow-x-auto max-h-24"
        >
          {JSON.stringify(sample, null, 2).slice(0, 300)}
        </pre>
      ))}
    </div>
  )
}
