import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, apiV2 } from '../api/client'

// ── Legacy: Processed dataset cache (V1 compat) ──

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: api.getDatasets })
}

export function useDataset(fingerprint: string) {
  return useQuery({
    queryKey: ['datasets', fingerprint],
    queryFn: () => api.getDataset(fingerprint),
    enabled: !!fingerprint,
  })
}

export function useDeleteDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (fp: string) => api.deleteDataset(fp),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

// ── Data Library: Catalog + Downloaded Datasets ──

export function useDataCatalog() {
  return useQuery({
    queryKey: ['data-catalog'],
    queryFn: apiV2.getDataCatalog,
  })
}

export function useDownloadedDatasets() {
  return useQuery({
    queryKey: ['downloaded-datasets'],
    queryFn: apiV2.getDownloadedDatasets,
  })
}

export function useDownloadDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ catalogId, maxSamples }: { catalogId: string; maxSamples?: number }) =>
      apiV2.downloadDataset(catalogId, maxSamples),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['data-catalog'] })
      qc.invalidateQueries({ queryKey: ['downloaded-datasets'] })
    },
  })
}

export function useDatasetSamples(name: string, n: number = 5) {
  return useQuery({
    queryKey: ['dataset-samples', name, n],
    queryFn: () => apiV2.getDatasetSamples(name, n),
    enabled: !!name,
  })
}

export function useDeleteDownloadedDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiV2.deleteDownloadedDataset(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['downloaded-datasets'] })
      qc.invalidateQueries({ queryKey: ['data-catalog'] })
    },
  })
}
