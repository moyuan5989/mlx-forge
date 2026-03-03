import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiV2 } from '../api/client'

export function useQueue() {
  return useQuery({
    queryKey: ['queue'],
    queryFn: apiV2.getQueue,
    refetchInterval: 5_000,
  })
}

export function useQueueStats() {
  return useQuery({
    queryKey: ['queue-stats'],
    queryFn: apiV2.getQueueStats,
    refetchInterval: 5_000,
  })
}

export function useSubmitJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (config: Record<string, unknown>) => apiV2.submitJob(config),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['queue'] })
      qc.invalidateQueries({ queryKey: ['queue-stats'] })
    },
  })
}

export function useCancelJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiV2.cancelJob(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['queue'] })
      qc.invalidateQueries({ queryKey: ['queue-stats'] })
    },
  })
}

export function usePromoteJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiV2.promoteJob(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['queue'] }),
  })
}
