import { useQuery, useMutation } from '@tanstack/react-query'
import { apiV2 } from '../api/client'

export function useHardware() {
  return useQuery({ queryKey: ['hardware'], queryFn: apiV2.getHardware })
}

export function useCompatibleModels() {
  return useQuery({ queryKey: ['compatible-models'], queryFn: apiV2.getCompatibleModels })
}

export function useMemoryEstimate() {
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => apiV2.estimateMemory(body),
  })
}
