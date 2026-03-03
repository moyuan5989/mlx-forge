import { useQuery } from '@tanstack/react-query'
import { api, apiV2 } from '../api/client'

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: api.getModels })
}

export function useSupportedArchitectures() {
  return useQuery({ queryKey: ['models', 'supported'], queryFn: api.getSupportedArchitectures })
}

export function useModelLibrary() {
  return useQuery({ queryKey: ['model-library'], queryFn: apiV2.getModelLibrary })
}
