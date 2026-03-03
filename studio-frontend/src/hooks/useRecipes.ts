import { useQuery, useMutation } from '@tanstack/react-query'
import { apiV2 } from '../api/client'

export function useRecipes() {
  return useQuery({ queryKey: ['recipes'], queryFn: apiV2.getRecipes })
}

export function useRecipe(id: string) {
  return useQuery({
    queryKey: ['recipes', id],
    queryFn: () => apiV2.getRecipe(id),
    enabled: !!id,
  })
}

export function useResolveRecipe() {
  return useMutation({
    mutationFn: ({ recipeId, body }: { recipeId: string; body: Record<string, unknown> }) =>
      apiV2.resolveRecipe(recipeId, body),
  })
}
