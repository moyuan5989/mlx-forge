import { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react'

export type Theme = 'dark' | 'light' | 'system'
export type ResolvedTheme = 'dark' | 'light'

interface ThemeContextValue {
  theme: Theme
  setTheme: (t: Theme) => void
  resolvedTheme: ResolvedTheme
  cycleTheme: () => void
}

export const ThemeContext = createContext<ThemeContextValue>({
  theme: 'dark',
  setTheme: () => {},
  resolvedTheme: 'dark',
  cycleTheme: () => {},
})

export function useTheme() {
  return useContext(ThemeContext)
}

function getSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function resolve(theme: Theme): ResolvedTheme {
  return theme === 'system' ? getSystemTheme() : theme
}

function applyTheme(resolved: ResolvedTheme) {
  document.documentElement.classList.toggle('light', resolved === 'light')
}

export function useThemeProvider(): ThemeContextValue {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem('lmforge-theme')
    if (stored === 'dark' || stored === 'light' || stored === 'system') return stored
    return 'dark'
  })

  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(() => resolve(theme))

  const setTheme = useCallback((t: Theme) => {
    localStorage.setItem('lmforge-theme', t)
    setThemeState(t)
    const r = resolve(t)
    setResolvedTheme(r)
    applyTheme(r)
  }, [])

  const cycleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark')
  }, [theme, setTheme])

  // Apply on mount
  useEffect(() => {
    applyTheme(resolve(theme))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for system theme changes when in system mode
  useEffect(() => {
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const handler = () => {
      const r = getSystemTheme()
      setResolvedTheme(r)
      applyTheme(r)
    }
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [theme])

  return useMemo(() => ({ theme, setTheme, resolvedTheme, cycleTheme }), [theme, setTheme, resolvedTheme, cycleTheme])
}
