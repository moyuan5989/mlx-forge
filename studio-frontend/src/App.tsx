import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeContext, useThemeProvider } from './hooks/useTheme'
import PageLayout from './components/layout/PageLayout'
import Dashboard from './pages/Dashboard'
import Experiments from './pages/Experiments'
import RunDetail from './pages/RunDetail'
import Models from './pages/Models'
import Datasets from './pages/Datasets'
import Playground from './pages/Playground'
import Settings from './pages/Settings'
import NewTraining from './pages/NewTraining'
import JobQueue from './pages/JobQueue'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5_000 },
  },
})

export default function App() {
  const themeCtx = useThemeProvider()

  return (
    <ThemeContext.Provider value={themeCtx}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <PageLayout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/new" element={<NewTraining />} />
              <Route path="/queue" element={<JobQueue />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/experiments/:id" element={<RunDetail />} />
              <Route path="/models" element={<Models />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/playground" element={<Playground />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </PageLayout>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeContext.Provider>
  )
}
