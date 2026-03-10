import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FlaskConical,
  Box,
  Database,
  MessageSquare,
  Settings,
  Plus,
  Moon,
  Sun,
  Monitor,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { useTheme } from '../../hooks/useTheme'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/new', label: 'New Training', icon: Plus },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/models', label: 'Models', icon: Box },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/playground', label: 'Playground', icon: MessageSquare },
  { to: '/settings', label: 'Settings', icon: Settings },
]

const themeIcons = { dark: Moon, light: Sun, system: Monitor }

export default function Sidebar() {
  const { theme, cycleTheme } = useTheme()
  const ThemeIcon = themeIcons[theme]

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-surface-sidebar border-r border-subtle flex flex-col">
      <div className="px-5 py-5">
        <h1 className="text-lg font-bold tracking-tight text-heading">
          MLX Forge <span className="text-indigo-400 font-normal text-sm">Studio</span>
        </h1>
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-indigo-500/10 text-indigo-400'
                  : 'text-body hover:text-heading hover:bg-surface-hover'
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-5 py-4 flex items-center justify-between">
        <span className="text-xs text-muted" id="app-version">v0.2.11</span>
        <button
          onClick={cycleTheme}
          className="p-1.5 rounded-md text-caption hover:text-heading hover:bg-surface-hover transition-colors"
          title={`Theme: ${theme}`}
        >
          <ThemeIcon className="h-4 w-4" />
        </button>
      </div>
    </aside>
  )
}
