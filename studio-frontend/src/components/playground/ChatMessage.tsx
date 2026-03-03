import { cn } from '../../lib/utils'
import { User, Bot } from 'lucide-react'

interface Props {
  role: 'user' | 'assistant'
  content: string
}

export default function ChatMessage({ role, content }: Props) {
  const isUser = role === 'user'
  return (
    <div className={cn('flex gap-3 py-3', isUser && 'flex-row-reverse')}>
      <div
        className={cn(
          'flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center',
          isUser ? 'bg-indigo-500/20' : 'bg-surface-muted'
        )}
      >
        {isUser ? (
          <User className="h-3.5 w-3.5 text-indigo-400" />
        ) : (
          <Bot className="h-3.5 w-3.5 text-body" />
        )}
      </div>
      <div
        className={cn(
          'rounded-lg px-3 py-2 max-w-[80%] text-sm whitespace-pre-wrap',
          isUser ? 'bg-indigo-600/20 text-heading' : 'bg-surface-input text-label'
        )}
      >
        {content}
      </div>
    </div>
  )
}
