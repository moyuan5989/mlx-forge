import Markdown from 'react-markdown'
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
          'rounded-lg px-3 py-2 max-w-[80%] text-sm',
          isUser ? 'bg-indigo-600/20 text-heading whitespace-pre-wrap' : 'bg-surface-input text-label'
        )}
      >
        {isUser ? (
          content
        ) : (
          <div className="prose prose-invert prose-sm max-w-none [&_pre]:bg-surface-muted [&_pre]:rounded [&_pre]:p-2 [&_code]:text-xs [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0">
            <Markdown>{content}</Markdown>
          </div>
        )}
      </div>
    </div>
  )
}
