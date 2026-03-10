import { useState, useRef, useEffect, useCallback } from 'react'
import { RotateCcw } from 'lucide-react'
import { useModels, useAdapters } from '../hooks/useModels'
import { useWebSocket } from '../hooks/useWebSocket'
import ChatMessage from '../components/playground/ChatMessage'
import ChatInput from '../components/playground/ChatInput'
import GenerationConfig from '../components/playground/GenerationConfig'
import type { WsInferenceMessage } from '../api/types'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function Playground() {
  const { data: models } = useModels()
  const { data: adapters } = useAdapters()
  const supportedModels = (models ?? []).filter((m) => m.supported)

  const [messages, setMessages] = useState<Message[]>([])
  const [generating, setGenerating] = useState(false)
  const [stats, setStats] = useState<{ numTokens: number; tokPerSec: number } | null>(null)
  const [config, setConfig] = useState({
    model: '',
    adapter: '',
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 512,
  })

  const generatingRef = useRef(false)
  const startTimeRef = useRef(0)
  const tokenCountRef = useRef(0)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleWsMessage = useCallback((raw: unknown) => {
    const msg = raw as WsInferenceMessage
    if (msg.type === 'token') {
      tokenCountRef.current += 1
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last?.role === 'assistant') {
          return [...prev.slice(0, -1), { ...last, content: last.content + msg.text }]
        }
        return [...prev, { role: 'assistant', content: msg.text }]
      })
    } else if (msg.type === 'done') {
      generatingRef.current = false
      setGenerating(false)
      const elapsed = (performance.now() - startTimeRef.current) / 1000
      setStats({
        numTokens: tokenCountRef.current,
        tokPerSec: elapsed > 0 ? tokenCountRef.current / elapsed : 0,
      })
    } else if (msg.type === 'error') {
      generatingRef.current = false
      setGenerating(false)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${msg.detail}` },
      ])
    }
  }, [])

  const { send, connected } = useWebSocket({
    url: '/ws/inference',
    onMessage: handleWsMessage,
    enabled: true,
  })

  function handleSend(text: string) {
    if (!config.model) return

    const userMsg: Message = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setGenerating(true)
    setStats(null)
    generatingRef.current = true
    startTimeRef.current = performance.now()
    tokenCountRef.current = 0

    send({
      type: 'generate',
      model: config.model,
      messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
      config: {
        adapter: config.adapter || undefined,
        temperature: config.temperature,
        top_p: config.topP,
        max_tokens: config.maxTokens,
      },
    })
  }

  function handleClear() {
    setMessages([])
    setStats(null)
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-heading">Playground</h2>

      <div className="flex gap-4 h-[calc(100vh-10rem)]">
        {/* Chat area */}
        <div className="flex-1 flex flex-col rounded-lg border border-subtle bg-surface-overlay overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-1">
            {messages.length === 0 && (
              <p className="text-sm text-muted text-center mt-20">
                {config.model
                  ? 'Send a message to start chatting.'
                  : 'Select a model to get started.'}
              </p>
            )}
            {messages.map((msg, i) => (
              <ChatMessage key={i} role={msg.role} content={msg.content} />
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="border-t border-subtle p-3 flex gap-2 items-end">
            <div className="flex-1">
              <ChatInput onSend={handleSend} disabled={generating || !config.model || !connected} />
            </div>
            <button
              className="rounded-md border border-default p-2 text-body hover:text-label hover:bg-surface-hover transition-colors"
              onClick={handleClear}
              title="Clear conversation"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Config panel */}
        <div className="w-72 rounded-lg border border-subtle bg-surface-card shadow-[var(--shadow-card)] p-4 overflow-y-auto">
          <h3 className="text-sm font-medium text-label mb-4">Generation Settings</h3>
          <GenerationConfig
            config={config}
            onChange={setConfig}
            models={supportedModels}
            adapters={adapters ?? []}
            stats={stats}
          />
          {!connected && (
            <p className="text-xs text-amber-400 mt-3">WebSocket disconnected. Reconnecting...</p>
          )}
        </div>
      </div>
    </div>
  )
}
