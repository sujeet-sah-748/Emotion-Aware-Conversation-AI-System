import { useState, useEffect, useRef, useCallback } from 'react'
import { useSelector } from 'react-redux'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'
import EmptyState from './EmptyState'
import Icon from '../common/Icon'

export default function ChatContainer() {
  const { user } = useSelector(state => state.auth)
  const { chats, activeChatId } = useSelector(state => state.chat)

  const scrollRef    = useRef(null)   // the scrollable messages div
  const bottomRef    = useRef(null)   // sentinel element at the end of messages
  const [text, setText]               = useState('')
  const [atBottom, setAtBottom]       = useState(true)
  const [unreadCount, setUnreadCount] = useState(0)

  const activeChat = chats.find(c => c.id === activeChatId)
  const messages   = activeChat?.messages || []

  // ── Scroll helpers ────────────────────────────────────────────────────────
  const scrollToBottom = useCallback((behavior = 'smooth') => {
    bottomRef.current?.scrollIntoView({ behavior, block: 'end' })
  }, [])

  // Track whether the user is already at (or near) the bottom
  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    const isNearBottom = distanceFromBottom < 80
    setAtBottom(isNearBottom)
    if (isNearBottom) setUnreadCount(0)
  }, [])

  // Auto-scroll when new messages arrive
  useEffect(() => {
    if (messages.length === 0) return

    if (atBottom) {
      // Already at bottom — follow new messages immediately
      scrollToBottom('smooth')
    } else {
      // User has scrolled up — just increment the badge, don't force scroll
      setUnreadCount(n => n + 1)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length])

  // When the chat switches: reset state and jump instantly (no animation)
  useEffect(() => {
    setText('')
    setAtBottom(true)
    setUnreadCount(0)
    // Use a microtask so React has painted the new messages before we scroll
    Promise.resolve().then(() => scrollToBottom('instant'))
  }, [activeChatId, scrollToBottom])

  // ── Empty state ───────────────────────────────────────────────────────────
  if (!activeChatId || messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col min-h-0">
        <EmptyState onPromptSelect={setText} />
        <MessageInput text={text} setText={setText} />
      </div>
    )
  }

  // ── Active chat ───────────────────────────────────────────────────────────
  return (
    <div className="flex-1 flex flex-col min-h-0 relative">

      {/* Scrollable messages area — min-h-0 is critical so flex-1 + overflow works */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto min-h-0 px-4 py-6 space-y-5"
      >
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            userName={user?.name}
          />
        ))}
        {/* Invisible sentinel — scrollIntoView target */}
        <div ref={bottomRef} className="h-px" />
      </div>

      {/* Scroll-to-bottom FAB — appears when user has scrolled up */}
      {!atBottom && (
        <button
          onClick={() => { scrollToBottom('smooth'); setUnreadCount(0) }}
          className="absolute bottom-[72px] left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-color)] shadow-md text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--text-tertiary)] transition-all duration-150"
        >
          <Icon name="chevronDown" size={14} />
          {unreadCount > 0
            ? `${unreadCount} new message${unreadCount > 1 ? 's' : ''}`
            : 'Scroll to bottom'}
        </button>
      )}

      <MessageInput text={text} setText={setText} />
    </div>
  )
}
