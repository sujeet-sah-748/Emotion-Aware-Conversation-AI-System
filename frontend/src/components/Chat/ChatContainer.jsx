import { useState, useEffect, useRef, useCallback } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'
import EmptyState from './EmptyState'
import Icon from '../common/Icon'
import PredictionPanel from './PredictionPanel'
import AffectVisualization from './AffectVisualization'
import { resetEmotion } from '../../store/slices/emotionSlice'

export default function ChatContainer() {
  const { user } = useSelector(state => state.auth)
  const { chats, activeChatId } = useSelector(state => state.chat)
  const affectState = useSelector(state => state.emotion.affectState)
  const dispatch = useDispatch()

  const scrollRef    = useRef(null)   // the scrollable messages div
  const bottomRef    = useRef(null)   // sentinel element at the end of messages
  const messagesLengthRef = useRef(0) // track messages length to prevent unnecessary scrolls
  const [text, setText]               = useState('')
  const [atBottom, setAtBottom]       = useState(true)
  const [unreadCount, setUnreadCount] = useState(0)
  const [showAffect, setShowAffect]   = useState(false) // Toggle for affect panel

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
    
    const lengthChanged = messagesLengthRef.current !== messages.length
    messagesLengthRef.current = messages.length
    
    if (!lengthChanged) return

    const lastMessage = messages[messages.length - 1]
    if (atBottom) {
      scrollToBottom('smooth')
    } else if (lastMessage?.role === 'bot') {
      // Only count incoming bot messages, not the user's own outgoing messages
      setUnreadCount(n => n + 1)
    }
  }, [messages, messages.length, atBottom, scrollToBottom])

  // When the chat switches: reset state and jump instantly (no animation)
  useEffect(() => {
    setText('')
    setAtBottom(true)
    setUnreadCount(0)
    dispatch(resetEmotion())
    // Use a macrotask so React has painted the new messages before we scroll
    setTimeout(() => scrollToBottom('instant'), 0)
  }, [activeChatId, scrollToBottom, dispatch])

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
      
      {/* Header with affect toggle */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border-color)] bg-[var(--bg-primary)]">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-[var(--text-primary)]">
            {activeChat?.title || 'Conversation'}
          </h2>
          <span className="text-xs text-[var(--text-muted)]">
            {messages.length} messages
          </span>
        </div>
        
        {/* Toggle affect panel button */}
        <button
          onClick={() => setShowAffect(!showAffect)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            showAffect 
              ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] border border-[var(--border-color)]' 
              : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]'
          }`}
          title={showAffect ? 'Hide emotional state' : 'Show emotional state'}
        >
          <span className="text-base">{affectState ? '📊' : '💭'}</span>
          {showAffect ? 'Hide Affect' : 'Show Affect'}
        </button>
      </div>

      <div className="flex-1 flex min-h-0">
        {/* Main messages area */}
        <div className={`flex-1 flex flex-col min-h-0 transition-all duration-300 ${showAffect ? 'mr-80' : ''}`}>
          {/* Scrollable messages area — min-h-0 is critical so flex-1 + overflow works */}
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="flex-1 overflow-y-auto min-h-0 px-4 py-6 space-y-5"
          >
            {messages.map((msg) =>
              msg.role === 'prediction' ? (
                <PredictionPanel key={msg.id} emotions={msg.emotions} />
              ) : (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  userName={user?.name}
                />
              )
            )}
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

        {/* Affect state panel (right sidebar) */}
        {showAffect && (
          <div className="fixed right-0 top-[48px] bottom-0 w-80 border-l border-[var(--border-color)] bg-[var(--bg-primary)] overflow-y-auto p-4">
            <AffectVisualization />
          </div>
        )}
      </div>
    </div>
  )
}
