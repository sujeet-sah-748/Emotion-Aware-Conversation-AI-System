import { useState } from 'react'
import { emotionColors, emotionLabels } from '../../utils/emotionColors'
import { formatMessageTime } from '../../utils/formatters'
import Icon from '../common/Icon'

// Format a 0–1 score as a percentage string, e.g. 0.782 → "78%"
function fmtScore(score) {
  return `${Math.round(score * 100)}%`
}

// Returns a color for a raw GoEmotions label.
// Falls back to a neutral grey for any label not in the map — future-proof
// against the model returning unexpected labels.
function getEmotionColor(label) {
  return emotionColors[label] ?? emotionColors.neutral
}

// Returns a human-readable display name for a raw GoEmotions label.
// Falls back to capitalised label name so nothing ever shows blank.
function getEmotionLabel(label) {
  return emotionLabels[label] ?? (label.charAt(0).toUpperCase() + label.slice(1))
}

export default function MessageBubble({ message, userName }) {
  const [copied, setCopied] = useState(false)
  const [showAllEmotions, setShowAllEmotions] = useState(false)
  const isUser = message.role === 'user'

  // ── Emotion data ────────────────────────────────────────────────────────
  // Prefer the `emotions` array stored by chatSlice (full scored list from model).
  // Fall back to constructing a single-entry array from `emotion` string for
  // old messages that predate this change (e.g. data still in localStorage).
  const emotionsArray = (message.emotions && message.emotions.length > 0)
    ? message.emotions
    : [{ label: message.emotion || 'neutral', score: 1.0 }]

  const topEmotion = emotionsArray[0]
  const topColor   = getEmotionColor(topEmotion.label)
  const topLabel   = getEmotionLabel(topEmotion.label)

  // Only show secondary emotions for user messages where the model returned
  // more than one label above a meaningful threshold (≥0.15)
  const secondaryEmotions = isUser
    ? emotionsArray.slice(1).filter(e => e.score >= 0.15)
    : []

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} max-w-[85%] ${isUser ? 'ml-auto' : 'mr-auto'} animate-slide-up`}>

      {/* Avatar */}
      <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-medium ${
        isUser
          ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]'
          : 'bg-[var(--text-primary)] text-[var(--bg-primary)]'
      }`}>
        {isUser ? (userName?.[0] || 'U') : <Icon name="brain" size={14} />}
      </div>

      {/* Bubble + Meta */}
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} gap-1`}>
        <div className={isUser ? 'msg-bubble-user group' : 'msg-bubble-bot group'}>
          <p className="whitespace-pre-wrap">{message.text}</p>

          {/* Hover actions */}
          <div className={`flex items-center gap-1 mt-2 pt-2 border-t ${
            isUser ? 'border-white/10' : 'border-[var(--border-color)]'
          } opacity-0 group-hover:opacity-100 transition-opacity`}>
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Copy"
              aria-label="Copy message"
            >
              <Icon
                name={copied ? 'check' : 'copy'}
                size={12}
                className={isUser ? 'text-white/70' : 'text-[var(--text-tertiary)]'}
              />
            </button>
            {!isUser && (
              <>
                <button className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors" title="Like" aria-label="Like message">
                  <Icon name="thumbsUp" size={12} className="text-[var(--text-tertiary)]" />
                </button>
                <button className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors" title="Dislike" aria-label="Dislike message">
                  <Icon name="thumbsDown" size={12} className="text-[var(--text-tertiary)]" />
                </button>
                <button className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors" title="Regenerate" aria-label="Regenerate response">
                  <Icon name="rotate" size={12} className="text-[var(--text-tertiary)]" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* ── Emotion metadata ───────────────────────────────────────────── */}
        <div className={`flex flex-col gap-1 px-1 ${isUser ? 'items-end' : 'items-start'}`}>

          {/* Top emotion + score + time — always visible */}
          <div className="flex items-center gap-2">
            {/* Top emotion badge — driven by model output */}
            <button
              onClick={() => secondaryEmotions.length > 0 && setShowAllEmotions(v => !v)}
              className={`flex items-center gap-1 text-[11px] text-[var(--text-muted)] ${
                secondaryEmotions.length > 0 ? 'cursor-pointer hover:text-[var(--text-secondary)] transition-colors' : 'cursor-default'
              }`}
              title={secondaryEmotions.length > 0 ? 'Click to see all detected emotions' : undefined}
            >
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: topColor }}
              />
              <span>{topLabel}</span>
              {/* Show confidence score when it came from the model */}
              {topEmotion.score < 0.99 && (
                <span className="opacity-60">{fmtScore(topEmotion.score)}</span>
              )}
              {secondaryEmotions.length > 0 && (
                <span className="opacity-50 ml-0.5">{showAllEmotions ? '▲' : '▼'}</span>
              )}
            </button>

            <span className="text-[11px] text-[var(--text-muted)]">
              {formatMessageTime(message.timestamp)}
            </span>
          </div>

          {/* Secondary emotion chips — shown on expand, driven by model scores */}
          {showAllEmotions && secondaryEmotions.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-0.5">
              {secondaryEmotions.map(e => (
                <span
                  key={e.label}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                >
                  <span
                    className="w-1 h-1 rounded-full flex-shrink-0"
                    style={{ backgroundColor: getEmotionColor(e.label) }}
                  />
                  {getEmotionLabel(e.label)}
                  <span className="opacity-60">{fmtScore(e.score)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
