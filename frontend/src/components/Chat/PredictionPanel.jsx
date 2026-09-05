import { useState } from 'react'
import { emotionColors, emotionLabels } from '../../utils/emotionColors'
import Icon from '../common/Icon'

function getColor(label) {
  return emotionColors[label] ?? emotionColors.neutral
}

function getLabel(label) {
  return emotionLabels[label] ?? (label.charAt(0).toUpperCase() + label.slice(1))
}

// Rendered inline in the message scroll area between user message and bot reply.
// Receives the emotions array directly as a prop — no Redux dependency.
export default function PredictionPanel({ emotions }) {
  const [open, setOpen] = useState(false)

  if (!emotions || emotions.length === 0) return null

  const topLabel = emotions[0]?.label ?? 'neutral'
  const topColor = getColor(topLabel)
  const maxScore = emotions[0]?.score ?? 1

  return (
    <div className="flex justify-center px-4">
      <div className="w-full max-w-[85%]">
        {/* Toggle button */}
        <button
          onClick={() => setOpen(v => !v)}
          className="w-full flex items-center justify-between px-3 py-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors group"
        >
          <div className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: topColor }}
            />
            <span className="text-xs font-medium text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors">
              Model prediction
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              — {getLabel(topLabel)} ({Math.round((emotions[0]?.score ?? 0) * 100)}%)
            </span>
          </div>
          <Icon
            name="chevronDown"
            size={14}
            className={`text-[var(--text-muted)] transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          />
        </button>

        {/* Collapsible body */}
        {open && (
          <div className="mt-1 px-3 py-3 rounded-xl border border-[var(--border-color)] bg-[var(--bg-secondary)] space-y-2.5">
            <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1">
              Top {emotions.length} detected emotions
            </p>

            {emotions.map((item, idx) => {
              const pct = Math.round(item.score * 100)
              const barWidth = maxScore > 0 ? (item.score / maxScore) * 100 : 0
              const color = getColor(item.label)
              const isTop = idx === 0

              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      {isTop && (
                        <span className="text-[9px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-muted)]">
                          Top
                        </span>
                      )}
                      <span
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-xs text-[var(--text-primary)] font-medium">
                        {getLabel(item.label)}
                      </span>
                      <span className="text-[11px] text-[var(--text-muted)]">
                        ({item.label})
                      </span>
                    </div>
                    <span className="text-xs font-semibold tabular-nums" style={{ color }}>
                      {pct}%
                    </span>
                  </div>

                  <div className="h-1.5 w-full rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${barWidth}%`, backgroundColor: color }}
                    />
                  </div>
                </div>
              )
            })}

            <p className="text-[11px] text-[var(--text-muted)] pt-1 border-t border-[var(--border-color)]">
              Scores are sigmoid probabilities from the LoRA classifier. The highest scoring label drives the response.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
