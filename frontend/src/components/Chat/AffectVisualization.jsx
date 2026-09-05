import { useSelector } from 'react-redux'

/**
 * AffectVisualization - Displays 3-tier emotional state
 * 
 * Shows:
 * - Situational (immediate/this turn)
 * - Short-term (session mood)
 * - Long-term (personality trait)
 * 
 * For each tier:
 * - Top emotions with percentages
 * - VAD coordinates
 * - Visual bars
 */
export default function AffectVisualization() {
  const affectState = useSelector(state => state.emotion.affectState)
  const emotionalEvents = useSelector(state => state.emotion.emotionalEvents)
  
  if (!affectState) {
    return (
      <div className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)]">
        <div className="text-sm text-[var(--text-muted)] text-center">
          No emotional data yet. Send a message to start tracking.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Header with trend */}
      <div className="flex items-center justify-between px-4 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)]">
        <div>
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">
            Emotional State
          </h3>
          <p className="text-xs text-[var(--text-muted)]">
            3-tier affect tracking
          </p>
        </div>
        <div className="flex items-center gap-2">
          <TrendIndicator trend={affectState.trend} />
          <span className="text-xs text-[var(--text-muted)]">
            {affectState.confidence && `${(affectState.confidence * 100).toFixed(0)}% confident`}
          </span>
        </div>
      </div>

      {/* Three tiers */}
      <TierPanel 
        title="Right Now" 
        subtitle="Situational (5min half-life)"
        bars={affectState.situational_bars}
        vad={affectState.situational_vad}
        tier="situational"
      />
      
      <TierPanel 
        title="Recent Mood" 
        subtitle="Short-term (45min half-life)"
        bars={affectState.short_term_bars}
        vad={affectState.short_term_vad}
        dominant={affectState.stm_dominant}
        tier="short_term"
      />
      
      <TierPanel 
        title="Your Baseline" 
        subtitle="Long-term (3day half-life)"
        bars={affectState.long_term_bars}
        vad={affectState.long_term_vad}
        dominant={affectState.ltm_dominant}
        tier="long_term"
      />

      {/* Recent emotional events */}
      {emotionalEvents.length > 0 && (
        <EventsTimeline events={emotionalEvents.slice(-5)} />
      )}
    </div>
  )
}

function TierPanel({ title, subtitle, bars, vad, dominant, tier }) {
  // Get top 5 emotions
  const topEmotions = Object.entries(bars || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .filter(([, score]) => score > 0.5) // Only show >0.5%

  return (
    <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)]">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h4 className="text-sm font-medium text-[var(--text-primary)]">{title}</h4>
          <p className="text-xs text-[var(--text-muted)]">{subtitle}</p>
        </div>
        {dominant && (
          <span className="px-2 py-1 text-xs font-medium rounded-md bg-[var(--bg-elevated)] text-[var(--text-primary)] capitalize">
            {dominant}
          </span>
        )}
      </div>

      {/* Emotion bars */}
      <div className="space-y-1.5 mb-3">
        {topEmotions.length > 0 ? (
          topEmotions.map(([emotion, score]) => (
            <EmotionBar key={emotion} label={emotion} score={score} tier={tier} />
          ))
        ) : (
          <div className="text-xs text-[var(--text-muted)] text-center py-2">
            Neutral state
          </div>
        )}
      </div>

      {/* VAD coordinates */}
      {vad && (
        <div className="pt-2 border-t border-[var(--border-color)]">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <VADValue label="V" value={vad.valence} tooltip="Valence (negative ↔ positive)" />
            <VADValue label="A" value={vad.arousal} tooltip="Arousal (calm ↔ excited)" />
            <VADValue label="D" value={vad.dominance} tooltip="Dominance (weak ↔ strong)" />
          </div>
        </div>
      )}
    </div>
  )
}

function EmotionBar({ label, score, tier }) {
  const colorMap = {
    // Positive emotions
    joy: 'bg-yellow-500',
    excitement: 'bg-orange-500',
    gratitude: 'bg-green-500',
    love: 'bg-pink-500',
    optimism: 'bg-blue-500',
    pride: 'bg-purple-500',
    relief: 'bg-teal-500',
    
    // Negative emotions
    sadness: 'bg-blue-700',
    anger: 'bg-red-600',
    fear: 'bg-gray-700',
    nervousness: 'bg-yellow-700',
    disappointment: 'bg-gray-600',
    disgust: 'bg-green-800',
    grief: 'bg-gray-800',
    
    // Neutral/mixed
    neutral: 'bg-gray-400',
    confusion: 'bg-purple-400',
    surprise: 'bg-yellow-400',
  }

  const color = colorMap[label] || 'bg-gray-500'
  const widthPercent = Math.min(score, 100)

  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-0.5">
        <span className="text-[var(--text-secondary)] capitalize">{label}</span>
        <span className="text-[var(--text-muted)]">{score.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 w-full bg-[var(--bg-elevated)] rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${widthPercent}%` }}
        />
      </div>
    </div>
  )
}

function VADValue({ label, value, tooltip }) {
  // Color based on value
  const getColor = () => {
    if (value > 0.3) return 'text-green-500'
    if (value < -0.3) return 'text-red-500'
    return 'text-[var(--text-muted)]'
  }

  return (
    <div className="flex flex-col items-center" title={tooltip}>
      <span className="text-[var(--text-muted)] text-xs mb-1">{label}</span>
      <span className={`text-sm font-mono font-semibold ${getColor()}`}>
        {value.toFixed(2)}
      </span>
    </div>
  )
}

function TrendIndicator({ trend }) {
  const icons = {
    rising: '↗',
    falling: '↘',
    steady: '→',
  }

  const colors = {
    rising: 'text-green-500',
    falling: 'text-red-500',
    steady: 'text-[var(--text-muted)]',
  }

  return (
    <div className={`flex items-center gap-1 ${colors[trend] || colors.steady}`}>
      <span className="text-lg">{icons[trend] || icons.steady}</span>
      <span className="text-xs capitalize">{trend}</span>
    </div>
  )
}

function EventsTimeline({ events }) {
  if (!events || events.length === 0) return null

  return (
    <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)]">
      <h4 className="text-sm font-medium text-[var(--text-primary)] mb-2">
        Recent Events
      </h4>
      <div className="space-y-2">
        {events.map((event, idx) => (
          <EventItem key={event.event_id || idx} event={event} />
        ))}
      </div>
    </div>
  )
}

function EventItem({ event }) {
  const kindIcons = {
    spike: '⚡',
    shift: '🔄',
    reinforcement: '✓',
    promotion: '⬆',
  }

  const kindColors = {
    spike: 'text-yellow-500',
    shift: 'text-blue-500',
    reinforcement: 'text-green-500',
    promotion: 'text-purple-500',
  }

  return (
    <div className="flex items-start gap-2 text-xs">
      <span className={`text-lg ${kindColors[event.kind]}`}>
        {kindIcons[event.kind] || '•'}
      </span>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-[var(--text-primary)] capitalize">
            {event.label}
          </span>
          <span className="text-[var(--text-muted)]">
            {event.tier}
          </span>
        </div>
        <div className="text-[var(--text-muted)]">
          {event.kind} • Δ{event.delta_magnitude?.toFixed(2)} • {(event.confidence * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  )
}
