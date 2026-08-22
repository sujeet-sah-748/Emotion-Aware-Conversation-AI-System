import Icon from '../common/Icon'

// Prompts are organised to cover the main GoEmotions label groups the model
// was trained on — positive, negative, and mixed/transitional states.
// Each prompt is designed to naturally elicit that emotion category so the
// model produces a meaningful, non-neutral prediction on first send.
const SUGGESTION_PROMPTS = [
  // Negative — most common reason people open a support chatbot
  { label: 'Stressed',       text: "I'm completely overwhelmed by everything on my plate right now." },
  { label: 'Angry',          text: "Something unfair happened to me today and I can't stop thinking about it." },
  { label: 'Anxious',        text: "I have a big presentation tomorrow and I can't stop worrying about it." },
  // Positive
  { label: 'Excited',        text: "I just got some amazing news and I can barely contain myself!" },
  { label: 'Grateful',       text: "I want to talk about someone who made a real difference in my life." },
  // Mixed / transitional
  { label: 'Mixed feelings', text: "I'm moving to a new city next week — excited but also really scared to leave everything behind." },
]

export default function EmptyState({ onPromptSelect }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-[var(--text-muted)] gap-5 px-4 animate-fade-in">
      <div className="w-16 h-16 rounded-2xl bg-[var(--bg-tertiary)] flex items-center justify-center">
        <Icon name="brain" size={32} className="opacity-40" />
      </div>

      <div className="text-center">
        <h3 className="text-base font-medium text-[var(--text-primary)] mb-1">
          Start a conversation
        </h3>
        <p className="text-sm max-w-xs">
          Share how you're feeling and the AI will respond with empathy based on your emotion.
        </p>
      </div>

      {/* Suggestion grid — two rows of three */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 w-full max-w-2xl">
        {SUGGESTION_PROMPTS.map((prompt) => (
          <button
            key={prompt.label}
            onClick={() => onPromptSelect(prompt.text)}
            className="flex flex-col gap-1 px-3 py-2.5 rounded-xl text-left bg-[var(--bg-tertiary)] hover:bg-[var(--border-color)] transition-all duration-150 group"
          >
            <span className="text-[11px] font-medium text-[var(--text-tertiary)] group-hover:text-[var(--text-secondary)] transition-colors uppercase tracking-wide">
              {prompt.label}
            </span>
            <span className="text-xs text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors leading-relaxed line-clamp-2">
              {prompt.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
