import { useSelector } from 'react-redux'
import { emotionLabels, emotionColors } from '../../utils/emotionColors'
import Icon from '../common/Icon'

export default function Header({ onMenuClick }) {
  const { activeChatId, chats } = useSelector(state => state.chat)
  const { currentEmotion } = useSelector(state => state.emotion)

  const activeChat = chats.find(c => c.id === activeChatId)
  const emotionColor = emotionColors[currentEmotion] || emotionColors.neutral
  const emotionLabel = emotionLabels[currentEmotion] || 'Neutral'

  return (
    <header className="h-14 flex items-center gap-3 px-4 border-b border-[var(--border-color)] bg-[var(--bg-primary)] flex-shrink-0">
      <button onClick={onMenuClick} className="icon-btn lg:hidden" aria-label="Open menu">
        <Icon name="menu" size={18} />
      </button>
      <h2 className="text-sm font-medium text-[var(--text-primary)] flex-1 truncate">
        {activeChat?.title || 'New conversation'}
      </h2>
      <div 
        className="emotion-badge"
        style={{ 
          backgroundColor: `${emotionColor}18`,
          color: emotionColor 
        }}
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: emotionColor }} />
        {emotionLabel}
      </div>
    </header>
  )
}
