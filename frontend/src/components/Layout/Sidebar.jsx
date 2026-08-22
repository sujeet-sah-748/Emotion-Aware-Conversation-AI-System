import { useState, useCallback } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { createChat, setActiveChat, deleteChat, renameChat } from '../../store/slices/chatSlice'
import { formatChatDate } from '../../utils/formatters'
import { getInitials } from '../../utils/formatters'
import Icon from '../common/Icon'

export default function Sidebar({ onNavigate }) {
  const dispatch = useDispatch()
  const { user } = useSelector(state => state.auth)
  const { chats, activeChatId } = useSelector(state => state.chat)
  const [searchQuery, setSearchQuery] = useState('')
  const [renamingId, setRenamingId] = useState(null)
  const [renameValue, setRenameValue] = useState('')

  const filteredChats = chats.filter(c => 
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  ).sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))

  const handleNewChat = useCallback(() => {
    dispatch(createChat())
  }, [dispatch])

  const handleSelect = useCallback((id) => {
    dispatch(setActiveChat(id))
  }, [dispatch])

  const handleDelete = useCallback((e, id) => {
    e.stopPropagation()
    dispatch(deleteChat(id))
  }, [dispatch])

  const startRename = useCallback((e, chat) => {
    e.stopPropagation()
    setRenamingId(chat.id)
    setRenameValue(chat.title)
  }, [])

  const submitRename = useCallback((e) => {
    // Prevent double-fire: onSubmit + onBlur both call this; bail if already committed
    if (!renamingId) return
    if (e?.preventDefault) e.preventDefault()
    if (renameValue.trim()) {
      dispatch(renameChat({ chatId: renamingId, title: renameValue.trim() }))
    }
    setRenamingId(null)
    setRenameValue('')
  }, [dispatch, renamingId, renameValue])

  return (
    <aside className="w-72 flex-shrink-0 flex flex-col h-full border-r border-[var(--border-color)] bg-[var(--bg-secondary)]">
      {/* Header */}
      <div className="p-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-[var(--text-primary)] text-[var(--bg-primary)] flex items-center justify-center">
          <Icon name="brain" size={18} />
        </div>
        <h1 className="text-base font-semibold text-[var(--text-primary)] flex-1">EmotionChat</h1>
      </div>

      {/* New Chat */}
      <div className="px-3 pb-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-dashed border-[var(--border-color)] text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--text-tertiary)] hover:bg-[var(--bg-tertiary)] transition-all duration-150"
        >
          <Icon name="plus" size={16} />
          New conversation
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Icon name="search" size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search history..."
            className="w-full pl-9 pr-3 py-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-primary)] text-sm text-[var(--text-primary)] outline-none focus:border-[var(--text-tertiary)] transition-colors"
          />
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {filteredChats.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-[var(--text-muted)] gap-2">
            <Icon name="message" size={28} className="opacity-40" />
            <p className="text-sm">No conversations yet</p>
          </div>
        ) : (
          filteredChats.map(chat => (
            <div
              key={chat.id}
              onClick={() => handleSelect(chat.id)}
              className={`sidebar-item group ${activeChatId === chat.id ? 'active' : ''}`}
            >
              <Icon name="message" size={16} className="flex-shrink-0 opacity-60" />
              <div className="flex-1 min-w-0">
                {renamingId === chat.id ? (
                  <form onSubmit={submitRename} className="flex-1">
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={e => setRenameValue(e.target.value)}
                      onBlur={() => submitRename()}
                      className="w-full bg-transparent text-sm outline-none text-[var(--text-primary)]"
                    />
                  </form>
                ) : (
                  <>
                    <p className="text-sm truncate">{chat.title}</p>
                    <p className="text-xs text-[var(--text-muted)]">{formatChatDate(chat.updatedAt)}</p>
                  </>
                )}
              </div>
              <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button onClick={e => startRename(e, chat)} className="icon-btn w-6 h-6" title="Rename">
                  <Icon name="edit" size={13} />
                </button>
                <button onClick={e => handleDelete(e, chat.id)} className="icon-btn w-6 h-6 text-red-500 hover:text-red-600" title="Delete">
                  <Icon name="trash" size={13} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-[var(--border-color)]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[var(--bg-tertiary)] flex items-center justify-center text-sm font-medium text-[var(--text-primary)]">
            {user ? getInitials(user.name) : '?'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-[var(--text-primary)] truncate">{user?.name || 'Guest'}</p>
            <p className="text-xs text-[var(--text-muted)] truncate">{user?.email || ''}</p>
          </div>
          <button onClick={() => onNavigate('settings')} className="icon-btn" title="Settings">
            <Icon name="settings" size={16} />
          </button>
          <button onClick={() => onNavigate('profile')} className="icon-btn" title="Profile">
            <Icon name="user" size={16} />
          </button>
          <button onClick={() => onNavigate('logout')} className="icon-btn" title="Logout">
            <Icon name="logout" size={16} />
          </button>
        </div>
      </div>
    </aside>
  )
}
