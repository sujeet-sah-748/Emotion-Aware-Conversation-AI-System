import { useState } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { clearAllChats } from '../../store/slices/chatSlice'
import { updateProfile, logout } from '../../store/slices/authSlice'
import { emotionColors } from '../../utils/emotionColors'
import { getInitials } from '../../utils/formatters'
import Icon from '../common/Icon'

export default function ProfilePanel({ onBack }) {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const user = useSelector(state => state.auth.user)
  const { chats } = useSelector(state => state.chat)
  const [editingName, setEditingName] = useState(false)
  // Derive name directly from selector so it stays in sync with external updates
  const [nameValue, setNameValue] = useState(user?.name || '')
  const [confirmClear, setConfirmClear] = useState(false)

  const totalMessages = chats.reduce((sum, c) => sum + c.messages.length, 0)

  const handleSaveName = () => {
    if (nameValue.trim()) {
      dispatch(updateProfile({ name: nameValue.trim() }))
    } else {
      // Reset to current stored name if the field was cleared
      setNameValue(user?.name || '')
    }
    setEditingName(false)
  }

  const handleClearHistory = () => {
    dispatch(clearAllChats())
    setConfirmClear(false)
  }

  const handleLogout = () => {
    dispatch(logout())
    navigate('/login')
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-[var(--bg-primary)]">
      <div className="h-14 flex items-center gap-3 px-4 border-b border-[var(--border-color)] flex-shrink-0">
        <button onClick={onBack} className="icon-btn">
          <Icon name="chevronLeft" size={20} />
        </button>
        <h2 className="text-base font-semibold text-[var(--text-primary)]">Profile</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-xl mx-auto">
          {/* Profile Header */}
          <div className="text-center pb-8 border-b border-[var(--border-color)]">
            <div className="w-20 h-20 rounded-full bg-[var(--bg-tertiary)] flex items-center justify-center text-2xl font-semibold text-[var(--text-primary)] mx-auto mb-3">
              {user ? getInitials(user.name) : '?'}
            </div>
            {editingName ? (
              <div className="flex items-center justify-center gap-2 mb-1">
                <input
                  autoFocus
                  value={nameValue}
                  onChange={e => setNameValue(e.target.value)}
                  onBlur={handleSaveName}
                  onKeyDown={e => e.key === 'Enter' && handleSaveName()}
                  className="text-center text-lg font-semibold bg-transparent border-b border-[var(--text-primary)] text-[var(--text-primary)] outline-none px-2"
                />
              </div>
            ) : (
              <h3
                className="text-lg font-semibold text-[var(--text-primary)] cursor-pointer hover:opacity-70 transition-opacity"
                onClick={() => {
                  setNameValue(user?.name || '')
                  setEditingName(true)
                }}
                title="Click to edit"
              >
                {user?.name || 'Guest'}
              </h3>
            )}
            <p className="text-sm text-[var(--text-muted)]">{user?.email || ''}</p>

            <div className="flex justify-center gap-10 mt-6">
              <div className="text-center">
                <p className="text-2xl font-semibold text-[var(--text-primary)]">{chats.length}</p>
                <p className="text-xs text-[var(--text-muted)] mt-1">Conversations</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-semibold text-[var(--text-primary)]">{totalMessages}</p>
                <p className="text-xs text-[var(--text-muted)] mt-1">Messages</p>
              </div>
            </div>

            <div className="flex flex-wrap justify-center gap-2 mt-5">
              {Object.entries(emotionColors).slice(0, 5).map(([key, color]) => (
                <span key={key} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </span>
              ))}
            </div>
          </div>

          {/* Account */}
          <section className="mt-8">
            <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Account
            </h3>
            <div className="space-y-1">
              <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">Display name</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">{user?.name}</p>
                </div>
                <button onClick={() => { setNameValue(user?.name || ''); setEditingName(true) }} className="icon-btn">
                  <Icon name="edit" size={14} />
                </button>
              </div>
              <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">Email address</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">{user?.email}</p>
                </div>
              </div>
              <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">Member since</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">August 2026</p>
                </div>
              </div>
            </div>
          </section>

          {/* Danger Zone */}
          <section className="mt-8">
            <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Danger zone
            </h3>
            <div className="space-y-1">
              <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">Delete all chat history</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">This action cannot be undone</p>
                </div>
                {confirmClear ? (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-muted)]">Are you sure?</span>
                    <button
                      onClick={handleClearHistory}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500 text-white hover:bg-red-600 transition-colors"
                    >
                      Yes, delete
                    </button>
                    <button
                      onClick={() => setConfirmClear(false)}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium border border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmClear(true)}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium border border-red-500 text-red-500 hover:bg-red-500/10 transition-colors"
                  >
                    Delete
                  </button>
                )}
              </div>
              <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">Log out</p>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">Sign out of your account</p>
                </div>
                <button
                  onClick={handleLogout}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                  Log out
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
