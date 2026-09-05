import { useSelector, useDispatch } from 'react-redux'
import {
  setAccentColor,
  toggleDarkMode,
  toggleCompactMode,
  toggleEmotionAlerts,
  toggleMemoryUpdates,
  toggleStoreHistory,
  toggleShareData,
} from '../../store/slices/settingsSlice'
import { accentColors } from '../../utils/emotionColors'
import Icon from '../common/Icon'

function Toggle({ checked, onChange, label, description }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)] last:border-0">
      <div>
        <p className="text-sm font-medium text-[var(--text-primary)]">{label}</p>
        <p className="text-xs text-[var(--text-muted)] mt-0.5">{description}</p>
      </div>
      <div className={`toggle-track ${checked ? 'on' : ''}`} onClick={onChange}>
        <div className="toggle-thumb" />
      </div>
    </div>
  )
}

export default function SettingsPanel({ onBack }) {
  const dispatch = useDispatch()
  const settings = useSelector(state => state.settings)

  return (
    <div className="flex-1 flex flex-col h-full bg-[var(--bg-primary)]">
      <div className="h-14 flex items-center gap-3 px-4 border-b border-[var(--border-color)] flex-shrink-0">
        <button onClick={onBack} className="icon-btn">
          <Icon name="chevronLeft" size={20} />
        </button>
        <h2 className="text-base font-semibold text-[var(--text-primary)]">Settings</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-xl mx-auto space-y-8">
          {/* Appearance */}
          <section>
            <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Appearance
            </h3>

            <div className="flex items-center justify-between py-3 border-b border-[var(--border-color)]">
              <div>
                <p className="text-sm font-medium text-[var(--text-primary)]">Accent color</p>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">Choose your preferred theme color</p>
              </div>
              <div className="flex gap-2">
                {accentColors.map(c => (
                  <button
                    key={c.key}
                    onClick={() => dispatch(setAccentColor(c.key))}
                    className={`color-swatch ${settings.accentColor === c.key ? 'active' : ''}`}
                    style={{ backgroundColor: c.hex }}
                    title={c.label}
                  />
                ))}
              </div>
            </div>

            <Toggle
              checked={settings.darkMode}
              onChange={() => dispatch(toggleDarkMode())}
              label="Dark mode"
              description="Switch between light and dark themes"
            />
            <Toggle
              checked={settings.compactMode}
              onChange={() => dispatch(toggleCompactMode())}
              label="Compact mode"
              description="Reduce spacing for denser layout"
            />
          </section>

          {/* Notifications */}
          <section>
            <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Notifications
            </h3>
            <Toggle
              checked={settings.emotionAlerts}
              onChange={() => dispatch(toggleEmotionAlerts())}
              label="Emotion alerts"
              description="Notify when emotion shifts significantly"
            />
            <Toggle
              checked={settings.memoryUpdates}
              onChange={() => dispatch(toggleMemoryUpdates())}
              label="Memory updates"
              description="Alert when new facts are stored"
            />
          </section>

          {/* Privacy */}
          <section>
            <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Privacy
            </h3>
            <Toggle
              checked={settings.storeHistory}
              onChange={() => dispatch(toggleStoreHistory())}
              label="Store conversation history"
              description="Save chats for memory retrieval"
            />
            <Toggle
              checked={settings.shareData}
              onChange={() => dispatch(toggleShareData())}
              label="Share emotion data"
              description="Contribute to empathy research"
            />
          </section>
        </div>
      </div>
    </div>
  )
}
