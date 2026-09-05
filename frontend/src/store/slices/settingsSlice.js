import { createSlice } from '@reduxjs/toolkit'

// Pure helper — defined at top so it's available everywhere in this file
function getHue(color) {
  const hues = {
    blue: 220,
    rose: 340,
    emerald: 150,
    amber: 38,
    violet: 260,
    teal: 170,
    orange: 24,
    pink: 320,
  }
  return hues[color] || 220
}

// Apply theme state to the DOM. Called from useEffect in components that
// dispatch settings changes — keeps reducers pure.
export function applyThemeToDom(settings) {
  if (settings.darkMode) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
  document.documentElement.style.setProperty('--accent-hue', getHue(settings.accentColor))
}

const initialState = {
  accentColor: localStorage.getItem('emotionchat_accent') || 'blue',
  darkMode: localStorage.getItem('emotionchat_dark') === 'true',
  compactMode: localStorage.getItem('emotionchat_compact') === 'true',
  emotionAlerts: localStorage.getItem('emotionchat_alerts') !== 'false',
  memoryUpdates: localStorage.getItem('emotionchat_memory') === 'true',
  storeHistory: localStorage.getItem('emotionchat_storehistory') !== 'false',
  shareData: localStorage.getItem('emotionchat_share') === 'true',
}

const settingsSlice = createSlice({
  name: 'settings',
  reducers: {
    setAccentColor: (state, action) => {
      state.accentColor = action.payload
      localStorage.setItem('emotionchat_accent', action.payload)
      // DOM update: caller must call applyThemeToDom(getState().settings) after dispatch
    },
    toggleDarkMode: (state) => {
      state.darkMode = !state.darkMode
      localStorage.setItem('emotionchat_dark', state.darkMode)
    },
    toggleCompactMode: (state) => {
      state.compactMode = !state.compactMode
      localStorage.setItem('emotionchat_compact', state.compactMode)
    },
    toggleEmotionAlerts: (state) => {
      state.emotionAlerts = !state.emotionAlerts
      localStorage.setItem('emotionchat_alerts', state.emotionAlerts)
    },
    toggleMemoryUpdates: (state) => {
      state.memoryUpdates = !state.memoryUpdates
      localStorage.setItem('emotionchat_memory', state.memoryUpdates)
    },
    toggleStoreHistory: (state) => {
      state.storeHistory = !state.storeHistory
      localStorage.setItem('emotionchat_storehistory', state.storeHistory)
    },
    toggleShareData: (state) => {
      state.shareData = !state.shareData
      localStorage.setItem('emotionchat_share', state.shareData)
    },
    // No-op action used as a signal in App.jsx to trigger theme application.
    // The actual state is loaded from localStorage in initialState.
    // DOM theme application is handled by applyThemeToDom() in App.jsx.
    loadSettings: (state) => {
      return state
    },
  },
  initialState,
})

export const {
  setAccentColor,
  toggleDarkMode,
  toggleCompactMode,
  toggleEmotionAlerts,
  toggleMemoryUpdates,
  toggleStoreHistory,
  toggleShareData,
  loadSettings,
} = settingsSlice.actions
export default settingsSlice.reducer
