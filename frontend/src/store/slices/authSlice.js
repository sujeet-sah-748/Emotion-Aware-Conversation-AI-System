import { createSlice } from '@reduxjs/toolkit'

function loadUserFromStorage() {
  try {
    const stored = localStorage.getItem('emotionchat_user')
    return stored ? JSON.parse(stored) : null
  } catch {
    return null
  }
}

const initialState = {
  user: loadUserFromStorage(),
  isAuthenticated: !!loadUserFromStorage(),
  loading: false,
  error: null,
}

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart: (state) => {
      state.loading = true
      state.error = null
    },
    loginSuccess: (state, action) => {
      state.user = action.payload
      state.isAuthenticated = true
      state.loading = false
      state.error = null
      localStorage.setItem('emotionchat_user', JSON.stringify(action.payload))
    },
    loginFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    registerSuccess: (state, action) => {
      state.user = action.payload
      state.isAuthenticated = true
      state.loading = false
      state.error = null
      localStorage.setItem('emotionchat_user', JSON.stringify(action.payload))
    },
    logout: (state) => {
      state.user = null
      state.isAuthenticated = false
      state.error = null
      localStorage.removeItem('emotionchat_user')
    },
    updateProfile: (state, action) => {
      state.user = { ...state.user, ...action.payload }
      localStorage.setItem('emotionchat_user', JSON.stringify(state.user))
    },
    clearError: (state) => {
      state.error = null
    },
  },
})

export const { loginStart, loginSuccess, loginFailure, registerSuccess, logout, updateProfile, clearError } = authSlice.actions
export default authSlice.reducer
