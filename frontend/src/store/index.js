import { configureStore } from '@reduxjs/toolkit'
import authReducer from './slices/authSlice'
import chatReducer from './slices/chatSlice'
import emotionReducer from './slices/emotionSlice'
import settingsReducer from './slices/settingsSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    chat: chatReducer,
    emotion: emotionReducer,
    settings: settingsReducer,
  },
})
