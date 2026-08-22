import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  currentEmotion: 'neutral',
  confidence: 0,
  emotionHistory: [],
  trajectory: 'stable',
}

const emotionSlice = createSlice({
  name: 'emotion',
  initialState,
  reducers: {
    setCurrentEmotion: (state, action) => {
      const { emotion, confidence } = action.payload
      state.currentEmotion = emotion
      state.confidence = confidence
      state.emotionHistory.push({
        emotion,
        confidence,
        timestamp: new Date().toISOString(),
      })
    },
    setTrajectory: (state, action) => {
      state.trajectory = action.payload
    },
    clearEmotionHistory: (state) => {
      state.emotionHistory = []
      state.currentEmotion = 'neutral'
      state.confidence = 0
    },
  },
})

export const { setCurrentEmotion, setTrajectory, clearEmotionHistory } = emotionSlice.actions
export default emotionSlice.reducer
