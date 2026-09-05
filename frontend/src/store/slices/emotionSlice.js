import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  // Legacy fields (backward compatibility)
  currentEmotion: 'neutral',
  confidence: 0,
  emotionHistory: [],
  trajectory: 'stable',
  lastPrediction: [],
  
  // NEW: Full affect state from Task 1.2
  affectState: null,
  emotionalEvents: [],
  sessionInfo: null,
}

const emotionSlice = createSlice({
  name: 'emotion',
  initialState,
  reducers: {
    setCurrentEmotion: (state, action) => {
      const { emotion, confidence, emotions, affectState, emotionalEvents, sessionInfo } = action.payload
      
      // Update legacy fields
      state.currentEmotion = emotion
      state.confidence = confidence
      
      // Store the full scored array if provided
      if (emotions && emotions.length > 0) {
        state.lastPrediction = emotions
      }
      
      // NEW: Store full affect state
      if (affectState) {
        state.affectState = affectState
        // Use trend from affect state if available
        state.trajectory = affectState.trend || state.trajectory
      }
      
      // NEW: Store emotional events
      if (emotionalEvents) {
        state.emotionalEvents = [
          ...state.emotionalEvents,
          ...emotionalEvents
        ]
        // Cap to last 100 events
        if (state.emotionalEvents.length > 100) {
          state.emotionalEvents = state.emotionalEvents.slice(-100)
        }
      }
      
      // NEW: Store session info
      if (sessionInfo) {
        state.sessionInfo = sessionInfo
      }
      
      // Update history
      state.emotionHistory.push({
        emotion,
        confidence,
        timestamp: new Date().toISOString(),
        // Include affect state snapshot
        affectState: affectState ? {
          situational: affectState.stm_dominant,
          shortTerm: affectState.stm_dominant,
          longTerm: affectState.ltm_dominant,
          trend: affectState.trend
        } : null
      })
      
      // Cap history to prevent unbounded growth
      if (state.emotionHistory.length > 100) {
        state.emotionHistory = state.emotionHistory.slice(-100)
      }
    },
    
    setAffectState: (state, action) => {
      // Direct update of affect state (for polling/refresh)
      state.affectState = action.payload
      if (action.payload?.trend) {
        state.trajectory = action.payload.trend
      }
    },
    
    setTrajectory: (state, action) => {
      state.trajectory = action.payload
    },
    
    clearEmotionHistory: (state) => {
      state.emotionHistory = []
      state.currentEmotion = 'neutral'
      state.confidence = 0
      state.lastPrediction = []
      state.emotionalEvents = []
      state.affectState = null
      state.sessionInfo = null
    },
    
    resetEmotion: (state) => {
      state.currentEmotion = 'neutral'
      state.confidence = 0
      state.lastPrediction = []
      // Keep affect state and events on reset (just clear current emotion)
    },
  },
})

export const { 
  setCurrentEmotion, 
  setAffectState,
  setTrajectory, 
  clearEmotionHistory, 
  resetEmotion 
} = emotionSlice.actions

export default emotionSlice.reducer
