// API client for the FastAPI backend at localhost:8000
// Vite proxies /api → http://localhost:8000 (prefix stripped) in dev

/**
 * Sends text to the backend LoRA emotion classifier.
 *
 * Returns:
 *   topEmotion   — raw GoEmotions label string of highest-scoring emotion
 *   emotions     — full array sorted by score desc: [{label, score}, ...]
 *                  up to top_k=5 entries (enough for mixed-emotion detection)
 *   usedFallback — true when nothing crossed the threshold and the model
 *                  just returned its best guess (low-confidence prediction)
 *
 * Throws on non-2xx so callers can catch and fall back to local detection.
 *
 * @param {string} text
 * @returns {Promise<{
 *   topEmotion: string,
 *   emotions: Array<{label: string, score: number}>,
 *   usedFallback: boolean
 * }>}
 */
export async function predictEmotion(text) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // top_k:5 gives enough labels to detect mixed emotional states
    // (e.g. excitement + sadness in a bittersweet message)
    body: JSON.stringify({ text, top_k: 5 }),
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `Backend error: ${response.status}`)
  }

  const data = await response.json()
  // emotions is already sorted descending by score from the backend
  const topEmotion = data.emotions[0]?.label || 'neutral'
  return {
    topEmotion,
    emotions: data.emotions,         // raw GoEmotions labels + scores
    usedFallback: data.used_fallback,
  }
}

/**
 * NEW: Chat with full emotion tracking (Task 1.2)
 * 
 * Sends text to the /chat endpoint which includes:
 * - Multi-label emotion detection
 * - 3-tier affect tracking (Situational/Short-term/Long-term)
 * - VAD coordinates per tier
 * - Emotional event logging
 * - Empathetic bot response
 * 
 * @param {string} text - User message
 * @param {string} userId - User identifier for session management
 * @returns {Promise<{
 *   text: string,
 *   emotions: Array<{label: string, score: number}>,
 *   usedFallback: boolean,
 *   botResponse: string,
 *   affectState: {
 *     situational_vad: {valence: number, arousal: number, dominance: number},
 *     short_term_vad: {valence: number, arousal: number, dominance: number},
 *     long_term_vad: {valence: number, arousal: number, dominance: number},
 *     situational_bars: {[label: string]: number},
 *     short_term_bars: {[label: string]: number},
 *     long_term_bars: {[label: string]: number},
 *     stm_dominant: string,
 *     ltm_dominant: string,
 *     trend: string,
 *     confidence: number
 *   },
 *   emotionalEvents: Array<object>,
 *   sessionInfo: {message_count: number, session_age_seconds: number}
 * }>}
 */
export async function chatWithEmotion(text, userId = 'default') {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      text, 
      user_id: userId,
      top_k: 5 
    }),
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `Chat error: ${response.status}`)
  }

  const data = await response.json()
  
  return {
    text: data.text,
    emotions: data.emotions,
    usedFallback: data.used_fallback,
    botResponse: data.bot_response,
    affectState: data.affect_state,
    emotionalEvents: data.emotional_events || [],
    sessionInfo: data.session_info || {},
    topEmotion: data.emotions[0]?.label || 'neutral',
  }
}

/**
 * Get current affect state for a user without sending a message
 * @param {string} userId - User identifier
 */
export async function getAffectState(userId = 'default') {
  const response = await fetch(`/api/session/${userId}/affect`)
  
  if (!response.ok) {
    throw new Error(`Failed to get affect state: ${response.status}`)
  }
  
  return await response.json()
}

/**
 * Get emotional event history for a user
 * @param {string} userId - User identifier
 * @param {number} limit - Number of events to retrieve
 */
export async function getEmotionalEvents(userId = 'default', limit = 50) {
  const response = await fetch(`/api/session/${userId}/events?limit=${limit}`)
  
  if (!response.ok) {
    throw new Error(`Failed to get events: ${response.status}`)
  }
  
  return await response.json()
}
