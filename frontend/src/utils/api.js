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
