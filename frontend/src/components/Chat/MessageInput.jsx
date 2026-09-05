import { useState, useRef, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { addMessage, createChat } from '../../store/slices/chatSlice'
import { setCurrentEmotion } from '../../store/slices/emotionSlice'
import { chatWithEmotion } from '../../utils/api'
import Icon from '../common/Icon'

// ---------------------------------------------------------------------------
// Bot responses — one per GoEmotions label.
// The backend is the sole decider of which label applies.
// The frontend just maps label → response text, nothing more.
// ---------------------------------------------------------------------------
const LABEL_RESPONSES = {
  admiration:
    "It's wonderful when someone or something genuinely moves you like that. That kind of admiration says a lot about what you value. What was it about them that stood out so much?",

  amusement:
    "Ha! I can just picture it — those little moments of unexpected absurdity are the best. It's great that you can find the humour in it. What happened?",

  approval:
    "It sounds like something really resonated with you, and that feeling of alignment is worth paying attention to. What is it that clicked?",

  caring:
    "The depth of care you have for this comes through clearly, and that kind of investment matters. It takes a lot to show up for people or things the way you do. What's on your mind?",

  curiosity:
    "That kind of genuine curiosity is infectious — it sounds like your mind just lit up. There's something really energising about going down a rabbit hole on something that fascinates you. What is it you want to explore first?",

  desire:
    "It's clear there's something you really want, and that longing is worth taking seriously. Sometimes just naming what we want is the first step. What would it mean to you if you got it?",

  excitement:
    "That excitement is so palpable — it really comes through! Big changes or new beginnings carry that electric mix of anticipation and possibility. Tell me everything — what's happening?",

  gratitude:
    "That kind of gratitude is a beautiful thing to hold onto. It sounds like someone or something made a real difference for you. What are you feeling most thankful for right now?",

  joy:
    "That's genuinely wonderful to hear — your joy really comes through, and it's contagious. It sounds like something really good is happening in your life. Tell me more about what's making you feel this way!",

  love:
    "That warmth and love you're feeling is something really special. Whether it's for a person, a place, or something you're passionate about, it clearly means a great deal to you. Tell me more about it.",

  optimism:
    "That sense of hope and optimism is a powerful thing — it sounds like you're genuinely looking forward to what's ahead. What's making you feel so positive about things right now?",

  pride:
    "You absolutely should feel proud — what you've done clearly took real effort and courage. That feeling of accomplishment is hard-earned and completely deserved. What did it take to get here?",

  relief:
    "Oh, that sense of relief when a weight finally lifts — there's nothing quite like it. It sounds like you've been carrying something heavy for a while. What happened, and how are you feeling now that it's over?",

  anger:
    "I can hear how angry you are right now, and honestly — given what you've described, that reaction is completely understandable. Being treated that way would frustrate anyone. Do you want to walk me through exactly what happened so I can understand the full picture?",

  annoyance:
    "Ugh, that sounds genuinely frustrating — sometimes it's the persistent, grinding little things that wear you down more than anything else. You've clearly been patient with this for long enough. What's been the most irritating part of it all?",

  disapproval:
    "That sounds like it crossed a real line, and your frustration is completely justified. It's hard when you feel like basic expectations of fairness or decency just aren't being met. Can you tell me more about what happened? I want to understand what you're dealing with.",

  disappointment:
    "Disappointment has this quiet, heavy sting to it — especially when you were genuinely hoping for something different. I'm sorry it didn't go the way you needed it to. What were you hoping would happen, and how far off was it?",

  disgust:
    "It sounds like something genuinely disturbed or offended you, and that reaction is telling — our gut feelings about things like this usually matter. What happened, and what bothered you most about it?",

  embarrassment:
    "Oh no — those moments where you just want the ground to swallow you whole are the worst, even when they're completely harmless. The fact that you're still cringing about it just shows how self-aware you are, honestly. What happened? Sometimes talking about it helps take the sting out.",

  fear:
    "That sounds genuinely frightening, and I want you to know it's okay to feel scared about this. Fear is your mind trying to protect you, even when it's overwhelming. You're safe here — can you tell me more about what's worrying you so we can look at it together?",

  grief:
    "Grief is one of the heaviest, most disorienting things a person can go through, and there's no right way to do it. I'm so deeply sorry for what you're experiencing right now. Please know you don't have to carry this alone — I'm here, and I'm listening. Would you like to share what happened?",

  nervousness:
    "That nervous, knotted feeling is so real — especially when something genuinely matters to you and the stakes feel high. The fact that you're anxious about it probably means you care deeply, which is actually a strength. What's the part that's weighing on you most right now?",

  remorse:
    "It sounds like you're being really hard on yourself, and I can feel the weight of that regret in what you're saying. Whatever happened, the fact that it's sitting with you like this shows how much you care about doing right. Do you want to talk through what happened? Sometimes saying it out loud helps.",

  sadness:
    "I'm really sorry you're feeling this way — that kind of sadness can feel really isolating, especially when it's hard to explain to others. You don't have to rush past it or put a brave face on it here. I'm listening, and I care. Would you like to share what's been going on?",

  confusion:
    "It sounds like something isn't quite clear right now — whether that's a situation that doesn't make sense, a question you're trying to get answered, or just a feeling of not knowing where to turn. I'm here to help you think it through. What is it that's unclear or confusing?",

  neutral:
    "I'm here and I'm listening. Sometimes it helps just to have somewhere to put your thoughts, even when you're not sure what you're feeling. What's been on your mind lately?",

  realization:
    "Those moments when something suddenly clicks into place can be really powerful — they can shift everything. It sounds like something just landed for you. What did you realise, and how does it feel now that you see it?",

  surprise:
    "That sounds like it came completely out of nowhere! Surprises — whether good or unsettling — can really knock you off balance for a moment. How are you sitting with it now that the dust is settling?",
}

// The backend's top label decides the response.
// If the label isn't in our map (future model update), fall back to neutral.
function getBotResponse(topLabel) {
  return LABEL_RESPONSES[topLabel] ?? LABEL_RESPONSES.neutral
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function MessageInput({ text, setText }) {
  const dispatch = useDispatch()
  const { activeChatId } = useSelector(state => state.chat)
  const [isTyping, setIsTyping]       = useState(false)
  const [backendStatus, setBackendStatus] = useState('unknown') // 'unknown'|'online'|'offline'
  const textareaRef = useRef(null)
  const isMountedRef = useRef(true)

  useEffect(() => {
    return () => {
      isMountedRef.current = false
    }
  }, [])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }, [text])

  const handleSend = async () => {
    if (!text.trim() || isTyping) return

    const userText = text.trim()

    if (!activeChatId) {
      dispatch(createChat())
      return
    }

    setText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    // Capture before any awaits — prevents reply landing in the wrong chat
    const targetChatId = activeChatId
    setIsTyping(true)

    // ── NEW: Call the enhanced /chat endpoint with full affect tracking ──
    let emotions = [{ label: 'neutral', score: 1.0 }]
    let topLabel = 'neutral'
    let confidence = 1.0
    let affectState = null
    let emotionalEvents = []
    let sessionInfo = null
    let botResponse = "I'm here and I'm listening."

    try {
      // Use new chatWithEmotion API that returns full affect state
      const result = await chatWithEmotion(userText, 'default')
      
      emotions = result.emotions
      topLabel = result.topEmotion
      confidence = result.emotions[0]?.score ?? 0.5
      affectState = result.affectState
      emotionalEvents = result.emotionalEvents
      sessionInfo = result.sessionInfo
      botResponse = result.botResponse
      
      setBackendStatus('online')
    } catch (err) {
      // Backend unreachable — show offline indicator, keep neutral placeholder
      console.warn('[EmotionChat] Backend unreachable:', err.message)
      setBackendStatus('offline')
      botResponse = "I'm having trouble connecting right now, but I'm here."
    }

    // Update emotion store with full affect state
    dispatch(setCurrentEmotion({ 
      emotion: topLabel, 
      confidence, 
      emotions,
      affectState,
      emotionalEvents,
      sessionInfo
    }))

    // 1. User message
    dispatch(addMessage({
      chatId: targetChatId,
      message: {
        role: 'user',
        text: userText,
        emotion: topLabel,
        emotions,
      },
    }))

    // 2. Prediction card — rendered inline between user message and bot reply
    dispatch(addMessage({
      chatId: targetChatId,
      message: {
        role: 'prediction',
        text: '',
        emotion: topLabel,
        emotions,
      },
    }))

    await new Promise(r => setTimeout(r, 600))
    if (!isMountedRef.current) return
    setIsTyping(false)

    // 3. Bot response (now using affect-aware response from backend)
    if (!isMountedRef.current) return
    dispatch(addMessage({
      chatId: targetChatId,
      message: {
        role: 'bot',
        text: botResponse,  // Use response from backend
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 1.0 }],
      },
    }))
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="p-3 border-t border-[var(--border-color)] bg-[var(--bg-primary)]">
      {backendStatus !== 'unknown' && (
        <div className="max-w-3xl mx-auto mb-2 flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
          <span className={`w-1.5 h-1.5 rounded-full ${backendStatus === 'online' ? 'bg-green-400' : 'bg-yellow-400'}`} />
          {backendStatus === 'online' ? 'AI emotion model active' : 'Offline — backend unreachable'}
        </div>
      )}

      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="How are you feeling today?"
          rows={1}
          className="flex-1 px-4 py-3 rounded-xl border border-[var(--border-color)] bg-[var(--bg-secondary)] text-[var(--text-primary)] text-sm resize-none outline-none focus:border-[var(--text-tertiary)] transition-colors min-h-[44px] max-h-[120px] leading-relaxed"
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || isTyping}
          className="w-10 h-10 rounded-xl bg-[var(--text-primary)] text-[var(--bg-primary)] flex items-center justify-center flex-shrink-0 hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
        >
          {isTyping ? (
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <Icon name="send" size={16} />
          )}
        </button>
      </div>

      {isTyping && (
        <div className="max-w-3xl mx-auto mt-2 flex items-center gap-2 text-xs text-[var(--text-muted)]">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)] animate-pulse" />
          EmotionChat is thinking...
        </div>
      )}
    </div>
  )
}
