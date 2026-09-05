import { createSlice } from '@reduxjs/toolkit'

const generateId = () => Date.now().toString(36) + Math.random().toString(36).substring(2)

// Each message now carries a full `emotions` array from the backend:
// [{ label: string, score: number }, ...]  sorted descending by score
// `emotion` (string) is kept as the top label for backwards-compat with
// any code that still reads msg.emotion directly (e.g. old localStorage data)
const demoChats = [
  {
    id: 'chat-1',
    title: 'Work stress and deadlines',
    createdAt: new Date(Date.now() - 7200000).toISOString(),
    updatedAt: new Date(Date.now() - 7200000).toISOString(),
    messages: [
      {
        id: 'm1', role: 'user',
        text: "I'm stressed about work",
        emotion: 'nervousness',
        emotions: [{ label: 'nervousness', score: 0.71 }, { label: 'disappointment', score: 0.34 }, { label: 'sadness', score: 0.21 }],
        timestamp: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: 'm2', role: 'bot',
        text: "Oh, it sounds like work is really getting to you right now, and that kind of stress can be incredibly heavy. It's completely understandable to feel that way. Would it help to talk about what's making you stressed, or would you prefer we just find a way to relax your mind for a bit?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.88 }],
        timestamp: new Date(Date.now() - 7195000).toISOString(),
      },
      {
        id: 'm3', role: 'user',
        text: "I just feel overwhelmed by deadlines",
        emotion: 'nervousness',
        emotions: [{ label: 'nervousness', score: 0.68 }, { label: 'fear', score: 0.42 }, { label: 'sadness', score: 0.29 }],
        timestamp: new Date(Date.now() - 7190000).toISOString(),
      },
      {
        id: 'm4', role: 'bot',
        text: "I hear you. Deadlines can feel like walls closing in. Let's break them down together — which one feels most urgent?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.82 }],
        timestamp: new Date(Date.now() - 7185000).toISOString(),
      },
    ],
  },
  {
    id: 'chat-2',
    title: 'Movie disappointment',
    createdAt: new Date(Date.now() - 86400000).toISOString(),
    updatedAt: new Date(Date.now() - 86400000).toISOString(),
    messages: [
      {
        id: 'm5', role: 'user',
        text: "I was really disappointed. The plot was boring and the characters felt shallow and uninteresting.",
        emotion: 'disappointment',
        emotions: [{ label: 'disappointment', score: 0.78 }, { label: 'disapproval', score: 0.55 }, { label: 'annoyance', score: 0.31 }],
        timestamp: new Date(Date.now() - 86400000).toISOString(),
      },
      {
        id: 'm6', role: 'bot',
        text: "Oh, that's truly disappointing. I can hear how much that let you down. It's so frustrating when a story you were hoping to enjoy just doesn't deliver. Would you like to tell me more about what you wished had been different?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.85 }],
        timestamp: new Date(Date.now() - 86395000).toISOString(),
      },
    ],
  },
  {
    id: 'chat-3',
    title: 'Promotion excitement!',
    createdAt: new Date(Date.now() - 172800000).toISOString(),
    updatedAt: new Date(Date.now() - 172800000).toISOString(),
    messages: [
      {
        id: 'm7', role: 'user',
        text: "I just got promoted at work! I'm so excited!",
        emotion: 'excitement',
        emotions: [{ label: 'excitement', score: 0.89 }, { label: 'joy', score: 0.62 }, { label: 'pride', score: 0.45 }],
        timestamp: new Date(Date.now() - 172800000).toISOString(),
      },
      {
        id: 'm8', role: 'bot',
        text: "That's wonderful news! Congratulations on your promotion! Your hard work is clearly paying off. How are you feeling about the new responsibilities?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.80 }],
        timestamp: new Date(Date.now() - 172795000).toISOString(),
      },
    ],
  },
  {
    id: 'chat-4',
    title: 'Exam anxiety',
    createdAt: new Date(Date.now() - 259200000).toISOString(),
    updatedAt: new Date(Date.now() - 259200000).toISOString(),
    messages: [
      {
        id: 'm9', role: 'user',
        text: "I feel very stressed about exams.",
        emotion: 'nervousness',
        emotions: [{ label: 'nervousness', score: 0.74 }, { label: 'fear', score: 0.38 }, { label: 'sadness', score: 0.22 }],
        timestamp: new Date(Date.now() - 259200000).toISOString(),
      },
      {
        id: 'm10', role: 'bot',
        text: "I remember you mentioned exams earlier. Want a quick study plan?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.86 }],
        timestamp: new Date(Date.now() - 259195000).toISOString(),
      },
    ],
  },
  {
    id: 'chat-5',
    title: 'Feeling lonely lately',
    createdAt: new Date(Date.now() - 345600000).toISOString(),
    updatedAt: new Date(Date.now() - 345600000).toISOString(),
    messages: [
      {
        id: 'm11', role: 'user',
        text: "I've been feeling quite lonely recently.",
        emotion: 'sadness',
        emotions: [{ label: 'sadness', score: 0.69 }, { label: 'grief', score: 0.31 }, { label: 'neutral', score: 0.18 }],
        timestamp: new Date(Date.now() - 345600000).toISOString(),
      },
      {
        id: 'm12', role: 'bot',
        text: "I'm sorry you're feeling that way. Loneliness can be really difficult. Would you like to talk about what's been going on, or maybe explore some ways to connect with others?",
        emotion: 'neutral',
        emotions: [{ label: 'neutral', score: 0.83 }],
        timestamp: new Date(Date.now() - 345595000).toISOString(),
      },
    ],
  },
]

function loadChatsFromStorage() {
  try {
    const stored = localStorage.getItem('emotionchat_chats')
    return stored ? JSON.parse(stored) : null
  } catch {
    return null
  }
}

const initialState = {
  chats: loadChatsFromStorage() || demoChats,
  activeChatId: null,
  isLoading: false,
  streamingMessageId: null,
}

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setActiveChat: (state, action) => {
      state.activeChatId = action.payload
    },
    createChat: (state, action) => {
      const newChat = {
        id: generateId(),
        title: action.payload?.title || 'New conversation',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: [],
      }
      state.chats.unshift(newChat)
      state.activeChatId = newChat.id
      localStorage.setItem('emotionchat_chats', JSON.stringify(state.chats))
    },
    addMessage: (state, action) => {
      const { chatId, message } = action.payload
      const chat = state.chats.find(c => c.id === chatId)
      if (chat) {
        chat.messages.push({
          ...message,
          // These always override whatever the caller passed
          id: generateId(),
          emotions: message.emotions || (message.emotion ? [{ label: message.emotion, score: 1.0 }] : [{ label: 'neutral', score: 1.0 }]),
          emotion: message.emotion || message.emotions?.[0]?.label || 'neutral',
          timestamp: new Date().toISOString(),
        })
        chat.updatedAt = new Date().toISOString()
        localStorage.setItem('emotionchat_chats', JSON.stringify(state.chats))
      }
    },
    updateMessage: (state, action) => {
      const { chatId, messageId, updates } = action.payload
      const chat = state.chats.find(c => c.id === chatId)
      if (chat) {
        const msg = chat.messages.find(m => m.id === messageId)
        if (msg) Object.assign(msg, updates)
        localStorage.setItem('emotionchat_chats', JSON.stringify(state.chats))
      }
    },
    renameChat: (state, action) => {
      const { chatId, title } = action.payload
      const chat = state.chats.find(c => c.id === chatId)
      if (chat) {
        chat.title = title
        localStorage.setItem('emotionchat_chats', JSON.stringify(state.chats))
      }
    },
    deleteChat: (state, action) => {
      state.chats = state.chats.filter(c => c.id !== action.payload)
      if (state.activeChatId === action.payload) {
        state.activeChatId = state.chats[0]?.id || null
      }
      localStorage.setItem('emotionchat_chats', JSON.stringify(state.chats))
    },
    clearAllChats: (state) => {
      state.chats = []
      state.activeChatId = null
      localStorage.removeItem('emotionchat_chats')
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload
    },
    setStreamingMessageId: (state, action) => {
      state.streamingMessageId = action.payload
    },
  },
})

export const {
  setActiveChat,
  createChat,
  addMessage,
  updateMessage,
  renameChat,
  deleteChat,
  clearAllChats,
  setLoading,
  setStreamingMessageId,
} = chatSlice.actions
export default chatSlice.reducer
