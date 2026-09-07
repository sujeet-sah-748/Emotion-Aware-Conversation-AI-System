# Emotion Chatbot — Frontend

React-based chat interface for the emotion-aware chatbot. Displays real-time affect state, manages multiple chat sessions, and communicates with the FastAPI backend.

---

## Features

- **Emotion-aware chat UI** — message bubbles with emotion context from the backend
- **3-tier affect visualization** — real-time panel showing Situational, Short-term, and Long-term VAD states as visual bars
- **Prediction panel** — displays raw emotion classification scores per message
- **Multi-session management** — sidebar for creating, switching, and managing parallel chat sessions
- **Auth flow** — login and registration pages with simulated local auth state
- **Settings panel** — theme and preference controls applied live to the DOM
- **Profile panel** — user profile view
- **Responsive layout** — collapsible mobile sidebar with overlay
- **Error boundary** — graceful top-level error handling with fallback UI

---

## Tech Stack

| Tool | Version |
|---|---|
| React | 18.3 |
| Vite | 5.4 |
| Redux Toolkit | 2.2 |
| React Router | v6 |
| Tailwind CSS | 3.4 |
| Lucide React | 0.436 |
| date-fns | 3.6 |

---

## Prerequisites

- Node.js 18 or later
- npm 9 or later
- The backend running at the URL configured in `.env`

---

## Setup and Installation

```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `VITE_API_URL` | Base URL of the FastAPI backend | `http://localhost:8000` |

All Vite environment variables must be prefixed with `VITE_` to be exposed to the browser bundle.

---

## Running in Development

```bash
npm run dev
```

The app starts at `http://localhost:5173` with hot module replacement enabled.

---

## Project Structure

```
src/
├── components/
│   ├── Auth/
│   │   ├── LoginForm.jsx         # Login form with validation
│   │   └── RegisterForm.jsx      # Registration form
│   ├── Chat/
│   │   ├── ChatContainer.jsx     # Main chat view, message list, scroll management
│   │   ├── MessageBubble.jsx     # Individual message with emotion metadata
│   │   ├── MessageInput.jsx      # Text input with send handling
│   │   ├── AffectVisualization.jsx  # 3-tier VAD visualization panel
│   │   ├── PredictionPanel.jsx   # Raw emotion scores per message
│   │   └── EmptyState.jsx        # Placeholder shown before first message
│   ├── Layout/
│   │   ├── Header.jsx            # Top bar with menu toggle and navigation
│   │   └── Sidebar.jsx           # Session list, navigation, new chat button
│   ├── Profile/
│   │   └── ProfilePanel.jsx      # User profile display and edit
│   ├── Settings/
│   │   └── SettingsPanel.jsx     # Theme selector and preferences
│   └── common/
│       ├── ErrorBoundary.jsx     # React error boundary wrapper
│       └── Icon.jsx              # Lucide icon helper component
├── hooks/
│   └── useAuth.js                # Auth state selector hook
├── store/
│   └── slices/
│       ├── authSlice.js          # Auth state (isAuthenticated, user)
│       ├── settingsSlice.js      # Theme, preferences, DOM application
│       └── chatSlice.js          # Chat sessions, messages, affect state
├── App.jsx                       # Root component, routing, layout shell
├── main.jsx                      # React DOM entry point, Redux Provider
└── index.css                     # Tailwind base + CSS custom properties (theme vars)
```

---

## Component Descriptions

### Auth
- **LoginForm** — email/password form that dispatches to `authSlice`. Simulated auth (no real backend auth endpoint required).
- **RegisterForm** — registration form, mirrors login flow.

### Chat
- **ChatContainer** — owns the active session's message list, calls the backend `/chat` endpoint, and manages scroll-to-bottom behavior.
- **MessageBubble** — renders a single user or assistant message. Optionally shows emotion label and confidence from the backend response.
- **MessageInput** — controlled textarea with keyboard shortcut (Enter to send, Shift+Enter for newline).
- **AffectVisualization** — reads the `affect_state` from the latest chat response and renders three sets of VAD bars (Situational, Short-term, Long-term).
- **PredictionPanel** — displays the ranked emotion scores list (`emotions` array) returned by the backend.
- **EmptyState** — decorative placeholder with suggested prompts shown when no messages exist yet.

### Layout
- **Header** — top navigation bar; shows hamburger on mobile, navigation actions (settings, profile, logout).
- **Sidebar** — lists all chat sessions from the Redux store, highlights the active one, and provides a "New chat" button.

### Common
- **ErrorBoundary** — class component that catches render errors and displays a fallback UI instead of a blank screen.
- **Icon** — thin wrapper around `lucide-react` that maps string names to icon components.

---

## Redux Store Slices

### `authSlice`
Manages authentication state.

| State key | Type | Description |
|---|---|---|
| `isAuthenticated` | `boolean` | Whether a user is logged in |
| `user` | `object \| null` | Current user data |

Actions: `login`, `logout`, `register`

### `settingsSlice`
Manages UI preferences and theme.

| State key | Type | Description |
|---|---|---|
| `theme` | `string` | Active theme name (e.g. `"dark"`, `"light"`) |
| `...preferences` | `any` | Additional preference keys |

Actions: `loadSettings`, `updateSettings`  
Helpers: `applyThemeToDom(settings)` — writes CSS custom properties to `:root`

### `chatSlice`
Manages chat sessions and messages.

| State key | Type | Description |
|---|---|---|
| `sessions` | `array` | All chat sessions with metadata |
| `activeSessionId` | `string \| null` | Currently selected session |
| `messages` | `object` | Messages keyed by session ID |

Actions: `createSession`, `setActiveSession`, `addMessage`, `deleteSession`
