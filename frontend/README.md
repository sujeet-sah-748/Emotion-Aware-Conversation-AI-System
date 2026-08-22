# EmotionChat Frontend

Emotion-aware chatbot frontend built with React 18, Redux Toolkit, Tailwind CSS, and Vite.

## Features

- **Authentication** — Login / Register with persistent sessions
- **Chat History** — Sidebar with searchable conversation list, rename & delete
- **Emotion Detection** — Real-time emotion badges on messages and header
- **Hover Actions** — Copy, like, dislike, regenerate on bot messages
- **Multi-Colour Theme** — 8 accent colors with live switching
- **Dark Mode** — Full dark/light theme toggle
- **Settings Panel** — Appearance, notifications, privacy toggles
- **Profile Panel** — Stats, emotion legend, account management, logout
- **Responsive** — Mobile-friendly with collapsible sidebar

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 |
| State | Redux Toolkit |
| Routing | React Router DOM |
| Styling | Tailwind CSS |
| Icons | Lucide React |
| Build | Vite |

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

The dev server runs on `http://localhost:5173` with API proxy to `http://localhost:8000`.

## Project Structure

```
src/
├── components/
│   ├── Auth/           # LoginForm, RegisterForm
│   ├── Chat/           # ChatContainer, MessageBubble, MessageInput, EmptyState
│   ├── Layout/         # Sidebar, Header
│   ├── Profile/        # ProfilePanel
│   ├── Settings/       # SettingsPanel
│   └── common/         # Icon wrapper
├── hooks/              # useAuth
├── store/              # Redux store + slices
├── utils/              # emotionColors, formatters
├── App.jsx
├── main.jsx
└── index.css
```

## Emotion Colors

| Emotion | Color |
|---------|-------|
| Joy | `#4ade80` |
| Sadness | `#f87171` |
| Anger | `#fbbf24` |
| Fear | `#a78bfa` |
| Surprise | `#60a5fa` |
| Stress | `#fb923c` |
| Anxiety | `#c084fc` |
| Loneliness | `#94a3b8` |
| Gratitude | `#34d399` |
| Neutral | `#9ca3af` |
