# An Emotion-Aware Conversational AI system with Tired and Multi-Label Affective Memory

An emotion-aware conversational architecture that integrates fine-grained emotion recognition with hierarchical affect-state tracking, contextual memory, adversarial emotion handling, and emotion-conditioned response generation.

---

## Architecture Overview

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│        Frontend             │  HTTP   │             Backend                  │
│   React 18 + Vite 5         │ ◄─────► │   FastAPI + Uvicorn (port 8000)      │
│   Redux Toolkit             │  REST   │   PyTorch LoRA Emotion Classifier    │
│   Tailwind CSS              │         │   Ollama LLM (phi4-mini, local)      │
│   Affect Visualization      │         │   Mem0 + Qdrant Memory System        │
│   Multi-session Chat UI     │         │   Redis Cache                        │
└─────────────────────────────┘         │   3-Tier Emotion Engine              │
                                        └──────────────────────────────────────┘
                                                        │
                                          ┌─────────────┼─────────────┐
                                          │             │             │
                                       Ollama        Qdrant        Redis
                                    (local LLM)  (vector store)  (cache)
```

The frontend sends chat messages to the backend REST API. The backend classifies emotions using a fine-tuned LoRA adapter, updates a 3-tier affect state, retrieves relevant memories from Qdrant via Mem0, builds an emotion-aware prompt, and calls Ollama locally to generate a response. All results — including the full affect state — are returned to the frontend for visualization.

---

## Key Features

- **28-label multi-emotion classification** using a fine-tuned LoRA adapter (PEFT) on top of a HuggingFace transformer
- **3-tier emotion tracking** — Situational (immediate), Short-term (decaying), and Long-term (persistent) affect states with VAD (Valence-Arousal-Dominance) representation
- **Intelligent memory** via Mem0 with Qdrant vector store — semantic search over past conversations and emotional events
- **Local LLM inference** with Ollama (phi4-mini) — no API costs, runs entirely on your machine
- **Redis caching** for emotion predictions, sessions, context, and memory results
- **Graceful fallback chain** — every optional component (Ollama, Redis, Qdrant) has a working fallback so the system never hard-crashes
- **Emotion-aware chat UI** with real-time affect visualization panels
- **Multi-session management** with per-user session isolation
- **Auth flow** (login/register) with theme and settings support

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend framework | React 18, Vite 5 |
| State management | Redux Toolkit |
| Routing | React Router v6 |
| Styling | Tailwind CSS |
| Backend framework | FastAPI, Uvicorn |
| ML inference | PyTorch, HuggingFace Transformers, PEFT (LoRA) |
| LLM | Ollama (phi4-mini, local) |
| Memory system | Mem0, Qdrant |
| Cache | Redis |
| Legacy vector store | ChromaDB |
| Embeddings | nomic-embed-text (via Ollama) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) installed and running
- Redis (optional but recommended)
- Qdrant (optional, used by Mem0)

### 1. Clone the repository

```bash
git clone <repository-url>
cd emotion-chatbot-frontend
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration (Ollama URL, Redis, Qdrant, etc.)
```

Pull the required Ollama models:

```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

Start the backend:

```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Frontend setup

```bash
cd ../frontend
npm install
cp .env.example .env   # or create .env manually
# Set VITE_API_URL=http://localhost:8000
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## Project Structure

```
emotion-chatbot-frontend/
├── backend/
│   ├── core/                   # Emotion engine, context manager
│   ├── memory/                 # Redis cache, Mem0 integration
│   ├── response/               # Ollama LLM, response generator
│   ├── final_adapter/          # LoRA adapter weights
│   ├── chroma_db/              # ChromaDB persistence (legacy fallback)
│   ├── main.py                 # FastAPI app and all endpoints
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── store/              # Redux slices
│   │   └── App.jsx
│   ├── package.json
│   └── .env
└── README.md
```

