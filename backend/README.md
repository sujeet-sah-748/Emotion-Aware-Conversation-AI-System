# Emotion Chatbot — Backend

FastAPI backend for the emotion-aware chatbot. Classifies user messages across 28 emotion labels, tracks affect state across three temporal tiers, retrieves long-term memories from a vector store, and generates empathetic responses through a local Ollama LLM.

---

## Features

- **28-label multi-emotion classification** using a fine-tuned LoRA adapter (PEFT) loaded on top of a HuggingFace transformer
- **VAD representation** — each emotion is mapped to a continuous Valence-Arousal-Dominance vector for nuanced affect tracking
- **3-tier emotion memory**
  - *Situational* — immediate, per-message affect
  - *Short-term* — decaying weighted average across recent turns
  - *Long-term* — slowly evolving baseline with wall-clock half-life decay
- **Emotional event logging** — spikes, shifts, reinforcements, and promotions are captured and returned with each response
- **Mem0 intelligent memory** with Qdrant vector store — semantic search over past conversations and emotional events; automatic memory extraction
- **Ollama LLM (phi4-mini)** for local, zero-cost response generation with emotion-aware prompt injection
- **Redis caching** for emotion predictions (1 h TTL), sessions (24 h TTL), context (5 min TTL), and memory (10 min TTL)
- **Per-user session isolation** with in-memory state and Redis persistence
- **Token-budgeted context assembly** — retrieves and packs memories within a configurable token window
- **Graceful fallback chain** — every optional component degrades cleanly; see [FALLBACK_MECHANISMS.md](../FALLBACK_MECHANISMS.md)
- **Model warm-up on startup** — LoRA classifier and sentence transformer embedder load before the first request

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI, Uvicorn |
| ML runtime | PyTorch |
| Emotion model | HuggingFace Transformers + PEFT (LoRA adapter) |
| LLM | Ollama (phi4-mini, local inference) |
| Memory system | Mem0, Qdrant |
| Embeddings | nomic-embed-text via Ollama |
| Cache | Redis + hiredis |
| Legacy vector store | ChromaDB |
| Schema validation | Pydantic v2 |

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running on `http://localhost:11434`
  - Pull required models: `ollama pull phi4-mini && ollama pull nomic-embed-text`
- **Redis** (optional) — for caching; system works without it
- **Qdrant** (optional) — for Mem0 vector store; falls back to ChromaDB, then in-memory

---

## Setup and Installation

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

### Ollama (LLM)

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_MODEL` | Model name to use for chat generation | `phi4-mini` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_TEMPERATURE` | Sampling temperature | `0.7` |
| `OLLAMA_MAX_TOKENS` | Max tokens per response | `1024` |

### Redis Cache

| Variable | Description | Default |
|---|---|---|
| `REDIS_ENABLED` | Enable Redis caching | `true` |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database index | `0` |
| `REDIS_PASSWORD` | Redis password (leave empty if none) | _(empty)_ |

### Mem0 Memory System

| Variable | Description | Default |
|---|---|---|
| `MEM0_VECTOR_STORE` | Vector store backend (`qdrant` or `chroma`) | `qdrant` |
| `QDRANT_URL` | Qdrant server URL | _(empty)_ |
| `QDRANT_API_KEY` | Qdrant API key (cloud) | _(empty)_ |
| `MEM0_EMBEDDER_PROVIDER` | Embedding provider (`ollama` or `openai`) | `ollama` |
| `MEM0_EMBEDDER_MODEL` | Embedding model name | `nomic-embed-text` |
| `MEM0_LLM_PROVIDER` | LLM provider for Mem0 (`ollama`) | `ollama` |
| `MEM0_LLM_MODEL` | LLM model for Mem0 memory extraction | `phi4-mini` |
| `MEM0_COLLECTION` | Qdrant/Chroma collection name | `emotion_chatbot_memories` |
| `MEM0_HISTORY_DB` | SQLite file for Mem0 history | `./mem0_history.db` |
| `MEM0_ENABLE_GRAPH` | Enable graph memory (requires Neo4j) | `false` |

### OpenAI (optional)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key — only needed if using OpenAI as Mem0 embedder |

### ChromaDB (legacy fallback vector store)

| Variable | Description | Default |
|---|---|---|
| `USE_CHROMADB` | Enable ChromaDB as fallback | `true` |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory | `./chroma_db` |
| `CHROMA_COLLECTION` | ChromaDB collection name | `emotion_memories` |

### Context Window

| Variable | Description | Default |
|---|---|---|
| `MAX_CONTEXT_TOKENS` | Total token budget for LLM context | `7168` |
| `RESERVED_TOKENS` | Tokens reserved for the response | `1024` |
| `USE_TIKTOKEN` | Use tiktoken for accurate token counting | `false` |

### Memory Retrieval

| Variable | Description | Default |
|---|---|---|
| `TOP_K_MEMORIES` | Number of memories to retrieve per turn | `5` |
| `MEMORY_RECENCY_WEIGHT` | Weight for recency scoring | `0.2` |
| `MEMORY_SIMILARITY_WEIGHT` | Weight for semantic similarity scoring | `0.7` |
| `MEMORY_IMPORTANCE_WEIGHT` | Weight for importance scoring | `0.1` |

---

## Running the Server

```bash
# Option 1 — run directly
python main.py

# Option 2 — via uvicorn (with auto-reload for development)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts on `http://localhost:8000`.  
Interactive API docs (Swagger UI): `http://localhost:8000/docs`

---

## API Endpoints

### General

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/docs` | Swagger UI (interactive API documentation) |

### Emotion & Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Raw emotion classification — returns scored label list from the LoRA model |
| `POST` | `/chat` | Main chat endpoint — full emotion tracking, memory retrieval, LLM response |
| `POST` | `/chat/title` | Generate a short title for a chat session from its first message |

#### `POST /chat` request body

```json
{
  "text": "I'm feeling really overwhelmed today.",
  "user_id": "user-123",
  "threshold": 0.5,
  "top_k": null
}
```

#### `POST /chat` response shape (abbreviated)

```json
{
  "text": "...",
  "emotions": [...],
  "used_fallback": false,
  "bot_response": "...",
  "affect_state": {
    "situational_vad": { "valence": 0.3, "arousal": 0.7, "dominance": 0.4 },
    "short_term_vad": { ... },
    "long_term_vad": { ... },
    "stm_dominant": "anxiety",
    "ltm_dominant": "neutral",
    "trend": "declining",
    "confidence": 0.82
  },
  "emotional_events": [...],
  "session_info": { ... }
}
```

### Session Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/session/{user_id}/affect` | Current affect state without sending a message |
| `GET` | `/session/{user_id}/events` | Emotional event history for a session |
| `GET` | `/session/{user_id}/context_diagnostics` | Context manager metrics (token usage, memory count) |
| `DELETE` | `/session/{user_id}` | Delete a user session |
| `GET` | `/sessions/active` | List all active in-memory sessions |

### Cache Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/cache/stats` | Redis cache statistics and TTL info |
| `GET` | `/cache/health` | Redis connection health check |
| `DELETE` | `/cache/clear` | Clear all or a specific cache type (`emotion`, `session`, `context`, `memory`) |

### Memory (Mem0)

| Method | Path | Description |
|---|---|---|
| `GET` | `/memory/health` | Mem0 system health and config |
| `GET` | `/memory/{user_id}` | Get all memories for a user |
| `POST` | `/memory/{user_id}/search` | Semantic search over a user's memories |
| `GET` | `/memory/{user_id}/emotional` | Get emotional memories, with optional emotion and importance filters |
| `GET` | `/memory/{user_id}/summary` | Memory profile summary and stats |
| `POST` | `/memory/{user_id}/add` | Manually add a memory |
| `DELETE` | `/memory/{user_id}/{memory_id}` | Delete a specific memory |
| `DELETE` | `/memory/{user_id}` | Delete all memories for a user |

---

## Fallback Mechanisms

Every optional component has a working fallback so the server never hard-crashes due to a missing dependency:

| Component | Fallback |
|---|---|
| Ollama LLM unavailable | Template-based rule-driven responses via `compose_response()` |
| Ollama request fails | Generic supportive message |
| Mem0 / Qdrant unavailable | Context engine in-memory vector store |
| ChromaDB unavailable | `InMemoryVectorStore` (brute-force search) |
| Redis unavailable | All cache ops return miss silently; system runs without caching |
| Emotion confidence below threshold | Returns highest-scoring label with `used_fallback=True` |
| Emotion inference error | Returns neutral VAD signal (`confidence=0.0`); conversation continues |
| tiktoken unavailable | `HeuristicTokenCounter` (~3.5 chars/token) |

For the complete fallback chain diagram and implementation details, see [FALLBACK_MECHANISMS.md](../FALLBACK_MECHANISMS.md).

---

## Project Structure

```
backend/
├── core/
│   ├── emotion_engine.py             # 3-tier affect tracking, VAD model, event logging
│   ├── context_engine.py             # Memory retrieval, vector store, token budgeting
│   ├── context_manager_integration.py # ChatContextManager — ties emotion + memory together
│   └── __init__.py
├── memory/
│   ├── mem0_integration.py           # Mem0 client, memory search, emotional event storage
│   ├── redis_cache.py                # Redis cache with graceful fallback
│   └── __init__.py
├── response/
│   ├── ollama_llm.py                 # Ollama client, emotion-aware prompt builder
│   ├── response_generator.py         # LoRA classifier, template-based fallback responses
│   └── __init__.py
├── final_adapter/                    # Fine-tuned LoRA adapter weights (PEFT)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── chroma_db/                        # ChromaDB persistence (legacy fallback vector store)
├── main.py                           # FastAPI app, all endpoints, session management
├── requirements.txt
└── .env.example
```
