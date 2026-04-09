# System Architecture - Emotion-Aware Chatbot

## Overview

The Emotion-Aware Chatbot is a full-stack application that provides empathetic, context-aware conversational AI. The system detects user emotions in real-time, maintains conversation memory across multiple storage layers, and generates responses that are both emotionally intelligent and contextually relevant.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  React + TypeScript + Vite + Tailwind CSS + shadcn/ui          │
│  (Browser-based SPA with real-time WebSocket connection)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/REST + WebSocket
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│              FastAPI + CORS + Authentication                     │
│                    (Python 3.11+)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼──────────┐ ┌─▼──────────────┐
│  EMOTION        │ │   MEMORY     │ │   RESPONSE     │
│  DETECTION      │ │   SERVICE    │ │   GENERATION   │
│  (RoBERTa)      │ │              │ │   (LangGraph)  │
└────────┬────────┘ └───┬──────────┘ └─┬──────────────┘
         │              │               │
         │      ┌───────┴───────┐       │
         │      │               │       │
         │  ┌───▼────┐  ┌──────▼───┐   │
         │  │ Vector │  │  Graph   │   │
         │  │   DB   │  │   DB     │   │
         │  │(Pinecone)│ │ (Neo4j) │   │
         │  └────────┘  └──────────┘   │
         │                              │
         └──────────────┬───────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼─────┐  ┌────────┐  ┌──────▼────┐
    │PostgreSQL│  │ Redis  │  │  OpenAI   │
    │   (DB)   │  │(Cache) │  │   (LLM)   │
    └──────────┘  └────────┘  └───────────┘
```

## Technology Stack

### Frontend
- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **Routing:** React Router v6
- **State Management:** Zustand (global state) + React Query (server state)
- **Styling:** Tailwind CSS + shadcn/ui components
- **Real-time:** Socket.io Client
- **HTTP Client:** Axios
- **Form Handling:** React Hook Form + Zod validation
- **Animations:** Framer Motion

### Backend
- **Framework:** FastAPI (Python 3.11+)
- **Workflow Engine:** LangGraph
- **LLM Integration:** LangChain
- **AI/ML:** RoBERTa (emotion detection)
- **Authentication:** JWT + bcrypt
- **ASGI Server:** Uvicorn

### Data Layer
- **Relational DB:** PostgreSQL 14+ (users, conversations, messages)
- **Vector DB:** Pinecone (semantic search, conversation embeddings)
- **Graph DB:** Neo4j 5+ (user facts, relationships, patterns)
- **Cache:** Redis 7+ (sessions, recent context, rate limiting)

### Infrastructure
- **Containerization:** Docker + Docker Compose
- **API Documentation:** OpenAPI/Swagger
- **Testing:** Pytest (backend), Vitest (frontend)
- **Migrations:** Alembic

## System Components

### 1. Frontend Architecture

```
client/
├── src/
│   ├── pages/              # Route-level components
│   │   ├── Home.tsx        # Landing page
│   │   ├── Login.tsx       # Authentication
│   │   ├── Register.tsx    # User registration
│   │   ├── Chat.tsx        # Main chat interface
│   │   └── Index.tsx       # Route configuration
│   │
│   ├── components/         # Reusable UI components
│   │   ├── chat/           # Chat-specific components
│   │   │   ├── ChatContainer.tsx    # Main chat layout
│   │   │   ├── ChatInput.tsx        # Message input
│   │   │   ├── ChatSidebar.tsx      # Conversation list
│   │   │   ├── MessageBubble.tsx    # Message display
│   │   │   ├── EmotionBadge.tsx     # Emotion indicator
│   │   │   └── TypingIndicator.tsx  # Loading state
│   │   ├── ui/             # shadcn/ui components
│   │   ├── NavLink.tsx     # Navigation
│   │   └── ThemeToggle.tsx # Dark/light mode
│   │
│   ├── store/              # State management
│   │   └── chatStore.ts    # Chat state (Zustand)
│   │
│   ├── lib/                # Utilities
│   │   ├── api.ts          # API client configuration
│   │   └── utils.ts        # Helper functions
│   │
│   ├── contexts/           # React contexts
│   │   └── ThemeContext.tsx
│   │
│   ├── hooks/              # Custom React hooks
│   │   ├── use-mobile.tsx
│   │   └── use-toast.ts
│   │
│   └── types/              # TypeScript definitions
│       └── chat.types.ts
```

### 2. Backend Architecture

```
server/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/      # API route handlers
│   │   │   ├── auth.py     # Authentication (register, login, refresh)
│   │   │   ├── chat.py     # Chat operations (send, history, list)
│   │   │   ├── emotion.py  # Emotion detection & analytics
│   │   │   ├── memory.py   # Memory management
│   │   │   └── health.py   # Health checks
│   │   │
│   │   ├── deps/           # Dependency injection
│   │   │   ├── auth_deps.py    # Auth dependencies
│   │   │   └── db_deps.py      # Database session
│   │   │
│   │   └── router.py       # API router configuration
│   │
│   ├── core/               # Core configuration
│   │   ├── config.py       # Settings (env vars)
│   │   ├── security.py     # JWT, password hashing
│   │   └── exceptions.py   # Custom exceptions
│   │
│   ├── models/             # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── conversation.py
│   │   ├── message.py
│   │   └── emotion.py
│   │
│   ├── schemas/            # Pydantic schemas (validation)
│   │   ├── user.py
│   │   ├── chat.py
│   │   ├── emotion.py
│   │   └── memory.py
│   │
│   ├── services/           # Business logic
│   │   ├── emotion/
│   │   │   ├── detector.py         # RoBERTa emotion detection
│   │   │   └── analyzer.py         # Emotion analytics
│   │   │
│   │   ├── memory/
│   │   │   ├── vector_store.py     # Pinecone client
│   │   │   ├── graph_store.py      # Neo4j client
│   │   │   ├── cache_manager.py    # Redis client
│   │   │   └── memory_service.py   # Unified memory interface
│   │   │
│   │   └── chat/
│   │       ├── langgraph_graph.py      # LangGraph workflow
│   │       ├── node_functions.py       # Workflow nodes
│   │       └── response_generator.py   # LLM response generation
│   │
│   ├── db/                 # Database utilities
│   │   ├── session.py      # DB session management
│   │   └── init_db.py      # Database initialization
│   │
│   ├── utils/              # Utilities
│   │   ├── text_processing.py
│   │   ├── embeddings.py
│   │   └── logging.py
│   │
│   └── main.py             # Application entry point
```

## Data Flow

### Complete Request-Response Cycle

```
1. USER INPUT
   User: "I feel very stressed about exams"
   ↓
   
2. FRONTEND PROCESSING
   - Capture input from ChatInput component
   - Update UI with user message
   - Send to backend via API/WebSocket
   ↓
   
3. API GATEWAY
   - Validate JWT token
   - Extract user_id
   - Route to chat endpoint
   ↓
   
4. EMOTION DETECTION (Parallel)
   - Preprocess text (lowercase, emoji handling)
   - RoBERTa model inference
   - Output: {primary: "stress", confidence: 0.92, scores: {...}}
   ↓
   
5. MEMORY RETRIEVAL (Parallel)
   ├─ Vector DB (Pinecone)
   │  └─ Semantic search for similar past conversations
   │     Result: "User discussed exams 2 weeks ago"
   │
   ├─ Graph DB (Neo4j)
   │  └─ Query user facts and relationships
   │     Result: "User is CS student, 3 past exam stress instances"
   │
   └─ Cache (Redis)
      └─ Fetch recent conversation context
         Result: "Last topic: study_schedule"
   ↓
   
6. CONTEXT ASSEMBLY
   - Combine all retrieved memories
   - Identify patterns (recurring exam anxiety)
   - Build user profile (prefers practical solutions)
   ↓
   
7. LANGGRAPH WORKFLOW
   State: {
     user_id, message, emotion, context, response, metadata
   }
   
   Nodes:
   ├─ Input Processing
   ├─ Emotion Detection (completed)
   ├─ Memory Retrieval (completed)
   ├─ Context Assembly (completed)
   ├─ Response Generation → LLM call with emotion-aware prompt
   ├─ Validation → Check response quality
   └─ Storage → Save to all databases
   ↓
   
8. RESPONSE GENERATION
   - Construct emotion-aware prompt
   - Include context and user profile
   - Call OpenAI/Anthropic LLM
   - Apply response template for "stress" emotion
   ↓
   
9. STORAGE (Parallel)
   ├─ PostgreSQL: Save message to conversations table
   ├─ Vector DB: Store message embedding
   ├─ Graph DB: Update relationships (User)-[:STRESSED_ABOUT]->(Exams)
   ├─ Redis: Update conversation cache
   └─ Emotion Log: Record emotion data
   ↓
   
10. API RESPONSE
    Return: {
      message: "I remember you mentioned exams...",
      emotion: {primary: "stress", confidence: 0.92},
      timestamp: "2026-04-07T10:30:00Z"
    }
    ↓
    
11. FRONTEND UPDATE
    - Receive response via WebSocket/HTTP
    - Update chatStore with new message
    - Render MessageBubble with EmotionBadge
    - Scroll to bottom
    - Clear input field
```

## Database Schema

### PostgreSQL Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    emotion_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Emotion logs table
CREATE TABLE emotion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    primary_emotion VARCHAR(50),
    secondary_emotion VARCHAR(50),
    confidence FLOAT,
    scores JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_emotion_logs_user_id ON emotion_logs(user_id);
CREATE INDEX idx_emotion_logs_timestamp ON emotion_logs(timestamp);
```

### Neo4j Graph Schema

```cypher
// User node
(:User {
  id: UUID,
  username: String,
  created_at: DateTime
})

// Topic node
(:Topic {
  name: String,
  category: String
})

// Emotion node
(:Emotion {
  type: String,
  intensity: Float
})

// Fact node
(:Fact {
  content: String,
  confidence: Float,
  created_at: DateTime
})

// Relationships
(:User)-[:DISCUSSED {timestamp: DateTime, count: Int}]->(:Topic)
(:User)-[:FELT {timestamp: DateTime, intensity: Float}]->(:Emotion)
(:User)-[:HAS_FACT]->(:Fact)
(:Topic)-[:TRIGGERS]->(:Emotion)
(:Topic)-[:RELATED_TO]->(:Topic)
```

### Vector Database (Pinecone)

```
Index: emotion-chatbot
Dimensions: 1536 (OpenAI ada-002 embeddings)
Metric: cosine

Metadata:
- user_id: UUID
- conversation_id: UUID
- message_id: UUID
- timestamp: ISO8601
- emotion: String
- role: String (user/assistant)
```

### Redis Cache Structure

```
# Session data
session:{user_id} → {token, expires_at, metadata}
TTL: 24 hours

# Conversation context
context:{user_id}:{conversation_id} → {messages[], last_emotion, topics[]}
TTL: 1 hour

# Emotion detection cache
emotion:{message_hash} → {primary, secondary, confidence, scores}
TTL: 5 minutes

# Rate limiting
ratelimit:{user_id}:{endpoint} → {count, reset_at}
TTL: 1 minute
```

## API Endpoints

### Authentication
```
POST   /api/v1/auth/register      # Create new user account
POST   /api/v1/auth/login         # Login and get tokens
POST   /api/v1/auth/refresh       # Refresh access token
POST   /api/v1/auth/logout        # Invalidate tokens
```

### Chat
```
POST   /api/v1/chat/message       # Send message (HTTP)
WS     /api/v1/chat/ws            # WebSocket connection
GET    /api/v1/chat/history       # Get conversation history
GET    /api/v1/chat/conversations # List all conversations
POST   /api/v1/chat/conversations # Create new conversation
DELETE /api/v1/chat/{id}          # Delete conversation
```

### Emotion
```
POST   /api/v1/emotion/detect     # Detect emotion from text
GET    /api/v1/emotion/history    # Get user's emotion history
GET    /api/v1/emotion/analytics  # Get emotion analytics/trends
```

### Memory
```
GET    /api/v1/memory/context     # Get conversation context
GET    /api/v1/memory/insights    # Get memory insights
POST   /api/v1/memory/clear       # Clear user memory
```

### Health
```
GET    /api/v1/health             # Basic health check
GET    /api/v1/health/db          # Database health
GET    /api/v1/health/services    # All services health
```

## Security Architecture

### Authentication Flow

```
1. REGISTRATION
   User → POST /auth/register → {email, username, password}
   ↓
   Backend: Hash password with bcrypt (12 rounds)
   ↓
   Store in PostgreSQL
   ↓
   Return: {user_id, username, email}

2. LOGIN
   User → POST /auth/login → {email, password}
   ↓
   Backend: Verify password
   ↓
   Generate JWT tokens:
   - Access token (30 min, HS256)
   - Refresh token (7 days, stored in Redis)
   ↓
   Return: {access_token, refresh_token, user}

3. AUTHENTICATED REQUEST
   User → GET /chat/history
   Headers: Authorization: Bearer <access_token>
   ↓
   Backend: Verify JWT signature and expiration
   ↓
   Extract user_id from token
   ↓
   Process request with user context

4. TOKEN REFRESH
   User → POST /auth/refresh → {refresh_token}
   ↓
   Backend: Verify refresh token in Redis
   ↓
   Generate new access token
   ↓
   Return: {access_token}
```

### Security Measures

- **Password Security:** bcrypt hashing with 12 rounds
- **Token Security:** JWT with HS256, short expiration times
- **HTTPS Only:** All production traffic over TLS
- **CORS:** Configured for specific frontend origin
- **Rate Limiting:** Per-user, per-endpoint limits
- **Input Validation:** Pydantic schemas for all inputs
- **SQL Injection Prevention:** SQLAlchemy ORM
- **XSS Prevention:** Content sanitization
- **CSRF Protection:** Token-based authentication

## Deployment Architecture

### Development Environment

```
Docker Compose:
├── api (FastAPI)
├── postgres (PostgreSQL 14)
├── redis (Redis 7)
├── neo4j (Neo4j 5)
└── client (Vite dev server)
```

### Production Environment (Recommended)

```
Cloud Provider: AWS/GCP/Azure

Frontend:
└── Vercel/Netlify (Static hosting + CDN)

Backend:
├── ECS/EKS/Cloud Run (Container orchestration)
├── Load Balancer (ALB/Cloud Load Balancing)
└── Auto-scaling (based on CPU/memory)

Databases:
├── RDS PostgreSQL (Multi-AZ)
├── ElastiCache Redis (Cluster mode)
├── Neo4j Aura (Managed)
└── Pinecone (Managed)

Monitoring:
├── CloudWatch/Stackdriver (Logs & metrics)
├── Sentry (Error tracking)
└── DataDog (APM)
```

## Performance Considerations

### Caching Strategy

1. **Redis Layers:**
   - L1: Session data (24h TTL)
   - L2: Conversation context (1h TTL)
   - L3: Emotion detection results (5min TTL)

2. **Database Optimization:**
   - Indexed queries on user_id, timestamp
   - Connection pooling (20 connections)
   - Read replicas for analytics queries

3. **API Optimization:**
   - Response compression (gzip)
   - Pagination for list endpoints
   - Lazy loading for conversation history

### Scalability

- **Horizontal Scaling:** Stateless API servers
- **Database Sharding:** By user_id for high volume
- **CDN:** Static assets and frontend
- **Async Processing:** Background jobs for analytics
- **WebSocket:** Sticky sessions with Redis pub/sub

## Monitoring & Observability

### Key Metrics

```
Application:
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Active WebSocket connections
- Messages per second

AI/ML:
- Emotion detection latency
- Emotion detection accuracy
- LLM response time
- Memory retrieval time

Infrastructure:
- CPU/Memory usage
- Database connections
- Cache hit rate
- Queue depth
```

### Logging

```python
# Structured logging with context
{
  "timestamp": "2026-04-07T10:30:00Z",
  "level": "INFO",
  "service": "emotion-detection",
  "user_id": "uuid",
  "event": "emotion_detected",
  "emotion": "stress",
  "confidence": 0.92,
  "latency_ms": 45
}
```

## Future Enhancements

1. **Multi-modal Input:** Voice and image emotion detection
2. **Personalization:** User-specific response styles
3. **Analytics Dashboard:** Emotion trends and insights
4. **Mobile Apps:** Native iOS/Android applications
5. **Multi-language Support:** i18n for global users
6. **Advanced Memory:** Long-term memory with forgetting curves
7. **Therapist Mode:** Professional mental health support features
8. **Group Chat:** Multi-user conversations with emotion awareness

## Conclusion

This architecture provides a scalable, maintainable foundation for an emotion-aware chatbot system. The separation of concerns between frontend, backend, and data layers allows for independent scaling and development. The multi-layered memory system ensures rich context awareness, while the emotion detection pipeline enables empathetic responses.

