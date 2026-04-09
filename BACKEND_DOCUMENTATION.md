# Emotion-Aware Chatbot - Backend Documentation

## Overview

The backend is built with **FastAPI** and **LangGraph**, providing an emotion-aware conversational AI system that:
- Detects user emotions in real-time using RoBERTa
- Maintains conversation context using vector, graph, and cache storage
- Generates empathetic, context-aware responses
- Tracks emotional patterns over time

---

## Architecture

### Tech Stack

**Core Framework:**
- **FastAPI** - Modern, fast web framework for building APIs
- **LangGraph** - Workflow orchestration for LLM applications
- **Python 3.11+**

**AI/ML:**
- **RoBERTa** - Emotion classification model
- **LangChain** - LLM integration and prompt management
- **OpenAI/Anthropic** - LLM for response generation

**Data Storage:**
- **PostgreSQL** - Primary relational database (users, conversations)
- **Pinecone/Weaviate** - Vector database for semantic search
- **Neo4j** - Graph database for relationship mapping
- **Redis** - Caching and session management

**Authentication:**
- **JWT** - Token-based authentication
- **bcrypt** - Password hashing

---

## Project Structure

```
server/
├── app/
│   ├── api/                      # API layer
│   │   └── v1/
│   │       ├── endpoints/        # Route handlers
│   │       │   ├── auth.py       # Authentication endpoints
│   │       │   ├── chat.py       # Chat endpoints
│   │       │   ├── emotion.py    # Emotion detection
│   │       │   ├── memory.py     # Memory management
│   │       │   ├── user.py       # User management
│   │       │   └── health.py     # Health checks
│   │       ├── deps/             # Dependencies
│   │       │   ├── auth_deps.py  # Auth dependencies
│   │       │   └── db_deps.py    # Database dependencies
│   │       └── router.py         # API router
│   ├── core/                     # Core configuration
│   │   ├── config.py             # Settings management
│   │   ├── security.py           # Security utilities
│   │   └── exceptions.py         # Custom exceptions
│   ├── models/                   # Database models
│   │   ├── user.py
│   │   ├── conversation.py
│   │   └── emotion.py
│   ├── schemas/                  # Pydantic schemas
│   │   ├── user.py
│   │   ├── chat.py
│   │   ├── emotion.py
│   │   └── memory.py
│   ├── services/                 # Business logic
│   │   ├── emotion/
│   │   │   ├── detector.py       # Emotion detection
│   │   │   ├── analyzer.py       # Emotion analysis
│   │   │   └── models/           # Model weights
│   │   ├── memory/
│   │   │   ├── vector_store.py   # Vector DB client
│   │   │   ├── graph_store.py    # Neo4j client
│   │   │   ├── cache_manager.py  # Redis client
│   │   │   └── memory_service.py # Memory orchestration
│   │   ├── chat/
│   │   │   ├── langgraph_graph.py    # LangGraph workflow
│   │   │   ├── node_functions.py     # Workflow nodes
│   │   │   └── response_generator.py # Response generation
│   │   └── user_service.py
│   ├── db/                       # Database utilities
│   │   ├── session.py            # DB session
│   │   └── init_db.py            # Initialization
│   ├── utils/                    # Utilities
│   │   ├── text_processing.py
│   │   ├── embeddings.py
│   │   └── logging.py
│   └── main.py                   # Application entry
├── tests/                        # Tests
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── scripts/                      # Utility scripts
│   ├── train_emotion_model.py
│   └── seed_data.py
├── alembic/                      # Database migrations
├── .env.example
├── requirements.txt
├── Dockerfile
└── pyproject.toml
```

---

## Data Flow

### Complete Request Flow

```
1. USER INPUT
   ↓
2. EMOTION DETECTION
   - Preprocess text (lowercase, emoji handling)
   - RoBERTa inference
   - Output: Primary & secondary emotions with confidence scores
   ↓
3. MEMORY RETRIEVAL (Parallel)
   ├─ Vector Search: Semantic similarity search
   ├─ Graph Query: User facts and relationships
   └─ Cache: Recent conversation context
   ↓
4. CONTEXT ASSEMBLY
   - Combine retrieved memories
   - Identify patterns
   - Build user profile
   ↓
5. RESPONSE GENERATION (LangGraph)
   - Emotion-aware prompt construction
   - LLM inference
   - Response validation
   ↓
6. OUTPUT
   - Return empathetic response
   ↓
7. STORAGE
   - Save to vector DB
   - Update graph relationships
   - Update cache
   - Log emotion data
```

### Example Scenario

**Input:** "I feel very stressed about exams"

**Processing:**
1. **Emotion Detection:** stress (0.92), anxiety (0.78)
2. **Memory Retrieval:**
   - Vector: Past conversations about exams
   - Graph: User is CS student, 3 past exam stress instances
   - Cache: Last topic was "study_schedule"
3. **Context:** Recurring exam anxiety, prefers practical solutions
4. **Response:** "I remember you mentioned exams a couple of weeks ago too, and you're in CS, right? This recurring stress around exam time makes sense. Want me to help you build a focused 3-day study plan, or would a quick mindfulness exercise help right now?"
5. **Storage:** Update all databases with new interaction

---

## API Endpoints

### Authentication

```http
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/refresh
POST /api/v1/auth/logout
```

### Chat

```http
POST   /api/v1/chat/message          # Send message
GET    /api/v1/chat/history          # Get conversation history
GET    /api/v1/chat/conversations    # List all conversations
DELETE /api/v1/chat/{id}             # Delete conversation
```

### Emotion

```http
POST /api/v1/emotion/detect          # Detect emotion from text
GET  /api/v1/emotion/history         # Get emotion history
GET  /api/v1/emotion/analytics       # Get emotion analytics
```

### Memory

```http
GET    /api/v1/memory/context        # Get conversation context
GET    /api/v1/memory/insights       # Get memory insights
POST   /api/v1/memory/clear          # Clear memory
```

### User

```http
GET /api/v1/user/profile             # Get user profile
PUT /api/v1/user/profile             # Update profile
GET /api/v1/user/preferences         # Get preferences
PUT /api/v1/user/preferences         # Update preferences
```

### Health

```http
GET /api/v1/health                   # Health check
GET /api/v1/health/db                # Database health
GET /api/v1/health/services          # Services health
```

---

## Core Components

### 1. Emotion Detection Service

**Location:** `app/services/emotion/detector.py`

**Responsibilities:**
- Load and manage RoBERTa model
- Preprocess input text
- Perform emotion inference
- Return emotion scores

**Emotions Detected:**
- Happy
- Sad
- Angry
- Anxious
- Stress
- Neutral

**Output Format:**
```python
{
    "primary": "stress",
    "secondary": "anxiety",
    "confidence": 0.92,
    "scores": {
        "happy": 0.05,
        "sad": 0.12,
        "angry": 0.08,
        "anxious": 0.78,
        "stress": 0.92,
        "neutral": 0.15
    }
}
```

### 2. Memory Service

**Location:** `app/services/memory/memory_service.py`

**Components:**

**Vector Store (Pinecone/Weaviate):**
- Stores conversation embeddings
- Enables semantic search
- Retrieves similar past conversations

**Graph Store (Neo4j):**
- Stores user facts and relationships
- Tracks emotional patterns
- Maps conversation topics

**Cache (Redis):**
- Stores recent conversation context
- Session management
- Quick access to working memory

**Unified Interface:**
```python
class MemoryService:
    async def retrieve_context(user_id: str, query: str) -> Context
    async def store_interaction(user_id: str, interaction: Interaction)
    async def get_insights(user_id: str) -> Insights
    async def clear_memory(user_id: str)
```

### 3. LangGraph Workflow

**Location:** `app/services/chat/langgraph_graph.py`

**Workflow Nodes:**
1. **Input Processing** - Validate and preprocess input
2. **Emotion Detection** - Detect user emotion
3. **Memory Retrieval** - Fetch relevant context
4. **Context Assembly** - Combine all context
5. **Response Generation** - Generate empathetic response
6. **Validation** - Validate response quality
7. **Storage** - Store interaction

**State Management:**
```python
class ChatState(TypedDict):
    user_id: str
    message: str
    emotion: EmotionData
    context: Context
    response: str
    metadata: dict
```

### 4. Response Generator

**Location:** `app/services/chat/response_generator.py`

**Features:**
- Emotion-aware prompt templates
- Context injection
- Tone adjustment based on emotion
- Response validation

**Response Templates:**

**Stress/Anxiety:**
```
Validation → Grounding technique → Offer help
"I hear that exams are weighing on you. That tension makes sense. 
Want me to suggest a quick breathing exercise or help break down 
your study plan?"
```

**Sadness:**
```
Acknowledge → Normalize → Gentle check-in
"It sounds like you're going through a rough patch. Those feelings 
are valid. Would talking about it help, or would you prefer a 
distraction?"
```

**Anger:**
```
Validate frustration → De-escalate → Problem-solve
"That's really frustrating when things don't work as expected. 
Let's figure this out together. What would make this better right now?"
```

---

## Database Models

### User Model

```python
class User(Base):
    id: UUID
    email: str
    username: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    preferences: JSON
    is_active: bool
```

### Conversation Model

```python
class Conversation(Base):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
```

### Message Model

```python
class Message(Base):
    id: UUID
    conversation_id: UUID
    role: str  # 'user' or 'assistant'
    content: str
    emotion_data: JSON
    timestamp: datetime
    metadata: JSON
```

### Emotion Log Model

```python
class EmotionLog(Base):
    id: UUID
    user_id: UUID
    message_id: UUID
    primary_emotion: str
    secondary_emotion: str
    confidence: float
    scores: JSON
    timestamp: datetime
```

---

## Configuration

### Environment Variables

```env
# Application
APP_NAME=Emotion Chatbot
APP_VERSION=1.0.0
DEBUG=False
API_V1_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/emotion_chatbot
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector Database
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX=emotion-chatbot

# LLM
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7

# Authentication
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Emotion Model
EMOTION_MODEL_PATH=./models/roberta-emotion
EMOTION_MODEL_DEVICE=cuda  # or 'cpu'

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Neo4j 5+
- Pinecone account (or Weaviate)

### Installation Steps

1. **Clone and navigate:**
```bash
cd server
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database:**
```bash
alembic upgrade head
python scripts/seed_data.py
```

6. **Download emotion model:**
```bash
python scripts/download_model.py
```

7. **Run development server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration

# With coverage
pytest --cov=app tests/
```

### Code Quality

```bash
# Format code
black app/
isort app/

# Lint
flake8 app/
pylint app/

# Type checking
mypy app/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Deployment

### Docker

```bash
# Build image
docker build -t emotion-chatbot-backend .

# Run container
docker run -p 8000:8000 --env-file .env emotion-chatbot-backend
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/emotion_chatbot
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
      - neo4j
  
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: emotion_chatbot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
  
  redis:
    image: redis:7-alpine
  
  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
```

### Production Deployment

**Recommended Stack:**
- **AWS ECS/EKS** or **Google Cloud Run**
- **RDS PostgreSQL** for database
- **ElastiCache Redis** for caching
- **Neo4j Aura** for graph database
- **Pinecone** for vector storage

---

## API Documentation

Once the server is running, access:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## Performance Optimization

### Caching Strategy

1. **Redis Cache:**
   - Recent conversations (TTL: 1 hour)
   - User sessions (TTL: 24 hours)
   - Emotion detection results (TTL: 5 minutes)

2. **Database Indexing:**
   - User ID indexes
   - Timestamp indexes
   - Emotion type indexes

3. **Connection Pooling:**
   - PostgreSQL: 20 connections
   - Redis: 10 connections
   - Neo4j: 5 connections

### Rate Limiting

```python
# Per user limits
- Chat messages: 60/minute
- Emotion detection: 100/minute
- Memory queries: 30/minute
```

---

## Security

### Authentication Flow

1. User registers/logs in
2. Server generates JWT access token (30 min) and refresh token (7 days)
3. Client stores tokens securely
4. Client sends access token in Authorization header
5. Server validates token on each request
6. Client refreshes token when expired

### Security Best Practices

- Passwords hashed with bcrypt (12 rounds)
- JWT tokens signed with HS256
- HTTPS only in production
- CORS configured for frontend origin
- Rate limiting on all endpoints
- Input validation with Pydantic
- SQL injection prevention with SQLAlchemy
- XSS prevention with content sanitization

---

## Monitoring & Logging

### Logging

```python
# Structured logging with context
logger.info(
    "emotion_detected",
    user_id=user_id,
    emotion=emotion,
    confidence=confidence
)
```

### Metrics to Track

- Request latency
- Emotion detection accuracy
- Memory retrieval time
- LLM response time
- Error rates
- Active users
- Conversations per day

### Health Checks

```http
GET /api/v1/health
{
    "status": "healthy",
    "database": "connected",
    "redis": "connected",
    "neo4j": "connected",
    "vector_db": "connected"
}
```

---

## Troubleshooting

### Common Issues

**Database Connection Errors:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string
echo $DATABASE_URL
```

**Redis Connection Errors:**
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG
```

**Emotion Model Loading Errors:**
```bash
# Verify model files exist
ls -la models/roberta-emotion/

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and tests
6. Submit a pull request

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues and questions:
- GitHub Issues: [repository]/issues
- Documentation: [repository]/wiki
- Email: support@example.com
