import os
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Redis cache imports
from memory.redis_cache import (
    create_redis_cache,
    create_cache_helpers,
    EmotionCache,
    SessionCache,
    ContextCache,
)

# Mem0 memory integration
from memory.mem0_integration import (
    create_mem0_manager,
    Mem0Config,
    Mem0MemoryManager,
)

# Ollama LLM integration
from response.ollama_llm import create_ollama_llm, OllamaLLM

logger = logging.getLogger(__name__)

# Import everything from response_generator — model loads once here
from response.response_generator import (
    TextRequest,
    EmotionResponse,
    predict_emotions,
    compose_response,
    device,
)

# Import emotion engine components
from core.emotion_engine import (
    build_emotion_manager,
    EmotionManager,
    AffectState,
    render_affect_line,
)

# Import context manager integration (Task 1.3)
from core.context_manager_integration import (
    ChatContextManager,
    create_session_context_manager,
)

# =============================================================================
# REDIS CACHE INITIALIZATION
# =============================================================================
redis_cache = create_redis_cache()
cache_helpers = create_cache_helpers(redis_cache)
emotion_cache: EmotionCache = cache_helpers["emotion"]
session_cache: SessionCache = cache_helpers["session"]
context_cache: ContextCache = cache_helpers["context"]

# =============================================================================
# MEM0 MEMORY INITIALIZATION (REQUIRED - no fallback)
# =============================================================================
try:
    mem0_config = Mem0Config.from_env()
    mem0_manager = create_mem0_manager(mem0_config)
    logger.info("Mem0 memory system initialized successfully")
except ImportError as e:
    raise RuntimeError(
        "Mem0 (mem0ai) is required but not installed. Install with: pip install mem0ai"
    ) from e
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize Mem0 memory system. Check your configuration (.env file). "
        f"Required: Vector store (Qdrant/Chroma), Embedder (OpenAI/Ollama). "
        f"Original error: {e}"
    ) from e

# =============================================================================
# OLLAMA LLM INITIALIZATION
# =============================================================================
try:
    ollama_llm = create_ollama_llm()
    logger.info("Ollama LLM initialized successfully")
except Exception as e:
    logger.warning(f"Ollama LLM initialization failed: {e}. Using fallback responses.")
    ollama_llm = None

# =============================================================================
# SESSION MANAGEMENT (With Redis caching and Mem0 memory)
# =============================================================================
sessions: Dict[str, Dict[str, Any]] = {}

def get_or_create_session(user_id: str) -> Dict[str, Any]:
    """
    Get or create a session for a user with Redis caching.
    Each session contains:
    - emotion_manager: EmotionManager instance for tiered affect tracking
    - context_manager: ChatContextManager for memory retrieval and context assembly
    - history: List of message objects for context
    - created_at: Session creation timestamp
    
    Sessions are cached in Redis for persistence across restarts.
    """
    # Try to get from in-memory cache first
    if user_id in sessions:
        return sessions[user_id]
    
    # Try to get from Redis cache
    if redis_cache.is_enabled():
        cached_session = session_cache.get_session(user_id)
        if cached_session:
            try:
                # Reconstruct session objects from cached data
                # Note: Complex objects like EmotionManager need special handling
                emot_mgr = build_emotion_manager()
                ctx_mgr = create_session_context_manager(
                    emotion_manager=emot_mgr,
                    system_prompt=None,
                )
                
                session = {
                    "emotion_manager": emot_mgr,
                    "context_manager": ctx_mgr,
                    "history": [Msg(m["content"], m["message_id"]) for m in cached_session.get("history", [])],
                    "created_at": cached_session.get("created_at", time.time()),
                }
                
                sessions[user_id] = session
                logger.info(f"Session restored from Redis for user {user_id}")
                return session
            except Exception as e:
                logger.warning(f"Failed to restore session from Redis for {user_id}: {e}")
    
    # Create new session
    emot_mgr = build_emotion_manager()
    ctx_mgr = create_session_context_manager(
        emotion_manager=emot_mgr,
        system_prompt=None,
    )
    
    sessions[user_id] = {
        "emotion_manager": emot_mgr,
        "context_manager": ctx_mgr,
        "history": [],
        "created_at": time.time()
    }
    
    # Cache in Redis
    if redis_cache.is_enabled():
        try:
            session_cache.set_session(user_id, {
                "history": [],
                "created_at": sessions[user_id]["created_at"],
            })
        except Exception as e:
            logger.warning(f"Failed to cache session in Redis for {user_id}: {e}")
    
    logger.info(f"New session created for user {user_id} with memory retrieval")
    return sessions[user_id]

# Helper class for message objects
class Msg:
    """Simple message object for emotion engine."""
    def __init__(self, content: str, message_id: str):
        self.content = content
        self.message_id = message_id

# =============================================================================
# NEW PYDANTIC MODELS FOR ENHANCED API
# =============================================================================
class ChatRequest(BaseModel):
    text: str
    user_id: str = "default"
    threshold: float = 0.5
    top_k: Optional[int] = None

class AffectStateResponse(BaseModel):
    """Emotional state across all three tiers."""
    situational_vad: Dict[str, float]
    short_term_vad: Dict[str, float]
    long_term_vad: Dict[str, float]
    situational_bars: Dict[str, float]
    short_term_bars: Dict[str, float]
    long_term_bars: Dict[str, float]
    stm_dominant: str
    ltm_dominant: str
    trend: str
    confidence: float

class ChatResponse(BaseModel):
    """Enhanced chat response with full affect state."""
    text: str
    emotions: List[Dict[str, Any]]
    used_fallback: bool
    bot_response: str
    affect_state: AffectStateResponse
    emotional_events: List[Dict[str, Any]]
    device: str
    session_info: Optional[Dict[str, Any]] = None

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Emotion-Aware Chatbot API",
    description="Backend API for multi-label emotion classification using LoRA.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
def read_root():
    return {"message": "Emotion-Aware Chatbot Backend is running!"}


@app.post("/predict", response_model=EmotionResponse)
def analyze_emotion(request: TextRequest):
    """Raw emotion classification with Redis caching — returns scored label list from the LoRA model."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Try to get from cache first
        if redis_cache.is_enabled():
            cached_result = emotion_cache.get_emotion(request.text)
            if cached_result is not None:
                logger.info("Emotion prediction served from cache")
                return EmotionResponse(**cached_result)
        
        # Cache miss - perform prediction
        emotions, used_fallback, all_scores = predict_emotions(
            request.text, request.threshold, request.top_k
        )
        
        # Build response
        response_data = {
            "text": request.text,
            "emotions": emotions,
            "used_fallback": used_fallback,
            "all_scores": all_scores if request.include_all_scores else None,
            "device": str(device),
        }
        
        # Cache the result
        if redis_cache.is_enabled():
            try:
                emotion_cache.set_emotion(request.text, response_data)
            except Exception as e:
                logger.warning(f"Failed to cache emotion result: {e}")
        
        return EmotionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_with_emotion(request: ChatRequest):
    """
    Enhanced emotion-aware chat endpoint with tiered affect tracking, memory retrieval, and Mem0 integration.
    
    This endpoint:
    1. Maintains per-user session state
    2. Tracks emotions across 3 tiers (situational, short-term, long-term)
    3. Retrieves relevant memories from Mem0 (intelligent memory system)
    4. Builds affect-aware context for LLM
    5. Returns full affect state and emotional events
    6. Generates empathetic responses
    7. Stores conversation in Mem0 for long-term memory
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Get or create user session
        session = get_or_create_session(request.user_id)
        mgr: EmotionManager = session["emotion_manager"]
        ctx_mgr: ChatContextManager = session["context_manager"]
        history: List[Msg] = session["history"]
        
        # Create message object
        msg = Msg(request.text, uuid.uuid4().hex)
        history.append(msg)
        
        # Process turn through emotion engine
        # This updates all 3 tiers and generates emotional events
        window = history[-mgr.context_turns:]
        events = mgr.process_turn(msg, window)
        
        # Get current affect state (with wall-clock decay applied)
        affect: AffectState = mgr.affect_state()
        
        # Get raw emotion predictions (for backward compatibility)
        emotions, used_fallback, all_scores = predict_emotions(
            request.text, request.threshold, request.top_k
        )
        
        # === Mem0: Retrieve relevant memories ===
        try:
            relevant_memories = mem0_manager.search_memories(
                query=request.text,
                user_id=request.user_id,
                limit=5,
            )
            logger.info(f"Retrieved {len(relevant_memories)} memories from Mem0 for user {request.user_id}")
        except Exception as e:
            logger.error(f"Mem0 memory retrieval failed for user {request.user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Memory retrieval error: {e}")
        
        # === Context Manager Integration ===
        # Add the turn to context manager
        ctx_mgr.add_turn(user_message=request.text)
        
        # Build LLM context with memory retrieval and affect injection
        llm_context = ctx_mgr.build_llm_context(
            current_query=request.text,
            affect_state=affect,
        )
        
        # Store significant emotional events in long-term memory (both systems)
        if events:
            for event in events:
                if event.delta_magnitude >= 0.35 and event.confidence >= 0.55:
                    event_desc = (
                        f"User expressed {event.tier} {event.label} "
                        f"(confidence: {event.confidence:.2f}). "
                        f"Context: {request.text[:100]}"
                    )
                    
                    # Store in context manager
                    ctx_mgr.remember_emotional_event(
                        event_description=event_desc,
                        affect_state=affect,
                        importance=min(1.0, event.delta_magnitude * event.confidence),
                    )
                    
                    # Store in Mem0
                    try:
                        mem0_manager.add_emotional_event(
                            event_description=event_desc,
                            user_id=request.user_id,
                            emotion_state={
                                "dominant_emotion": event.label.value,
                                "confidence": event.confidence,
                                "valence": affect.short_term_vad.valence,
                                "arousal": affect.short_term_vad.arousal,
                                "dominance": affect.short_term_vad.dominance,
                            },
                            importance=min(1.0, event.delta_magnitude * event.confidence),
                        )
                    except Exception as e:
                        logger.error(f"Failed to store emotional event in Mem0: {e}")
                        raise HTTPException(status_code=500, detail=f"Memory write error: {e}")
        
        # Generate empathetic response using Ollama LLM
        if ollama_llm:
            try:
                # Build conversation history for context
                conversation_history = []
                for h_msg in history[-6:]:  # Last 6 messages
                    conversation_history.append({
                        "role": "user",
                        "content": h_msg.content
                    })
                
                # Generate emotion-aware response with Ollama
                bot_response = ollama_llm.generate_emotion_aware_response(
                    user_message=request.text,
                    emotion_state={
                        "dominant_emotion": affect.stm_dominant.value,
                        "confidence": affect.confidence,
                        "valence": affect.short_term_vad.valence,
                        "arousal": affect.short_term_vad.arousal,
                        "dominance": affect.short_term_vad.dominance,
                        "trend": affect.trend,
                    },
                    memories=[
                        {"memory": m.get("memory", m.get("content", ""))}
                        for m in relevant_memories
                    ],
                    conversation_history=conversation_history[:-1],  # Exclude current message
                )
                logger.info(f"Generated response using Ollama (phi4-mini)")
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}. Using fallback.")
                bot_response = compose_response(request.text, emotions, used_fallback, all_scores)
        else:
            # Fallback to rule-based response generator
            bot_response = compose_response(request.text, emotions, used_fallback, all_scores)
        
        # Add bot response to context
        ctx_mgr.add_turn(user_message=None, assistant_response=bot_response)
        
        # === Mem0: Store conversation turn ===
        try:
            mem0_manager.add_conversation_turn(
                user_message=request.text,
                assistant_message=bot_response,
                user_id=request.user_id,
                emotion_state={
                    "dominant_emotion": affect.stm_dominant.value,
                    "confidence": affect.confidence,
                    "valence": affect.short_term_vad.valence,
                    "arousal": affect.short_term_vad.arousal,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store conversation turn in Mem0: {e}")
            raise HTTPException(status_code=500, detail=f"Memory write error: {e}")
        
        # Log context diagnostics
        logger.info(
            f"Context for user {request.user_id}: "
            f"{llm_context['diagnostics']['total_tokens']}/{llm_context['diagnostics']['budget_tokens']} tokens, "
            f"{llm_context['diagnostics']['memories_retrieved']} memories retrieved, "
            f"{len(relevant_memories)} Mem0 memories found"
        )
        
        # Build response with full affect state and context info
        return ChatResponse(
            text=request.text,
            emotions=emotions,
            used_fallback=used_fallback,
            bot_response=bot_response,
            affect_state=AffectStateResponse(
                situational_vad=affect.situational_vad.to_dict(),
                short_term_vad=affect.short_term_vad.to_dict(),
                long_term_vad=affect.long_term_vad.to_dict(),
                situational_bars=affect.situational_bar,
                short_term_bars=affect.short_term_bar,
                long_term_bars=affect.long_term_bar,
                stm_dominant=affect.stm_dominant.value,
                ltm_dominant=affect.ltm_dominant.value,
                trend=affect.trend,
                confidence=affect.confidence
            ),
            emotional_events=[e.to_dict() for e in events],
            device=str(device),
            session_info={
                "message_count": len(history),
                "session_age_seconds": time.time() - session["created_at"],
                "memories_retrieved": llm_context['diagnostics']['memories_retrieved'],
                "context_tokens": llm_context['diagnostics']['total_tokens'],
                "mem0_memories": len(relevant_memories),
            }
        )
    except Exception as e:
        logger.exception(f"Chat error for user {request.user_id}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/session/{user_id}/affect")
def get_affect_state(user_id: str):
    """
    Get current affect state for a user session without sending a message.
    Useful for checking emotional state or for visualization.
    """
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Send a message first.")
    
    try:
        session = sessions[user_id]
        mgr: EmotionManager = session["emotion_manager"]
        affect: AffectState = mgr.affect_state()
        
        return {
            "affect_state": AffectStateResponse(
                situational_vad=affect.situational_vad.to_dict(),
                short_term_vad=affect.short_term_vad.to_dict(),
                long_term_vad=affect.long_term_vad.to_dict(),
                situational_bars=affect.situational_bar,
                short_term_bars=affect.short_term_bar,
                long_term_bars=affect.long_term_bar,
                stm_dominant=affect.stm_dominant.value,
                ltm_dominant=affect.ltm_dominant.value,
                trend=affect.trend,
                confidence=affect.confidence
            ).dict(),
            "session_info": {
                "message_count": len(session["history"]),
                "session_age_seconds": time.time() - session["created_at"],
                "created_at": session["created_at"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{user_id}/events")
def get_emotional_events(user_id: str, limit: int = 50):
    """
    Get emotional event history for a user.
    Events include spikes, shifts, reinforcements, and promotions.
    """
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session = sessions[user_id]
        mgr: EmotionManager = session["emotion_manager"]
        events = mgr.events()
        
        # Return most recent events
        recent_events = events[-limit:] if len(events) > limit else events
        
        return {
            "user_id": user_id,
            "total_events": len(events),
            "events": [e.to_dict() for e in recent_events]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{user_id}")
def delete_session(user_id: str):
    """Delete a user session (for testing or user request) and clear from Redis cache."""
    deleted = False
    
    # Delete from in-memory
    if user_id in sessions:
        del sessions[user_id]
        deleted = True
    
    # Delete from Redis
    if redis_cache.is_enabled():
        try:
            if session_cache.delete_session(user_id):
                deleted = True
        except Exception as e:
            logger.warning(f"Failed to delete session from Redis for {user_id}: {e}")
    
    if deleted:
        return {"message": f"Session for user {user_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions/active")
def get_active_sessions():
    """Get list of active sessions (for monitoring/debugging)."""
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {
                "user_id": user_id,
                "message_count": len(session["history"]),
                "age_seconds": time.time() - session["created_at"],
                # BUG FIX #4: use direct key access with a guard to avoid
                # AttributeError when context_manager is None.
                "context_metrics": (
                    session["context_manager"].get_metrics()
                    if session.get("context_manager") is not None
                    else {}
                ),
            }
            for user_id, session in sessions.items()
        ]
    }


@app.get("/session/{user_id}/context_diagnostics")
def get_context_diagnostics(user_id: str):
    """
    Get context manager diagnostics for a session.
    Shows memory retrieval stats, token usage, etc.
    """
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session = sessions[user_id]
        ctx_mgr: ChatContextManager = session.get("context_manager")
        
        if not ctx_mgr:
            raise HTTPException(status_code=404, detail="Context manager not initialized for this session.")
        
        metrics = ctx_mgr.get_metrics()
        
        return {
            "user_id": user_id,
            "session_age_seconds": time.time() - session["created_at"],
            "context_metrics": metrics,
            "memory_count": metrics.get("long_term_memories", 0),
            "emotional_events": metrics.get("emotional_events", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CACHE MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/cache/stats")
def get_cache_stats():
    """Get Redis cache statistics."""
    if not redis_cache.is_enabled():
        return {"enabled": False, "message": "Redis cache is disabled"}
    
    try:
        stats = redis_cache.get_stats()
        return {
            "enabled": True,
            "redis_stats": stats,
            "cache_types": {
                "emotion": "Emotion analysis results (1 hour TTL)",
                "session": "User session state (24 hour TTL)",
                "context": "LLM context assembly (5 minute TTL)",
                "memory": "Memory retrieval results (10 minute TTL)",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear")
def clear_cache(cache_type: Optional[str] = None):
    """
    Clear cache entries.
    
    Parameters:
    - cache_type: Optional. One of "emotion", "session", "context", "memory", or None for all.
    """
    if not redis_cache.is_enabled():
        raise HTTPException(status_code=503, detail="Redis cache is disabled")
    
    try:
        if cache_type is None:
            # Clear all cache
            redis_cache.clear_all()
            return {"message": "All cache cleared successfully"}
        elif cache_type in ["emotion", "session", "context", "memory"]:
            # Clear specific cache type
            count = redis_cache.delete_pattern(cache_type, "*")
            return {"message": f"Cleared {count} {cache_type} cache entries"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache_type. Must be one of: emotion, session, context, memory, or None"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/health")
def cache_health():
    """Check Redis cache health."""
    if not redis_cache.is_enabled():
        return {
            "status": "disabled",
            "message": "Redis cache is disabled",
        }
    
    try:
        # Test connection
        redis_cache._client.ping()
        stats = redis_cache.get_stats()
        
        return {
            "status": "healthy",
            "enabled": True,
            "total_keys": stats.get("total_keys", 0),
            "hit_rate": f"{stats.get('hit_rate', 0) * 100:.2f}%",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "enabled": True,
            "error": str(e),
        }


# =============================================================================
# MEM0 MEMORY ENDPOINTS
# NOTE: /memory/health MUST be declared before /memory/{user_id} so FastAPI
# does not swallow the literal path segment "health" as a user_id value.
# =============================================================================

@app.get("/memory/health")
def memory_health():
    """Check Mem0 memory system health."""
    if not mem0_manager:
        return {
            "status": "disabled",
            "message": "Mem0 memory system is not initialized",
        }

    return {
        "status": "healthy",
        "enabled": True,
        "config": {
            "vector_store": mem0_manager.config.vector_store,
            "embedder": f"{mem0_manager.config.embedder_provider}/{mem0_manager.config.embedder_model}",
            "graph_enabled": mem0_manager.config.enable_graph,
        }
    }


@app.get("/memory/{user_id}")
def get_user_memories(user_id: str, limit: Optional[int] = 20):
    """
    Get all memories for a user from Mem0.
    
    Parameters:
    - user_id: User identifier
    - limit: Maximum number of memories to return
    """
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        memories = mem0_manager.get_all_memories(user_id=user_id, limit=limit)
        
        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "memories": memories,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{user_id}/search")
def search_user_memories(user_id: str, query: str, limit: int = 5):
    """
    Search memories for a user using semantic search.
    
    Parameters:
    - user_id: User identifier
    - query: Search query
    - limit: Maximum results
    """
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        memories = mem0_manager.search_memories(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        
        return {
            "user_id": user_id,
            "query": query,
            "results": memories,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{user_id}/emotional")
def get_emotional_memories(
    user_id: str,
    emotion: Optional[str] = None,
    min_importance: float = 0.0,
    limit: int = 10,
):
    """
    Get emotional memories for a user.
    
    Parameters:
    - user_id: User identifier
    - emotion: Optional emotion filter (e.g., "joy", "sadness")
    - min_importance: Minimum importance threshold (0-1)
    - limit: Maximum results
    """
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        memories = mem0_manager.search_emotional_memories(
            user_id=user_id,
            emotion=emotion,
            min_importance=min_importance,
            limit=limit,
        )
        
        return {
            "user_id": user_id,
            "emotion_filter": emotion,
            "min_importance": min_importance,
            "total_results": len(memories),
            "memories": memories,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{user_id}/summary")
def get_memory_summary(user_id: str):
    """
    Get a summary of user's memory profile.
    
    Returns statistics about stored memories, emotion distribution, etc.
    """
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        summary = mem0_manager.get_memory_summary(user_id=user_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{user_id}/add")
def add_user_memory(
    user_id: str,
    text: str,
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Manually add a memory for a user.
    
    Parameters:
    - user_id: User identifier
    - text: Memory content
    - importance: Importance score (0-1)
    - metadata: Optional metadata
    """
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        result = mem0_manager.add_memory(
            text=text,
            user_id=user_id,
            metadata=metadata,
        )
        
        return {
            "user_id": user_id,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{user_id}/{memory_id}")
def delete_user_memory(user_id: str, memory_id: str):
    """Delete a specific memory."""
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        success = mem0_manager.delete_memory(
            memory_id=memory_id,
            user_id=user_id,
        )
        
        if success:
            return {"message": f"Memory {memory_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{user_id}")
def delete_all_user_memories(user_id: str):
    """Delete all memories for a user."""
    if not mem0_manager:
        raise HTTPException(status_code=503, detail="Mem0 memory system not available")
    
    try:
        success = mem0_manager.delete_all_memories(user_id=user_id)
        
        if success:
            return {"message": f"All memories for user {user_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete memories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 EMOTION-AWARE CHATBOT BACKEND - PRODUCTION READY v3.0")
    print("="*70)
    print(f"📍 Server: http://localhost:8000")
    print(f"📄 API docs: http://localhost:8000/docs")
    print(f"🔧 Device: {device}")
    print(f"🧠 Emotion Engine: ACTIVE (Tiered Affect Tracking)")
    print(f"🧩 Context Engine: ACTIVE (Memory Retrieval + Affect Injection)")
    print(f"💾 Redis Cache: {'ENABLED' if redis_cache.is_enabled() else 'DISABLED'}")
    if redis_cache.is_enabled():
        print(f"   └─ Caching: Emotions, Sessions, Context, Memory")
    print(f"🧬 Mem0 Memory: ENABLED")
    print(f"   └─ Vector Store: {mem0_manager.config.vector_store}")
    print(f"   └─ Embedder: {mem0_manager.config.embedder_provider}/{mem0_manager.config.embedder_model}")
    print(f"   └─ Features: Semantic search, temporal context, auto-extraction")
    print(f"🤖 LLM Engine: {'Ollama (phi4-mini)' if ollama_llm else 'Rule-based fallback'}")
    if ollama_llm:
        print(f"   └─ Model: {ollama_llm.model}")
        print(f"   └─ Local inference (no API costs)")
        print(f"   └─ Emotion-aware response generation")
    print(f"📊 Features:")
    print(f"   ✓ 3-Tier Emotion Memory (Situational/Short-term/Long-term)")
    print(f"   ✓ 28-Label Multi-emotion Detection")
    print(f"   ✓ VAD Continuous Affect Representation")
    print(f"   ✓ Wall-clock Decay with Half-lives")
    print(f"   ✓ Emotional Event Logging")
    print(f"   ✓ Per-User Session Management")
    if redis_cache.is_enabled():
        print(f"   ✓ Redis Caching Layer (Performance Boost)")
    print(f"   ✓ Mem0 Intelligent Memory System")
    print(f"   ✓ Automatic Memory Extraction")
    print(f"   ✓ Semantic Memory Search")
    print(f"   ✓ Emotional Memory Tagging")
    if ollama_llm:
        print(f"   ✓ Local LLM Response Generation (phi4-mini)")
        print(f"   ✓ Context-Aware Conversations")
        print(f"   ✓ Memory-Enhanced Responses")
    print(f"   ✓ Token-Budgeted Context Assembly")
    print(f"   ✓ Affect-Aware Prompt Injection")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")