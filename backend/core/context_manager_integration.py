"""
context_manager_integration.py

Integration layer connecting context_engine.py with the chat backend.
Provides memory-aware, emotion-aware response generation.

This module bridges:
- emotion_engine.py (production emotion tracking)
- context_engine.py (memory retrieval & context assembly)
- main.py (API endpoints)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import emotion engine components (single source of truth for emotion tracking)
from core.emotion_engine import (
    EmotionManager,
    AffectState,
    EmotionalMemoryWriter,
)

# Import context engine components (memory & context assembly)
from core.context_engine import (
    ContextManager,
    MemoryRetriever,
    TiktokenCounter,          # Only tiktoken - no HeuristicTokenCounter
    SentenceTransformerEmbedder,  # Real semantic embedder - no HashingEmbedder
    ContextBudget,
    RetrievalConfig,
    TruncatingSummarizer,
    Role,
    Message,
    ChromaVectorStore,        # Only ChromaDB - no InMemoryVectorStore
)

logger = logging.getLogger(__name__)


# =============================================================================
# Vector Store Creation - ChromaDB REQUIRED (no fallback)
# =============================================================================
def create_vector_store(
    persist_dir: str = "./chroma_db",
    collection_name: str = "emotion_memories",
    embedding_dimension: int = 512,
) -> ChromaVectorStore:
    """
    Create ChromaDB vector store. NO FALLBACK TO IN-MEMORY.
    
    If ChromaDB fails, the application should fail fast and loud.
    Silent fallback to in-memory causes:
    - Data loss on restart/scaling
    - OOM crashes with large datasets
    - Inconsistent behavior between environments
    
    Raises:
        RuntimeError: If ChromaDB cannot be initialized
    """
    try:
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
        )
        logger.info(f"ChromaDB vector store initialized at {persist_dir}")
        return vector_store
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB is required but not installed. Install with: pip install chromadb"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChromaDB at {persist_dir}. "
            f"Check that the directory is writable and ChromaDB is properly installed. "
            f"Original error: {e}"
        ) from e


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_tokens: int = 7168  # For GPT-4 8k context (leaving room for response)
    reserved_for_response: int = 1024
    max_memory_ratio: float = 0.30
    max_system_ratio: float = 0.25
    max_affect_ratio: float = 0.05
    history_soft_cap_tokens: int = 3000
    top_k_memories: int = 5
    tiktoken_model: str = "gpt-4o"  # Model for tiktoken encoding
    chroma_persist_dir: str = "./chroma_db"  # Directory for ChromaDB persistence
    chroma_collection: str = "emotion_memories"  # ChromaDB collection name


class ChatContextManager:
    """
    Wrapper around ContextManager that bridges our emotion_engine.py
    with context_engine.py for memory-aware response generation.
    """
    
    def __init__(
        self,
        emotion_manager: EmotionManager,
        config: Optional[ContextConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.config = config or ContextConfig()
        self.emotion_manager = emotion_manager  # Our production emotion manager from emotion_engine.py
        
        # Initialize tiktoken counter (REQUIRED - no heuristic fallback)
        try:
            self.token_counter = TiktokenCounter(model=self.config.tiktoken_model)
            logger.info(f"TiktokenCounter initialized for model: {self.config.tiktoken_model}")
        except ImportError as e:
            raise RuntimeError(
                "tiktoken is required but not installed. Install with: pip install tiktoken"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize tiktoken for model {self.config.tiktoken_model}. "
                f"Original error: {e}"
            ) from e
        
        # Initialize semantic embedder (REQUIRED - no HashingEmbedder fallback).
        # HashingEmbedder silently returns semantically garbage vectors, making
        # vector search return irrelevant results with no visible error.
        try:
            self.embedder = SentenceTransformerEmbedder()
            logger.info(f"SentenceTransformerEmbedder initialized (dim={self.embedder.dimension})")
        except RuntimeError:
            raise  # Already has a clear message - propagate as-is

        # Initialize ChromaDB vector store (REQUIRED - no fallback).
        # Dimension is derived from the embedder to stay in sync.
        vector_store = create_vector_store(
            persist_dir=self.config.chroma_persist_dir,
            collection_name=self.config.chroma_collection,
            embedding_dimension=self.embedder.dimension,
        )
        
        # Initialize memory retriever
        self.retriever = MemoryRetriever(
            embedder=self.embedder,
            store=vector_store,
            config=RetrievalConfig(
                top_k=self.config.top_k_memories,
                min_score=0.05,
                weight_similarity=0.7,
                weight_recency=0.2,
                weight_importance=0.1,
            ),
        )
        
        # Initialize emotional memory writer (uses our production emotion manager)
        emotional_writer = EmotionalMemoryWriter(
            retriever=self.retriever,
            min_magnitude=0.35,  # Match our delta threshold from architecture
            min_confidence=0.55,  # Match our confidence threshold from architecture
        )
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        
        # Initialize context manager (with emotion manager from emotion_engine.py)
        self.context_manager = ContextManager(
            retriever=self.retriever,
            token_counter=self.token_counter,
            budget=ContextBudget(
                max_tokens=self.config.max_tokens,
                reserved_for_response=self.config.reserved_for_response,
                max_memory_ratio=self.config.max_memory_ratio,
                max_system_ratio=self.config.max_system_ratio,
                max_affect_ratio=self.config.max_affect_ratio,
            ),
            system_prompt=system_prompt,
            summarizer=TruncatingSummarizer(max_chars=500),
            history_soft_cap_tokens=self.config.history_soft_cap_tokens,
            emotion_manager=self.emotion_manager,  # Use production emotion manager
            emotional_writer=emotional_writer,
        )
        
        logger.info("ChatContextManager initialized with production emotion_engine.py tracking")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for empathetic chat."""
        return """You are an empathetic AI assistant specialized in emotional support and understanding.

Your role:
- Listen actively and validate emotions
- Show genuine empathy and understanding
- Ask clarifying questions when appropriate
- Never dismiss or minimize feelings
- Recognize emotional patterns across conversations
- Draw on previous interactions when relevant

Communication style:
- Warm and genuine
- Not overly formal
- Naturally conversational
- Emotionally intelligent
- Respectful of boundaries

You have access to:
- The user's recent conversation history
- Relevant memories from past interactions
- Current emotional state estimate (treat as a signal, not certainty)

Remember: The emotional context is a model estimate to help you be more empathetic. 
It's not a diagnosis. Always respect the user's own description of their feelings."""
    
    def add_turn(
        self,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
    ) -> None:
        """
        Add a conversation turn to the context manager.
        This maintains the conversation history for memory retrieval.
        """
        # Add user message if provided
        if user_message:
            self.context_manager.add_user_message(user_message)
        
        # Add assistant response if provided
        if assistant_response:
            self.context_manager.add_assistant_message(assistant_response)
    
    def build_llm_context(
        self,
        current_query: str,
        affect_state: Optional[AffectState] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete context for LLM including:
        - System prompt
        - Affect block (emotional state)
        - Retrieved memories
        - Recent conversation history
        
        Returns dict with:
        - messages: List of messages in chat format
        - memories: Retrieved memories for logging
        - diagnostics: Context assembly stats
        """
        # Build context (this retrieves memories and assembles prompt)
        snapshot = self.context_manager.build_context(query=current_query)
        
        # Convert to dict format
        return {
            "messages": snapshot.to_chat_format(),
            "memories": [
                {
                    "content": m.memory.content,
                    "score": m.score,
                    "similarity": m.similarity,
                    "importance": m.importance,
                }
                for m in snapshot.memories_used
            ],
            "diagnostics": {
                "total_tokens": snapshot.total_tokens,
                "budget_tokens": snapshot.budget_tokens,
                "memories_retrieved": len(snapshot.memories_used),
                "messages_dropped": snapshot.dropped_history_count,
                **snapshot.diagnostics,
            },
        }
    
    def remember_explicitly(
        self,
        content: str,
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Explicitly store something in long-term memory.
        Use for important facts, user preferences, significant events.
        """
        self.context_manager.remember(
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        logger.info(f"Explicit memory stored: {content[:50]}... (importance={importance})")
    
    def remember_emotional_event(
        self,
        event_description: str,
        affect_state: AffectState,
        importance: float = 0.8,
    ) -> None:
        """
        Store a significant emotional event in memory.
        """
        metadata = {
            "kind": "emotional_event",
            "situational_dominant": affect_state.stm_dominant.value,
            "long_term_dominant": affect_state.ltm_dominant.value,
            "trend": affect_state.trend,
            "confidence": affect_state.confidence,
        }
        
        self.remember_explicitly(
            content=event_description,
            importance=importance,
            metadata=metadata,
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get context manager metrics for monitoring."""
        return self.context_manager.metrics()
    
    def reset_conversation(self, keep_memories: bool = True) -> None:
        """
        Reset conversation history while optionally keeping memories.
        Useful for starting a new session.
        """
        self.context_manager.reset(keep_memories=keep_memories, keep_emotions=False)
        logger.info(f"Conversation reset (keep_memories={keep_memories})")


def create_session_context_manager(
    emotion_manager: EmotionManager,
    system_prompt: Optional[str] = None,
) -> ChatContextManager:
    """
    Factory function to create a ChatContextManager for a user session.
    """
    return ChatContextManager(
        emotion_manager=emotion_manager,
        config=ContextConfig(),
        system_prompt=system_prompt,
    )
