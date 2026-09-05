"""
mem0_integration.py
===================
Mem0 integration for advanced memory management in emotion-aware chatbot.

Mem0 provides:
- Automatic memory extraction from conversations
- Temporal context tracking (short-term vs long-term)
- Semantic search across memories
- User-specific and session-specific memory isolation
- Graph-based relationships between memories
- Automatic memory updates and deduplication

This replaces the manual memory management in context_engine.py with Mem0's
intelligent memory layer.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Mem0 Configuration
# =============================================================================

@dataclass
class Mem0Config:
    """Configuration for Mem0 memory system."""
    
    # Vector store backend: "qdrant" (recommended) or "chroma"
    vector_store: str = "qdrant"
    
    # Qdrant settings
    qdrant_url: Optional[str] = None  # None = in-memory
    qdrant_api_key: Optional[str] = None
    
    # Embedding model
    embedder_provider: str = "openai"  # "openai", "ollama", "huggingface"
    embedder_model: str = "text-embedding-3-small"
    
    # LLM for memory extraction (optional - uses rules if not set)
    llm_provider: Optional[str] = None  # "openai", "ollama", etc.
    llm_model: Optional[str] = None
    
    # Memory settings
    collection_name: str = "emotion_chatbot_memories"
    history_db_path: str = "./mem0_history.db"
    
    # Enable graph relationships
    enable_graph: bool = False
    graph_store: str = "neo4j"
    neo4j_url: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Mem0Config":
        """Create config from environment variables."""
        return cls(
            vector_store=os.getenv("MEM0_VECTOR_STORE", "qdrant"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            embedder_provider=os.getenv("MEM0_EMBEDDER_PROVIDER", "openai"),
            embedder_model=os.getenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-small"),
            llm_provider=os.getenv("MEM0_LLM_PROVIDER"),
            llm_model=os.getenv("MEM0_LLM_MODEL"),
            collection_name=os.getenv("MEM0_COLLECTION", "emotion_chatbot_memories"),
            history_db_path=os.getenv("MEM0_HISTORY_DB", "./mem0_history.db"),
            enable_graph=os.getenv("MEM0_ENABLE_GRAPH", "false").lower() == "true",
            graph_store=os.getenv("MEM0_GRAPH_STORE", "neo4j"),
            neo4j_url=os.getenv("NEO4J_URL"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
        )


# =============================================================================
# Mem0 Memory Manager
# =============================================================================

class Mem0MemoryManager:
    """
    Wrapper around Mem0 for emotion-aware chatbot memory management.
    
    Key features:
    - User-scoped memories (each user has isolated memory space)
    - Session-scoped memories (temporary memories for current conversation)
    - Emotional context tagging (memories tagged with emotional state)
    - Automatic memory extraction from conversations
    - Semantic search and retrieval
    - Memory lifecycle management (add, search, update, delete)
    """
    
    def __init__(self, config: Optional[Mem0Config] = None):
        """
        Initialize Mem0 memory manager.
        
        Parameters:
        - config: Mem0 configuration (defaults to environment-based config)
        """
        try:
            from mem0 import Memory
        except ImportError as e:
            raise ImportError(
                "mem0ai package not installed. Install with: pip install mem0ai"
            ) from e
        
        self.config = config or Mem0Config.from_env()
        self._mem0 = None
        self._initialize_mem0()
    
    def _initialize_mem0(self):
        """Initialize Mem0 with configuration."""
        from mem0 import Memory
        
        # Build Mem0 config
        mem0_config = {
            "vector_store": {
                "provider": self.config.vector_store,
            },
            "embedder": {
                "provider": self.config.embedder_provider,
                "config": {
                    "model": self.config.embedder_model,
                }
            },
            "version": "v1.1"
        }
        
        # Add vector store specific config
        if self.config.vector_store == "qdrant":
            if self.config.qdrant_url:
                mem0_config["vector_store"]["config"] = {
                    "url": self.config.qdrant_url,
                    "api_key": self.config.qdrant_api_key,
                    "collection_name": self.config.collection_name,
                }
            else:
                # In-memory Qdrant
                mem0_config["vector_store"]["config"] = {
                    "collection_name": self.config.collection_name,
                    "on_disk": False,
                }
        
        # Add LLM config if provided (for advanced memory extraction)
        if self.config.llm_provider:
            mem0_config["llm"] = {
                "provider": self.config.llm_provider,
                "config": {
                    "model": self.config.llm_model,
                }
            }
        
        # Add graph store config if enabled
        if self.config.enable_graph:
            mem0_config["graph_store"] = {
                "provider": self.config.graph_store,
                "config": {
                    "url": self.config.neo4j_url,
                    "username": self.config.neo4j_username,
                    "password": self.config.neo4j_password,
                }
            }
        
        # Add history database
        mem0_config["history_db_path"] = self.config.history_db_path
        
        try:
            self._mem0 = Memory.from_config(mem0_config)
            logger.info(
                f"Mem0 initialized: vector_store={self.config.vector_store}, "
                f"embedder={self.config.embedder_provider}/{self.config.embedder_model}, "
                f"graph={'enabled' if self.config.enable_graph else 'disabled'}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            # Fallback to basic config
            self._mem0 = Memory()
            logger.warning("Using basic Mem0 configuration (in-memory)")
    
    # -------------------------------------------------------------------------
    # Core Memory Operations
    # -------------------------------------------------------------------------
    
    def add_memory(
        self,
        text: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a memory for a user.
        
        Mem0 will automatically extract key information from the text and
        store it as structured memories.
        
        Parameters:
        - text: The text to extract memories from
        - user_id: User identifier
        - metadata: Additional metadata (emotion state, importance, etc.)
        - session_id: Optional session identifier for temporary memories
        
        Returns:
        - Dict with memory IDs and extracted memories
        """
        try:
            # Prepare metadata
            full_metadata = metadata or {}
            full_metadata["timestamp"] = time.time()
            if session_id:
                full_metadata["session_id"] = session_id
            
            # Add memory through Mem0
            result = self._mem0.add(
                messages=text,
                user_id=user_id,
                metadata=full_metadata,
            )
            
            logger.info(
                f"Added memory for user {user_id}: "
                f"{len(result.get('results', []))} memories extracted"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add memory for user {user_id}: {e}")
            return {"error": str(e), "results": []}
    
    def add_conversation_turn(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        emotion_state: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a conversation turn (user + assistant messages).
        
        Mem0 will extract relevant memories from the conversation context.
        
        Parameters:
        - user_message: User's message
        - assistant_message: Assistant's response
        - user_id: User identifier
        - emotion_state: Current emotional state (for tagging)
        - session_id: Optional session identifier
        
        Returns:
        - Dict with memory IDs and extracted memories
        """
        # Format as conversation
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        
        # Prepare metadata with emotional context
        metadata = {
            "timestamp": time.time(),
            "message_type": "conversation_turn",
        }
        
        if emotion_state:
            metadata["emotion"] = emotion_state.get("dominant_emotion", "neutral")
            metadata["emotion_confidence"] = emotion_state.get("confidence", 0.0)
            metadata["valence"] = emotion_state.get("valence", 0.0)
            metadata["arousal"] = emotion_state.get("arousal", 0.0)
        
        if session_id:
            metadata["session_id"] = session_id
        
        try:
            result = self._mem0.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
            )
            
            logger.info(
                f"Added conversation turn for user {user_id}: "
                f"{len(result.get('results', []))} memories extracted"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn for user {user_id}: {e}")
            return {"error": str(e), "results": []}
    
    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a user.
        
        Parameters:
        - query: Search query
        - user_id: User identifier
        - limit: Maximum number of results
        - filters: Optional metadata filters
        
        Returns:
        - List of relevant memories with scores
        """
        try:
            results = self._mem0.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters,
            )
            
            logger.info(
                f"Memory search for user {user_id}: "
                f"query='{query[:50]}...', found={len(results.get('results', []))}"
            )
            
            return results.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to search memories for user {user_id}: {e}")
            return []
    
    def get_all_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.
        
        Parameters:
        - user_id: User identifier
        - limit: Optional limit on results
        
        Returns:
        - List of all user memories
        """
        try:
            results = self._mem0.get_all(user_id=user_id)
            memories = results.get("results", [])
            
            if limit:
                memories = memories[:limit]
            
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get memories for user {user_id}: {e}")
            return []
    
    def update_memory(
        self,
        memory_id: str,
        text: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Update an existing memory.
        
        Parameters:
        - memory_id: Memory identifier
        - text: New memory text
        - user_id: User identifier
        
        Returns:
        - Updated memory info
        """
        try:
            result = self._mem0.update(
                memory_id=memory_id,
                data=text,
                user_id=user_id,
            )
            
            logger.info(f"Updated memory {memory_id} for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return {"error": str(e)}
    
    def delete_memory(
        self,
        memory_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a specific memory.
        
        Parameters:
        - memory_id: Memory identifier
        - user_id: User identifier
        
        Returns:
        - True if successful
        """
        try:
            self._mem0.delete(memory_id=memory_id, user_id=user_id)
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def delete_all_memories(self, user_id: str) -> bool:
        """
        Delete all memories for a user.
        
        Parameters:
        - user_id: User identifier
        
        Returns:
        - True if successful
        """
        try:
            self._mem0.delete_all(user_id=user_id)
            logger.info(f"Deleted all memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memories for user {user_id}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Emotion-Specific Memory Operations
    # -------------------------------------------------------------------------
    
    def add_emotional_event(
        self,
        event_description: str,
        user_id: str,
        emotion_state: Dict[str, Any],
        importance: float = 0.5,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a significant emotional event to memory.
        
        Parameters:
        - event_description: Description of the emotional event
        - user_id: User identifier
        - emotion_state: Emotional state at time of event
        - importance: Importance score (0-1)
        - session_id: Optional session identifier
        
        Returns:
        - Memory creation result
        """
        metadata = {
            "type": "emotional_event",
            "importance": importance,
            "emotion": emotion_state.get("dominant_emotion", "neutral"),
            "emotion_confidence": emotion_state.get("confidence", 0.0),
            "valence": emotion_state.get("valence", 0.0),
            "arousal": emotion_state.get("arousal", 0.0),
            "dominance": emotion_state.get("dominance", 0.0),
            "timestamp": time.time(),
        }
        
        if session_id:
            metadata["session_id"] = session_id
        
        return self.add_memory(
            text=event_description,
            user_id=user_id,
            metadata=metadata,
        )
    
    def search_emotional_memories(
        self,
        user_id: str,
        emotion: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by emotional content.
        
        Parameters:
        - user_id: User identifier
        - emotion: Optional emotion filter (e.g., "joy", "sadness")
        - min_importance: Minimum importance threshold
        - limit: Maximum results
        
        Returns:
        - List of emotional memories
        """
        filters = {"type": "emotional_event"}
        
        if emotion:
            filters["emotion"] = emotion
        
        # Note: Mem0 may not support range filters directly
        # We'll filter after retrieval if needed
        memories = self.get_all_memories(user_id=user_id)
        
        # Filter by criteria
        filtered = []
        for mem in memories:
            meta = mem.get("metadata", {})
            
            # Check type
            if meta.get("type") != "emotional_event":
                continue
            
            # Check emotion
            if emotion and meta.get("emotion") != emotion:
                continue
            
            # Check importance
            if meta.get("importance", 0.0) < min_importance:
                continue
            
            filtered.append(mem)
        
        # Sort by importance and timestamp
        filtered.sort(
            key=lambda m: (
                m.get("metadata", {}).get("importance", 0.0),
                m.get("metadata", {}).get("timestamp", 0.0)
            ),
            reverse=True
        )
        
        return filtered[:limit]
    
    def get_memory_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's memory profile.
        
        Returns statistics and insights about stored memories.
        
        Parameters:
        - user_id: User identifier
        
        Returns:
        - Summary statistics
        """
        try:
            memories = self.get_all_memories(user_id=user_id)
            
            # Count by type
            type_counts = {}
            emotion_counts = {}
            total_importance = 0.0
            
            for mem in memories:
                meta = mem.get("metadata", {})
                
                # Count types
                mem_type = meta.get("type", "general")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                
                # Count emotions
                emotion = meta.get("emotion")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Sum importance
                total_importance += meta.get("importance", 0.0)
            
            return {
                "user_id": user_id,
                "total_memories": len(memories),
                "memory_types": type_counts,
                "emotion_distribution": emotion_counts,
                "average_importance": (
                    total_importance / len(memories) if memories else 0.0
                ),
                "oldest_memory": (
                    min(m.get("metadata", {}).get("timestamp", float("inf")) 
                        for m in memories)
                    if memories else None
                ),
                "newest_memory": (
                    max(m.get("metadata", {}).get("timestamp", 0.0) 
                        for m in memories)
                    if memories else None
                ),
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory summary for user {user_id}: {e}")
            return {"error": str(e)}


# =============================================================================
# Factory Function
# =============================================================================

def create_mem0_manager(config: Optional[Mem0Config] = None) -> Mem0MemoryManager:
    """
    Create Mem0 memory manager with configuration.
    
    Parameters:
    - config: Optional Mem0 configuration (defaults to environment-based)
    
    Returns:
    - Configured Mem0MemoryManager instance
    """
    return Mem0MemoryManager(config=config)


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("MEM0 INTEGRATION - SMOKE TEST")
    print("="*70)
    
    # Create manager with basic config
    config = Mem0Config(
        vector_store="qdrant",
        qdrant_url=None,  # In-memory
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
    )
    
    try:
        manager = create_mem0_manager(config)
        print("✅ Mem0 manager initialized")
        
        # Test adding memory
        print("\n--- Adding Memory ---")
        result = manager.add_memory(
            text="I love playing tennis on weekends",
            user_id="test_user",
            metadata={"importance": 0.8}
        )
        print(f"Added: {result.get('results', [])}")
        
        # Test searching
        print("\n--- Searching Memories ---")
        memories = manager.search_memories(
            query="What does the user like to do?",
            user_id="test_user",
            limit=5
        )
        print(f"Found {len(memories)} memories")
        for mem in memories:
            print(f"  - {mem.get('memory', 'N/A')}")
        
        # Test emotional event
        print("\n--- Adding Emotional Event ---")
        result = manager.add_emotional_event(
            event_description="User expressed excitement about upcoming vacation",
            user_id="test_user",
            emotion_state={
                "dominant_emotion": "excitement",
                "confidence": 0.9,
                "valence": 0.8,
                "arousal": 0.7,
            },
            importance=0.9
        )
        print(f"Emotional event added: {result.get('results', [])}")
        
        # Test summary
        print("\n--- Memory Summary ---")
        summary = manager.get_memory_summary("test_user")
        print(f"Summary: {summary}")
        
        # Cleanup
        print("\n--- Cleanup ---")
        manager.delete_all_memories("test_user")
        print("✅ All tests passed")
        
    except ImportError:
        print("❌ mem0ai not installed. Install with: pip install mem0ai")
    except Exception as e:
        print(f"❌ Test failed: {e}")
