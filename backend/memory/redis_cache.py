"""
redis_cache.py
==============
Redis caching layer for emotion chatbot.

Caches:
1. Emotion analysis results (by message hash)
2. Session state (emotion manager, context manager)
3. Memory retrieval results
4. LLM context assembly results

Design goals:
- Transparent caching (plug-and-play)
- Configurable TTL per cache type
- JSON serialization for complex objects
- Thread-safe operations
- Graceful fallback if Redis unavailable
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional, Dict, List, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache client with automatic serialization and graceful fallback.
    
    Features:
    - JSON serialization for simple types
    - Pickle fallback for complex objects
    - Automatic key prefixing by cache type
    - TTL support
    - Connection pooling
    - Thread-safe
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 5.0,
        connection_pool_size: int = 10,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._client = None
        self._lock = threading.RLock()
        
        if not enabled:
            logger.warning("Redis cache disabled - running without cache")
            return
        
        try:
            import redis
            
            # Create connection pool
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                max_connections=connection_pool_size,
                decode_responses=False,  # We'll handle encoding/decoding
            )
            
            self._client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self._client.ping()
            logger.info(
                f"Redis cache connected: {host}:{port} db={db} "
                f"(pool_size={connection_pool_size})"
            )
            
        except ImportError:
            logger.error("redis package not installed - cache disabled")
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e} - cache disabled")
            self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if cache is enabled and connected."""
        return self._enabled and self._client is not None
    
    def _make_key(self, cache_type: str, key: str) -> str:
        """Generate prefixed cache key."""
        return f"emochat:{cache_type}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value to JSON only. NO PICKLE FALLBACK.
        
        If serialization fails, it raises an exception - this is intentional.
        Pickle fallback creates security vulnerabilities (RCE) and data silos.
        """
        try:
            json_str = json.dumps(value, default=str)
            return b"json:" + json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot serialize value to JSON. Only JSON-serializable types are supported. "
                f"Do not cache complex objects like model instances or class objects. "
                f"Original error: {e}"
            )
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize JSON data only. NO PICKLE SUPPORT.
        
        Legacy pickle data is rejected for security reasons.
        """
        if data.startswith(b"json:"):
            return json.loads(data[5:].decode("utf-8"))
        elif data.startswith(b"pickle:"):
            raise ValueError(
                "Pickle-serialized data detected in cache. This is a security risk. "
                "Clear the cache and restart with JSON-only serialization."
            )
        else:
            # Unknown format
            raise ValueError(
                "Unknown cache data format. Expected 'json:' prefix. "
                "Cache may be corrupted. Clear and restart."
            )
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_enabled():
            return None
        
        try:
            full_key = self._make_key(cache_type, key)
            data = self._client.get(full_key)
            
            if data is None:
                return None
            
            value = self._deserialize(data)
            logger.debug(f"Cache HIT: {cache_type}:{key}")
            return value
            
        except Exception as e:
            logger.warning(f"Cache GET error for {cache_type}:{key}: {e}")
            return None
    
    def set(
        self,
        cache_type: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Parameters:
        - cache_type: Type of cache (e.g., "emotion", "session", "memory")
        - key: Cache key
        - value: Value to cache
        - ttl: Time to live in seconds (None = no expiration)
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
        
        try:
            full_key = self._make_key(cache_type, key)
            data = self._serialize(value)
            
            if ttl is not None:
                self._client.setex(full_key, ttl, data)
            else:
                self._client.set(full_key, data)
            
            logger.debug(f"Cache SET: {cache_type}:{key} (ttl={ttl})")
            return True
            
        except Exception as e:
            logger.warning(f"Cache SET error for {cache_type}:{key}: {e}")
            return False
    
    def delete(self, cache_type: str, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_enabled():
            return False
        
        try:
            full_key = self._make_key(cache_type, key)
            result = self._client.delete(full_key)
            logger.debug(f"Cache DELETE: {cache_type}:{key}")
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache DELETE error for {cache_type}:{key}: {e}")
            return False
    
    def delete_pattern(self, cache_type: str, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Example: delete_pattern("session", "user123:*") deletes all user123 sessions
        """
        if not self.is_enabled():
            return 0
        
        try:
            full_pattern = self._make_key(cache_type, pattern)
            keys = self._client.keys(full_pattern)
            
            if not keys:
                return 0
            
            count = self._client.delete(*keys)
            logger.debug(f"Cache DELETE pattern: {cache_type}:{pattern} ({count} keys)")
            return count
            
        except Exception as e:
            logger.warning(f"Cache DELETE pattern error for {cache_type}:{pattern}: {e}")
            return 0
    
    def exists(self, cache_type: str, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_enabled():
            return False
        
        try:
            full_key = self._make_key(cache_type, key)
            return bool(self._client.exists(full_key))
        except Exception as e:
            logger.warning(f"Cache EXISTS error for {cache_type}:{key}: {e}")
            return False
    
    def increment(self, cache_type: str, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        if not self.is_enabled():
            return None
        
        try:
            full_key = self._make_key(cache_type, key)
            return self._client.incrby(full_key, amount)
        except Exception as e:
            logger.warning(f"Cache INCREMENT error for {cache_type}:{key}: {e}")
            return None
    
    def get_ttl(self, cache_type: str, key: str) -> Optional[int]:
        """Get remaining TTL for key (seconds). -1 = no expiry, -2 = doesn't exist."""
        if not self.is_enabled():
            return None
        
        try:
            full_key = self._make_key(cache_type, key)
            return self._client.ttl(full_key)
        except Exception as e:
            logger.warning(f"Cache TTL error for {cache_type}:{key}: {e}")
            return None
    
    def clear_all(self) -> bool:
        """Clear ALL emochat cache keys (use with caution!)."""
        if not self.is_enabled():
            return False
        
        try:
            keys = self._client.keys("emochat:*")
            if keys:
                self._client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys")
            return True
        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_enabled():
            return {"enabled": False}
        
        try:
            info = self._client.info("stats")
            keys = self._client.keys("emochat:*")
            
            return {
                "enabled": True,
                "total_keys": len(keys),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
                    if info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0) > 0
                    else 0.0
                ),
            }
        except Exception as e:
            logger.warning(f"Cache STATS error: {e}")
            return {"enabled": True, "error": str(e)}


# =============================================================================
# Cache Decorators for Common Patterns
# =============================================================================

def cache_result(
    cache: RedisCache,
    cache_type: str,
    key_fn: Callable[..., str],
    ttl: Optional[int] = None,
):
    """
    Decorator to cache function results.
    
    Example:
        @cache_result(cache, "emotion", lambda text: hashlib.md5(text.encode()).hexdigest(), ttl=300)
        def predict_emotion(text: str):
            # expensive operation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = key_fn(*args, **kwargs)
            
            # Try to get from cache
            cached = cache.get(cache_type, cache_key)
            if cached is not None:
                return cached
            
            # Cache miss - call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_type, cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# =============================================================================
# Specialized Cache Helpers
# =============================================================================

class EmotionCache:
    """Helper for caching emotion analysis results."""
    
    def __init__(self, redis_cache: RedisCache, ttl: int = 3600):
        self.cache = redis_cache
        self.ttl = ttl
        self.cache_type = "emotion"
    
    def get_emotion(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached emotion analysis for text."""
        key = self._hash_text(text)
        return self.cache.get(self.cache_type, key)
    
    def set_emotion(self, text: str, result: Dict[str, Any]) -> bool:
        """Cache emotion analysis result."""
        key = self._hash_text(text)
        return self.cache.set(self.cache_type, key, result, ttl=self.ttl)
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash for text (cache key)."""
        return hashlib.md5(text.strip().lower().encode()).hexdigest()


class SessionCache:
    """Helper for caching session state."""
    
    def __init__(self, redis_cache: RedisCache, ttl: int = 86400):
        self.cache = redis_cache
        self.ttl = ttl
        self.cache_type = "session"
    
    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session state."""
        return self.cache.get(self.cache_type, user_id)
    
    def set_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache session state."""
        return self.cache.set(self.cache_type, user_id, session_data, ttl=self.ttl)
    
    def delete_session(self, user_id: str) -> bool:
        """Delete cached session."""
        return self.cache.delete(self.cache_type, user_id)
    
    def session_exists(self, user_id: str) -> bool:
        """Check if session exists in cache."""
        return self.cache.exists(self.cache_type, user_id)


class ContextCache:
    """Helper for caching LLM context assembly results."""
    
    def __init__(self, redis_cache: RedisCache, ttl: int = 300):
        self.cache = redis_cache
        self.ttl = ttl
        self.cache_type = "context"
    
    def get_context(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """Get cached context for user and query."""
        key = f"{user_id}:{self._hash_query(query)}"
        return self.cache.get(self.cache_type, key)
    
    def set_context(self, user_id: str, query: str, context: Dict[str, Any]) -> bool:
        """Cache context assembly result."""
        key = f"{user_id}:{self._hash_query(query)}"
        return self.cache.set(self.cache_type, key, context, ttl=self.ttl)
    
    @staticmethod
    def _hash_query(query: str) -> str:
        """Generate hash for query."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]


class MemoryCache:
    """Helper for caching memory retrieval results."""
    
    def __init__(self, redis_cache: RedisCache, ttl: int = 600):
        self.cache = redis_cache
        self.ttl = ttl
        self.cache_type = "memory"
    
    def get_memories(self, user_id: str, query: str, top_k: int) -> Optional[List[Any]]:
        """Get cached memory retrieval result."""
        key = f"{user_id}:{self._hash_query(query)}:{top_k}"
        return self.cache.get(self.cache_type, key)
    
    def set_memories(
        self,
        user_id: str,
        query: str,
        top_k: int,
        memories: List[Any],
    ) -> bool:
        """Cache memory retrieval result."""
        key = f"{user_id}:{self._hash_query(query)}:{top_k}"
        return self.cache.set(self.cache_type, key, memories, ttl=self.ttl)
    
    def invalidate_user_memories(self, user_id: str) -> int:
        """Invalidate all cached memories for a user."""
        return self.cache.delete_pattern(self.cache_type, f"{user_id}:*")
    
    @staticmethod
    def _hash_query(query: str) -> str:
        """Generate hash for query."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]


# =============================================================================
# Factory Functions
# =============================================================================

def create_redis_cache(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
    enabled: bool = True,
) -> RedisCache:
    """
    Create Redis cache client with environment variable support.
    
    Priority: function args > environment variables > defaults
    """
    import os
    
    return RedisCache(
        host=host or os.getenv("REDIS_HOST", "localhost"),
        port=port or int(os.getenv("REDIS_PORT", "6379")),
        db=db or int(os.getenv("REDIS_DB", "0")),
        password=password or os.getenv("REDIS_PASSWORD"),
        enabled=enabled and os.getenv("REDIS_ENABLED", "true").lower() == "true",
    )


def create_cache_helpers(redis_cache: RedisCache) -> Dict[str, Any]:
    """Create all cache helper instances."""
    return {
        "emotion": EmotionCache(redis_cache, ttl=3600),         # 1 hour
        "session": SessionCache(redis_cache, ttl=86400),        # 24 hours
        "context": ContextCache(redis_cache, ttl=300),          # 5 minutes
        "memory": MemoryCache(redis_cache, ttl=600),            # 10 minutes
    }


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("REDIS CACHE - SMOKE TEST")
    print("="*70)
    
    # Create cache
    cache = create_redis_cache()
    
    if not cache.is_enabled():
        print("❌ Redis not available - skipping tests")
        exit(1)
    
    print("✅ Redis connected")
    
    # Test basic operations
    print("\n--- Basic Operations ---")
    cache.set("test", "key1", {"data": "value1"}, ttl=60)
    result = cache.get("test", "key1")
    print(f"Set and get: {result}")
    assert result == {"data": "value1"}
    
    # Test emotion cache
    print("\n--- Emotion Cache ---")
    emotion_cache = EmotionCache(cache, ttl=60)
    emotion_cache.set_emotion("I am happy", {"emotion": "joy", "score": 0.9})
    result = emotion_cache.get_emotion("I am happy")
    print(f"Emotion cache: {result}")
    assert result["emotion"] == "joy"
    
    # Test session cache
    print("\n--- Session Cache ---")
    session_cache = SessionCache(cache, ttl=60)
    session_cache.set_session("user123", {"messages": 5, "created": time.time()})
    result = session_cache.get_session("user123")
    print(f"Session cache: {result}")
    assert result["messages"] == 5
    
    # Test stats
    print("\n--- Cache Stats ---")
    stats = cache.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    cache.clear_all()
    print("✅ All tests passed")
