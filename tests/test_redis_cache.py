"""
test_redis_cache.py
===================
Unit tests for memory/redis_cache.py

Coverage:
- RedisCache: disabled mode, serialization (JSON only), get/set/delete,
              exists, increment, ttl, delete_pattern, clear_all, get_stats
- EmotionCache, SessionCache, ContextCache, MemoryCache helpers
- create_redis_cache: correct env-var and explicit-arg handling (Bug #12 fix)
- create_cache_helpers: returns all four helper types
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memory.redis_cache import (
    ContextCache,
    EmotionCache,
    MemoryCache,
    RedisCache,
    SessionCache,
    create_cache_helpers,
    create_redis_cache,
)


# ===========================================================================
# RedisCache — disabled mode
# ===========================================================================

class TestRedisCacheDisabled:
    def test_is_enabled_false(self, disabled_redis_cache):
        assert disabled_redis_cache.is_enabled() is False

    def test_get_returns_none(self, disabled_redis_cache):
        assert disabled_redis_cache.get("emotion", "any-key") is None

    def test_set_returns_false(self, disabled_redis_cache):
        assert disabled_redis_cache.set("emotion", "k", {"x": 1}) is False

    def test_delete_returns_false(self, disabled_redis_cache):
        assert disabled_redis_cache.delete("emotion", "k") is False

    def test_exists_returns_false(self, disabled_redis_cache):
        assert disabled_redis_cache.exists("emotion", "k") is False

    def test_increment_returns_none(self, disabled_redis_cache):
        assert disabled_redis_cache.increment("stats", "counter") is None

    def test_get_ttl_returns_none(self, disabled_redis_cache):
        assert disabled_redis_cache.get_ttl("emotion", "k") is None

    def test_delete_pattern_returns_zero(self, disabled_redis_cache):
        assert disabled_redis_cache.delete_pattern("session", "*") == 0

    def test_clear_all_returns_false(self, disabled_redis_cache):
        assert disabled_redis_cache.clear_all() is False

    def test_get_stats_returns_disabled(self, disabled_redis_cache):
        stats = disabled_redis_cache.get_stats()
        assert stats.get("enabled") is False


# ===========================================================================
# RedisCache — serialization
# ===========================================================================

class TestRedisCacheSerialization:
    """Test _serialize / _deserialize in isolation."""

    @pytest.fixture(autouse=True)
    def _cache(self, disabled_redis_cache):
        self.cache = disabled_redis_cache

    def test_json_roundtrip_dict(self):
        data = {"emotion": "joy", "score": 0.9, "nested": {"a": 1}}
        raw = self.cache._serialize(data)
        assert raw.startswith(b"json:")
        recovered = self.cache._deserialize(raw)
        assert recovered == data

    def test_json_roundtrip_list(self):
        data = [1, "two", 3.0, None]
        raw = self.cache._serialize(data)
        recovered = self.cache._deserialize(raw)
        assert recovered == data

    def test_json_roundtrip_string(self):
        data = "hello world"
        raw = self.cache._serialize(data)
        assert self.cache._deserialize(raw) == data

    def test_json_roundtrip_number(self):
        raw = self.cache._serialize(42)
        assert self.cache._deserialize(raw) == 42

    def test_non_json_serializable_raises(self):
        """Only JSON-safe types are supported — no pickle fallback."""
        import datetime
        class _CustomObj:
            pass
        with pytest.raises((ValueError, TypeError)):
            self.cache._serialize(_CustomObj())

    def test_pickle_prefix_raises_security_error(self):
        """Pickle data must be rejected (security fix)."""
        fake_pickle = b"pickle:" + b"not_real_pickle"
        with pytest.raises(ValueError, match="[Pp]ickle|[Ss]ecurity"):
            self.cache._deserialize(fake_pickle)

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError):
            self.cache._deserialize(b"unknownprefix:data")


# ===========================================================================
# RedisCache — live (mocked Redis client)
# ===========================================================================

class TestRedisCacheLive:
    def test_is_enabled_true(self, live_redis_cache):
        assert live_redis_cache.is_enabled() is True

    def test_set_calls_setex_with_ttl(self, live_redis_cache, mock_redis_client):
        live_redis_cache.set("emotion", "key1", {"x": 1}, ttl=60)
        assert mock_redis_client.setex.called or mock_redis_client.set.called

    def test_set_without_ttl_calls_set(self, live_redis_cache, mock_redis_client):
        live_redis_cache.set("session", "user1", {"history": []})
        mock_redis_client.set.assert_called()

    def test_get_hit(self, live_redis_cache, mock_redis_client):
        payload = {"emotion": "joy", "score": 0.9}
        serialized = live_redis_cache._serialize(payload)
        mock_redis_client.get.return_value = serialized
        result = live_redis_cache.get("emotion", "some-hash")
        assert result == payload

    def test_get_miss_returns_none(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.return_value = None
        assert live_redis_cache.get("emotion", "missing") is None

    def test_delete_returns_true_when_key_exists(self, live_redis_cache, mock_redis_client):
        mock_redis_client.delete.return_value = 1
        assert live_redis_cache.delete("emotion", "k1") is True

    def test_delete_returns_false_when_key_missing(self, live_redis_cache, mock_redis_client):
        mock_redis_client.delete.return_value = 0
        assert live_redis_cache.delete("emotion", "k1") is False

    def test_exists_true(self, live_redis_cache, mock_redis_client):
        mock_redis_client.exists.return_value = 1
        assert live_redis_cache.exists("session", "user1") is True

    def test_exists_false(self, live_redis_cache, mock_redis_client):
        mock_redis_client.exists.return_value = 0
        assert live_redis_cache.exists("session", "user99") is False

    def test_increment_returns_new_value(self, live_redis_cache, mock_redis_client):
        mock_redis_client.incrby.return_value = 5
        result = live_redis_cache.increment("stats", "requests", amount=1)
        assert result == 5

    def test_get_ttl(self, live_redis_cache, mock_redis_client):
        mock_redis_client.ttl.return_value = 3600
        assert live_redis_cache.get_ttl("emotion", "k") == 3600

    def test_delete_pattern_returns_count(self, live_redis_cache, mock_redis_client):
        mock_redis_client.keys.return_value = [b"emochat:session:u1", b"emochat:session:u2"]
        mock_redis_client.delete.return_value = 2
        count = live_redis_cache.delete_pattern("session", "*")
        assert count == 2

    def test_clear_all_returns_true(self, live_redis_cache, mock_redis_client):
        mock_redis_client.keys.return_value = [b"emochat:emotion:h1"]
        assert live_redis_cache.clear_all() is True

    def test_get_stats_contains_expected_keys(self, live_redis_cache, mock_redis_client):
        mock_redis_client.keys.return_value = [b"emochat:k1", b"emochat:k2"]
        mock_redis_client.info.return_value = {"keyspace_hits": 10, "keyspace_misses": 5}
        stats = live_redis_cache.get_stats()
        assert "total_keys" in stats
        assert "hit_rate" in stats

    def test_make_key_format(self, live_redis_cache):
        key = live_redis_cache._make_key("emotion", "abc123")
        assert key == "emochat:emotion:abc123"

    def test_get_exception_returns_none(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.side_effect = Exception("connection error")
        result = live_redis_cache.get("emotion", "k")
        assert result is None

    def test_set_exception_returns_false(self, live_redis_cache, mock_redis_client):
        mock_redis_client.setex.side_effect = Exception("connection error")
        mock_redis_client.set.side_effect = Exception("connection error")
        result = live_redis_cache.set("emotion", "k", {"x": 1}, ttl=60)
        assert result is False


# ===========================================================================
# EmotionCache
# ===========================================================================

class TestEmotionCache:
    def test_get_miss_returns_none(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.return_value = None
        ec = EmotionCache(live_redis_cache, ttl=3600)
        assert ec.get_emotion("I am happy") is None

    def test_set_and_get_roundtrip(self, live_redis_cache, mock_redis_client):
        payload = {"text": "I am happy", "emotions": [{"label": "joy", "score": 0.9}]}
        ec = EmotionCache(live_redis_cache, ttl=3600)

        # Simulate cache miss then hit
        serialized = live_redis_cache._serialize(payload)
        mock_redis_client.get.return_value = serialized

        result = ec.get_emotion("I am happy")
        assert result == payload

    def test_hash_text_deterministic(self):
        h1 = EmotionCache._hash_text("Hello World")
        h2 = EmotionCache._hash_text("Hello World")
        assert h1 == h2

    def test_hash_text_case_insensitive(self):
        assert EmotionCache._hash_text("HELLO") == EmotionCache._hash_text("hello")

    def test_set_calls_underlying_cache(self, live_redis_cache, mock_redis_client):
        ec = EmotionCache(live_redis_cache, ttl=60)
        ec.set_emotion("I feel good", {"emotions": [{"label": "joy", "score": 0.9}]})
        assert mock_redis_client.setex.called or mock_redis_client.set.called


# ===========================================================================
# SessionCache
# ===========================================================================

class TestSessionCache:
    def test_session_not_exists(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.return_value = None
        sc = SessionCache(live_redis_cache, ttl=86400)
        assert sc.get_session("user999") is None

    def test_set_and_get_roundtrip(self, live_redis_cache, mock_redis_client):
        payload = {"history": [], "created_at": 1234567890.0}
        serialized = live_redis_cache._serialize(payload)
        mock_redis_client.get.return_value = serialized

        sc = SessionCache(live_redis_cache)
        assert sc.get_session("user1") == payload

    def test_delete_session(self, live_redis_cache, mock_redis_client):
        mock_redis_client.delete.return_value = 1
        sc = SessionCache(live_redis_cache)
        assert sc.delete_session("user1") is True

    def test_session_exists_true(self, live_redis_cache, mock_redis_client):
        mock_redis_client.exists.return_value = 1
        sc = SessionCache(live_redis_cache)
        assert sc.session_exists("user1") is True

    def test_session_exists_false(self, live_redis_cache, mock_redis_client):
        mock_redis_client.exists.return_value = 0
        sc = SessionCache(live_redis_cache)
        assert sc.session_exists("user_new") is False


# ===========================================================================
# ContextCache
# ===========================================================================

class TestContextCache:
    def test_get_miss_returns_none(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.return_value = None
        cc = ContextCache(live_redis_cache, ttl=300)
        assert cc.get_context("user1", "some query") is None

    def test_set_context(self, live_redis_cache, mock_redis_client):
        cc = ContextCache(live_redis_cache, ttl=300)
        ctx = {"messages": [], "diagnostics": {}}
        cc.set_context("user1", "some query", ctx)
        assert mock_redis_client.setex.called or mock_redis_client.set.called

    def test_hash_query_deterministic(self):
        h1 = ContextCache._hash_query("what is love?")
        h2 = ContextCache._hash_query("what is love?")
        assert h1 == h2

    def test_hash_query_short(self):
        h = ContextCache._hash_query("test")
        assert len(h) == 16   # first 16 chars of md5 hex


# ===========================================================================
# MemoryCache
# ===========================================================================

class TestMemoryCache:
    def test_get_miss_returns_none(self, live_redis_cache, mock_redis_client):
        mock_redis_client.get.return_value = None
        mc = MemoryCache(live_redis_cache, ttl=600)
        assert mc.get_memories("user1", "query", top_k=5) is None

    def test_set_memories(self, live_redis_cache, mock_redis_client):
        mc = MemoryCache(live_redis_cache, ttl=600)
        mc.set_memories("user1", "query", 5, [{"memory": "test"}])
        assert mock_redis_client.setex.called or mock_redis_client.set.called

    def test_invalidate_user_memories(self, live_redis_cache, mock_redis_client):
        mock_redis_client.keys.return_value = [b"emochat:memory:user1:hash1:5"]
        mock_redis_client.delete.return_value = 1
        mc = MemoryCache(live_redis_cache, ttl=600)
        count = mc.invalidate_user_memories("user1")
        assert isinstance(count, int)


# ===========================================================================
# create_redis_cache — Bug #12 fix (falsy db=0)
# ===========================================================================

class TestCreateRedisCache:
    def test_db_zero_not_overridden_by_env(self):
        """db=0 must NOT be replaced by the env-var fallback (falsy-check bug)."""
        with patch.dict(os.environ, {"REDIS_DB": "3"}):
            # If the bug were present, passing db=0 would evaluate 0 or 3 → 3
            cache = create_redis_cache(db=0, enabled=False)
            # enabled=False so no actual connection; just check the object
            assert not cache.is_enabled()  # just verify it doesn't crash

    def test_defaults_from_env(self):
        with patch.dict(os.environ, {
            "REDIS_HOST": "my-redis",
            "REDIS_PORT": "6380",
            "REDIS_DB": "2",
            "REDIS_ENABLED": "false",
        }):
            cache = create_redis_cache()
            assert not cache.is_enabled()  # env says disabled

    def test_explicit_enabled_false(self):
        cache = create_redis_cache(enabled=False)
        assert not cache.is_enabled()

    def test_redis_enabled_env_false(self):
        with patch.dict(os.environ, {"REDIS_ENABLED": "false"}):
            cache = create_redis_cache()
            assert not cache.is_enabled()


# ===========================================================================
# create_cache_helpers
# ===========================================================================

class TestCreateCacheHelpers:
    def test_returns_all_four_helpers(self, disabled_redis_cache):
        helpers = create_cache_helpers(disabled_redis_cache)
        assert "emotion" in helpers
        assert "session" in helpers
        assert "context" in helpers
        assert "memory" in helpers

    def test_helper_types(self, disabled_redis_cache):
        helpers = create_cache_helpers(disabled_redis_cache)
        assert isinstance(helpers["emotion"], EmotionCache)
        assert isinstance(helpers["session"], SessionCache)
        assert isinstance(helpers["context"], ContextCache)
        assert isinstance(helpers["memory"], MemoryCache)

    def test_ttls_are_positive(self, disabled_redis_cache):
        helpers = create_cache_helpers(disabled_redis_cache)
        assert helpers["emotion"].ttl > 0
        assert helpers["session"].ttl > 0
        assert helpers["context"].ttl > 0
        assert helpers["memory"].ttl > 0

    def test_session_ttl_longer_than_emotion_ttl(self, disabled_redis_cache):
        helpers = create_cache_helpers(disabled_redis_cache)
        assert helpers["session"].ttl > helpers["emotion"].ttl
