"""
test_mem0_integration.py
=========================
Unit tests for memory/mem0_integration.py

Coverage:
- Mem0Config: defaults, from_env
- Mem0MemoryManager: add_memory, add_conversation_turn, search_memories,
  get_all_memories, update_memory (Bug #11), delete_memory, delete_all_memories,
  add_emotional_event, search_emotional_memories, get_memory_summary
- create_mem0_manager factory
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from memory.mem0_integration import (
    Mem0Config,
    Mem0MemoryManager,
    create_mem0_manager,
)


# ===========================================================================
# Mem0Config
# ===========================================================================

class TestMem0Config:
    def test_default_values(self):
        cfg = Mem0Config()
        assert cfg.vector_store == "qdrant"
        assert cfg.embedder_provider == "openai"
        assert cfg.embedder_model == "text-embedding-3-small"
        assert cfg.enable_graph is False

    def test_from_env_defaults(self):
        cfg = Mem0Config.from_env()
        assert isinstance(cfg.vector_store, str)
        assert isinstance(cfg.embedder_provider, str)

    def test_from_env_reads_env_vars(self):
        with patch.dict(os.environ, {
            "MEM0_VECTOR_STORE": "chroma",
            "MEM0_EMBEDDER_PROVIDER": "ollama",
            "MEM0_EMBEDDER_MODEL": "nomic-embed-text",
            "MEM0_ENABLE_GRAPH": "true",
        }):
            cfg = Mem0Config.from_env()
        assert cfg.vector_store == "chroma"
        assert cfg.embedder_provider == "ollama"
        assert cfg.embedder_model == "nomic-embed-text"
        assert cfg.enable_graph is True

    def test_qdrant_url_from_env(self):
        with patch.dict(os.environ, {"QDRANT_URL": "http://qdrant:6333"}):
            cfg = Mem0Config.from_env()
        assert cfg.qdrant_url == "http://qdrant:6333"

    def test_none_qdrant_url_when_not_set(self):
        env = {k: v for k, v in os.environ.items() if k != "QDRANT_URL"}
        with patch.dict(os.environ, env, clear=True):
            cfg = Mem0Config.from_env()
        assert cfg.qdrant_url is None

    def test_collection_name_from_env(self):
        with patch.dict(os.environ, {"MEM0_COLLECTION": "custom_collection"}):
            cfg = Mem0Config.from_env()
        assert cfg.collection_name == "custom_collection"


# ===========================================================================
# Mem0MemoryManager — core CRUD
# ===========================================================================

class TestMem0MemoryManagerCRUD:
    def test_add_memory_calls_mem0_add(self, mem0_manager, mock_mem0):
        mem0_manager.add_memory("user likes hiking", user_id="user1")
        mock_mem0.add.assert_called_once()

    def test_add_memory_passes_user_id(self, mem0_manager, mock_mem0):
        mem0_manager.add_memory("likes coffee", user_id="user42")
        call_kwargs = mock_mem0.add.call_args.kwargs
        assert call_kwargs.get("user_id") == "user42"

    def test_add_memory_includes_timestamp_in_metadata(self, mem0_manager, mock_mem0):
        mem0_manager.add_memory("test memory", user_id="u1", metadata={"key": "val"})
        call_kwargs = mock_mem0.add.call_args.kwargs
        meta = call_kwargs.get("metadata", {})
        assert "timestamp" in meta

    def test_add_memory_returns_result_dict(self, mem0_manager):
        result = mem0_manager.add_memory("I enjoy music", user_id="u1")
        assert isinstance(result, dict)

    def test_add_memory_on_exception_returns_error_dict(self, mem0_manager, mock_mem0):
        mock_mem0.add.side_effect = Exception("API error")
        result = mem0_manager.add_memory("test", user_id="u1")
        assert "error" in result

    def test_get_all_memories_calls_mem0(self, mem0_manager, mock_mem0):
        mem0_manager.get_all_memories("user1")
        mock_mem0.get_all.assert_called_once_with(user_id="user1")

    def test_get_all_memories_returns_list(self, mem0_manager):
        result = mem0_manager.get_all_memories("user1")
        assert isinstance(result, list)

    def test_get_all_memories_limit_applied(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [{"id": f"m{i}"} for i in range(20)]
        }
        result = mem0_manager.get_all_memories("user1", limit=5)
        assert len(result) <= 5

    def test_get_all_on_exception_returns_empty(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.side_effect = Exception("DB error")
        result = mem0_manager.get_all_memories("user1")
        assert result == []

    def test_search_memories_calls_search(self, mem0_manager, mock_mem0):
        mem0_manager.search_memories("hiking trails", user_id="u1", limit=3)
        mock_mem0.search.assert_called_once()

    def test_search_memories_passes_user_id(self, mem0_manager, mock_mem0):
        mem0_manager.search_memories("tennis", user_id="athlete99")
        call_kwargs = mock_mem0.search.call_args.kwargs
        assert call_kwargs.get("user_id") == "athlete99"

    def test_search_memories_returns_list(self, mem0_manager):
        result = mem0_manager.search_memories("query", user_id="u1")
        assert isinstance(result, list)

    def test_search_on_exception_returns_empty(self, mem0_manager, mock_mem0):
        mock_mem0.search.side_effect = Exception("search failed")
        result = mem0_manager.search_memories("query", user_id="u1")
        assert result == []

    def test_delete_memory_calls_delete(self, mem0_manager, mock_mem0):
        mem0_manager.delete_memory("mem-123", user_id="u1")
        mock_mem0.delete.assert_called_once()

    def test_delete_memory_returns_true_on_success(self, mem0_manager):
        result = mem0_manager.delete_memory("mem-1", user_id="u1")
        assert result is True

    def test_delete_memory_returns_false_on_exception(self, mem0_manager, mock_mem0):
        mock_mem0.delete.side_effect = Exception("not found")
        result = mem0_manager.delete_memory("mem-bad", user_id="u1")
        assert result is False

    def test_delete_all_memories_returns_true(self, mem0_manager):
        result = mem0_manager.delete_all_memories("u1")
        assert result is True

    def test_delete_all_on_exception_returns_false(self, mem0_manager, mock_mem0):
        mock_mem0.delete_all.side_effect = Exception("failed")
        result = mem0_manager.delete_all_memories("u1")
        assert result is False


# ===========================================================================
# Mem0MemoryManager — update_memory (Bug #11 fix)
# ===========================================================================

class TestUpdateMemory:
    def test_update_memory_does_not_pass_user_id_to_mem0(self, mem0_manager, mock_mem0):
        """
        Bug #11 fix: Mem0's update() does not accept user_id.
        The wrapper must call self._mem0.update(memory_id=..., data=...)
        WITHOUT user_id as a kwarg.
        """
        mem0_manager.update_memory("mem-1", "updated text", user_id="u1")
        mock_mem0.update.assert_called_once()
        call_kwargs = mock_mem0.update.call_args.kwargs
        # user_id must NOT be forwarded to the underlying Mem0 API
        assert "user_id" not in call_kwargs

    def test_update_memory_passes_memory_id_and_data(self, mem0_manager, mock_mem0):
        mem0_manager.update_memory("mem-42", "new content", user_id="u1")
        call_kwargs = mock_mem0.update.call_args.kwargs
        assert call_kwargs.get("memory_id") == "mem-42"
        assert call_kwargs.get("data") == "new content"

    def test_update_memory_returns_result(self, mem0_manager):
        result = mem0_manager.update_memory("mem-1", "updated", user_id="u1")
        assert isinstance(result, dict)

    def test_update_memory_on_exception_returns_error_dict(self, mem0_manager, mock_mem0):
        mock_mem0.update.side_effect = Exception("not found")
        result = mem0_manager.update_memory("bad-id", "text", user_id="u1")
        assert "error" in result


# ===========================================================================
# Mem0MemoryManager — conversation turn
# ===========================================================================

class TestAddConversationTurn:
    def test_adds_user_and_assistant_messages(self, mem0_manager, mock_mem0):
        mem0_manager.add_conversation_turn(
            user_message="I am sad",
            assistant_message="I'm sorry to hear that.",
            user_id="u1",
        )
        mock_mem0.add.assert_called_once()
        call_kwargs = mock_mem0.add.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_emotion_state_added_to_metadata(self, mem0_manager, mock_mem0):
        mem0_manager.add_conversation_turn(
            user_message="feeling scared",
            assistant_message="You're safe.",
            user_id="u1",
            emotion_state={"dominant_emotion": "fear", "confidence": 0.8,
                           "valence": -0.6, "arousal": 0.7},
        )
        call_kwargs = mock_mem0.add.call_args.kwargs
        meta = call_kwargs.get("metadata", {})
        assert meta.get("emotion") == "fear"

    def test_session_id_added_when_provided(self, mem0_manager, mock_mem0):
        mem0_manager.add_conversation_turn(
            user_message="hi",
            assistant_message="hello",
            user_id="u1",
            session_id="sess-abc",
        )
        call_kwargs = mock_mem0.add.call_args.kwargs
        assert call_kwargs.get("metadata", {}).get("session_id") == "sess-abc"

    def test_on_exception_returns_error_dict(self, mem0_manager, mock_mem0):
        mock_mem0.add.side_effect = Exception("network error")
        result = mem0_manager.add_conversation_turn("msg", "resp", "u1")
        assert "error" in result


# ===========================================================================
# Mem0MemoryManager — emotional events
# ===========================================================================

class TestEmotionalEvents:
    def test_add_emotional_event_includes_emotion_metadata(self, mem0_manager, mock_mem0):
        mem0_manager.add_emotional_event(
            event_description="User expressed grief",
            user_id="u1",
            emotion_state={"dominant_emotion": "grief", "confidence": 0.9,
                           "valence": -0.85, "arousal": 0.10, "dominance": -0.70},
            importance=0.9,
        )
        call_kwargs = mock_mem0.add.call_args.kwargs
        meta = call_kwargs.get("metadata", {})
        assert meta.get("type") == "emotional_event"
        assert meta.get("emotion") == "grief"
        assert abs(meta.get("importance", 0) - 0.9) < 1e-6

    def test_search_emotional_memories_filters_by_type(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "felt joy", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.8, "timestamp": time.time()}},
                {"id": "m2", "memory": "general note", "metadata": {"type": "general"}},
            ]
        }
        results = mem0_manager.search_emotional_memories("u1")
        assert all(r["metadata"]["type"] == "emotional_event" for r in results)

    def test_search_emotional_memories_filters_by_emotion(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "joy", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.8, "timestamp": time.time()}},
                {"id": "m2", "memory": "sadness", "metadata": {"type": "emotional_event", "emotion": "sadness", "importance": 0.7, "timestamp": time.time()}},
            ]
        }
        results = mem0_manager.search_emotional_memories("u1", emotion="joy")
        assert all(r["metadata"]["emotion"] == "joy" for r in results)

    def test_search_emotional_memories_min_importance_filter(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "high", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.9, "timestamp": time.time()}},
                {"id": "m2", "memory": "low", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.2, "timestamp": time.time()}},
            ]
        }
        results = mem0_manager.search_emotional_memories("u1", min_importance=0.5)
        importances = [r["metadata"]["importance"] for r in results]
        assert all(i >= 0.5 for i in importances)

    def test_search_emotional_memories_limit_respected(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": f"m{i}", "memory": f"event{i}",
                 "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.8, "timestamp": time.time()}}
                for i in range(20)
            ]
        }
        results = mem0_manager.search_emotional_memories("u1", limit=5)
        assert len(results) <= 5

    def test_search_sorted_by_importance_descending(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "low", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.3, "timestamp": time.time()}},
                {"id": "m2", "memory": "high", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.9, "timestamp": time.time()}},
            ]
        }
        results = mem0_manager.search_emotional_memories("u1")
        importances = [r["metadata"]["importance"] for r in results]
        assert importances == sorted(importances, reverse=True)


# ===========================================================================
# Mem0MemoryManager — get_memory_summary
# ===========================================================================

class TestGetMemorySummary:
    def test_returns_dict(self, mem0_manager):
        result = mem0_manager.get_memory_summary("u1")
        assert isinstance(result, dict)

    def test_contains_total_memories(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {"results": []}
        result = mem0_manager.get_memory_summary("u1")
        assert "total_memories" in result

    def test_empty_memories_summary(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {"results": []}
        result = mem0_manager.get_memory_summary("u1")
        assert result["total_memories"] == 0
        assert result["average_importance"] == 0.0

    def test_emotion_distribution_counted(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.return_value = {
            "results": [
                {"id": "m1", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.8}},
                {"id": "m2", "metadata": {"type": "emotional_event", "emotion": "joy", "importance": 0.7}},
                {"id": "m3", "metadata": {"type": "general", "importance": 0.5}},
            ]
        }
        result = mem0_manager.get_memory_summary("u1")
        assert result["emotion_distribution"].get("joy", 0) == 2

    def test_on_exception_returns_error_dict(self, mem0_manager, mock_mem0):
        mock_mem0.get_all.side_effect = Exception("DB error")
        result = mem0_manager.get_memory_summary("u1")
        assert "error" in result


# ===========================================================================
# create_mem0_manager factory
# ===========================================================================

class TestCreateMem0Manager:
    def test_returns_manager_instance(self, mock_mem0):
        with patch("memory.mem0_integration.Mem0MemoryManager._initialize_mem0"):
            mgr = create_mem0_manager(Mem0Config())
        assert isinstance(mgr, Mem0MemoryManager)

    def test_uses_provided_config(self, mock_mem0):
        cfg = Mem0Config(vector_store="chroma", embedder_provider="ollama")
        with patch("memory.mem0_integration.Mem0MemoryManager._initialize_mem0"):
            mgr = create_mem0_manager(cfg)
        assert mgr.config.vector_store == "chroma"
        assert mgr.config.embedder_provider == "ollama"
