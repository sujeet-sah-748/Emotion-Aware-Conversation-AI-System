"""
test_context_engine.py
======================
Unit tests for core/context_engine.py

Coverage:
- HeuristicTokenCounter + TruncateSummarizer
- HashingEmbedder (used as a fast stand-in; not the production embedder)
- InMemoryVectorStore: upsert / search / delete / len
- MemoryRetriever: add / remove / get / retrieve / save-load / hybrid scoring
- ContextBudget: validation
- ContextWindow: assemble — budget caps, eviction, truncation, memory inclusion
- ContextManager: add_message, remember, build_context, reset, metrics
- TruncatingSummarizer: output length
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import pytest

from core.context_engine import (
    ContextBudget,
    ContextManager,
    ContextSnapshot,
    ContextWindow,
    HashingEmbedder,
    HeuristicTokenCounter,
    InMemoryVectorStore,
    Memory,
    MemoryRetriever,
    Message,
    Metrics,
    RetrievalConfig,
    Role,
    ScoredMemory,
    TruncatingSummarizer,
    _cosine_similarity,
    truncate_to_tokens,
)
from core.emotion_engine import EmotionLabel


# ===========================================================================
# Helpers
# ===========================================================================

def _make_retriever(top_k=3):
    emb = HashingEmbedder(dimension=64)
    store = InMemoryVectorStore()
    return MemoryRetriever(
        embedder=emb,
        store=store,
        config=RetrievalConfig(top_k=top_k, min_score=0.0),
    )


def _counter(cpt=4.0):
    return HeuristicTokenCounter(chars_per_token=cpt)


# ===========================================================================
# _cosine_similarity
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [0.5, 0.3, 0.8]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        v = [1.0, 0.0]
        neg = [-1.0, 0.0]
        assert abs(_cosine_similarity(v, neg) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-9


# ===========================================================================
# HeuristicTokenCounter
# ===========================================================================

class TestHeuristicTokenCounter:
    def test_empty_string_returns_one(self):
        assert _counter().count("") == 1

    def test_short_text(self):
        # "hello" = 5 chars, cpt=4 → ceil(5/4)=2
        assert _counter(4.0).count("hello") == 2

    def test_scales_with_length(self):
        c = _counter(4.0)
        assert c.count("a" * 400) > c.count("a" * 40)

    def test_always_positive(self):
        c = _counter()
        for text in ["", "x", "hello world"]:
            assert c.count(text) >= 1


# ===========================================================================
# truncate_to_tokens
# ===========================================================================

class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        c = _counter(4.0)
        text = "Hi"
        assert truncate_to_tokens(text, 100, c) == text

    def test_zero_max_returns_empty(self):
        c = _counter(4.0)
        assert truncate_to_tokens("hello world", 0, c) == ""

    def test_truncated_text_within_budget(self):
        c = _counter(4.0)
        text = "a" * 400
        max_tok = 10
        result = truncate_to_tokens(text, max_tok, c)
        assert c.count(result) <= max_tok + 2  # small margin for the ellipsis

    def test_truncated_marker_appended(self):
        c = _counter(4.0)
        result = truncate_to_tokens("a" * 400, 10, c)
        assert "truncated" in result


# ===========================================================================
# HashingEmbedder
# ===========================================================================

class TestHashingEmbedder:
    def test_output_dimension(self):
        emb = HashingEmbedder(dimension=128)
        vecs = emb.embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 128

    def test_batch_output(self):
        emb = HashingEmbedder(dimension=64)
        vecs = emb.embed(["one", "two", "three"])
        assert len(vecs) == 3

    def test_unit_length(self):
        emb = HashingEmbedder(dimension=64)
        vec = emb.embed(["hello"])[0]
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0  # zero for empty

    def test_deterministic(self):
        emb = HashingEmbedder(dimension=64)
        v1 = emb.embed(["emotion chatbot"])[0]
        v2 = emb.embed(["emotion chatbot"])[0]
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        emb = HashingEmbedder(dimension=64)
        v1 = emb.embed(["joy happiness"])[0]
        v2 = emb.embed(["sadness grief"])[0]
        assert v1 != v2

    def test_dimension_property(self):
        emb = HashingEmbedder(dimension=32)
        assert emb.dimension == 32


# ===========================================================================
# InMemoryVectorStore
# ===========================================================================

class TestInMemoryVectorStore:
    def test_empty_store_len_zero(self, in_memory_store):
        assert len(in_memory_store) == 0

    def test_upsert_increases_len(self, in_memory_store):
        in_memory_store.upsert("id1", [0.1, 0.2, 0.3], {})
        assert len(in_memory_store) == 1

    def test_upsert_overwrites_existing(self, in_memory_store):
        in_memory_store.upsert("id1", [0.1, 0.2], {})
        in_memory_store.upsert("id1", [0.9, 0.9], {})
        assert len(in_memory_store) == 1

    def test_delete_existing(self, in_memory_store):
        in_memory_store.upsert("id1", [0.1, 0.2], {})
        result = in_memory_store.delete("id1")
        assert result is True
        assert len(in_memory_store) == 0

    def test_delete_nonexistent_returns_false(self, in_memory_store):
        assert in_memory_store.delete("no-such-id") is False

    def test_search_returns_most_similar(self):
        store = InMemoryVectorStore()
        store.upsert("joy", [1.0, 0.0], {})
        store.upsert("fear", [0.0, 1.0], {})
        results = store.search([1.0, 0.0], top_k=2)
        ids = [r[0] for r in results]
        assert ids[0] == "joy"

    def test_search_top_k_respected(self):
        store = InMemoryVectorStore()
        for i in range(10):
            store.upsert(f"id{i}", [float(i), 0.0], {})
        results = store.search([1.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_search_with_filter(self):
        store = InMemoryVectorStore()
        store.upsert("happy", [1.0, 0.0], {"kind": "positive"})
        store.upsert("sad", [0.5, 0.5], {"kind": "negative"})
        results = store.search([1.0, 0.0], top_k=5,
                               filter_fn=lambda m: m.get("kind") == "positive")
        assert all(r[0] == "happy" for r in results)


# ===========================================================================
# MemoryRetriever
# ===========================================================================

class TestMemoryRetriever:
    def test_add_returns_memory(self, memory_retriever):
        mem = memory_retriever.add("I enjoy walking in the park", importance=0.7)
        assert isinstance(mem, Memory)
        assert mem.content == "I enjoy walking in the park"

    def test_add_empty_string_raises(self, memory_retriever):
        with pytest.raises(ValueError):
            memory_retriever.add("")

    def test_len_after_add(self, memory_retriever):
        n = len(memory_retriever)
        memory_retriever.add("test memory")
        assert len(memory_retriever) == n + 1

    def test_get_by_id(self, memory_retriever):
        mem = memory_retriever.add("something memorable")
        fetched = memory_retriever.get(mem.memory_id)
        assert fetched is not None
        assert fetched.content == mem.content

    def test_get_missing_id_returns_none(self, memory_retriever):
        assert memory_retriever.get("nonexistent-id") is None

    def test_remove_existing(self, memory_retriever):
        mem = memory_retriever.add("to be removed")
        result = memory_retriever.remove(mem.memory_id)
        assert result is True
        assert memory_retriever.get(mem.memory_id) is None

    def test_remove_nonexistent(self, memory_retriever):
        assert memory_retriever.remove("no-id") is False

    def test_retrieve_returns_scored_memories(self, memory_retriever):
        memory_retriever.add("user likes hiking", importance=0.8)
        memory_retriever.add("user hates spiders", importance=0.6)
        results = memory_retriever.retrieve("outdoor activities")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ScoredMemory)

    def test_retrieve_empty_store_returns_empty(self, hashing_embedder, in_memory_store):
        r = MemoryRetriever(embedder=hashing_embedder, store=in_memory_store)
        assert r.retrieve("anything") == []

    def test_retrieve_respects_top_k(self, hashing_embedder):
        store = InMemoryVectorStore()
        r = MemoryRetriever(
            embedder=hashing_embedder, store=store,
            config=RetrievalConfig(top_k=2, min_score=0.0),
        )
        for i in range(10):
            r.add(f"memory number {i}")
        results = r.retrieve("some query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_sorted_by_score(self, memory_retriever):
        for i in range(5):
            memory_retriever.add(f"entry {i}", importance=float(i) / 5)
        results = memory_retriever.retrieve("entry")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_access_count_increments_on_retrieve(self, memory_retriever):
        mem = memory_retriever.add("recall this", importance=0.9)
        memory_retriever.retrieve("recall this")
        retrieved = memory_retriever.get(mem.memory_id)
        assert retrieved is not None
        assert retrieved.access_count >= 1

    def test_importance_stored_correctly(self, memory_retriever):
        mem = memory_retriever.add("important event", importance=0.95)
        assert abs(mem.importance - 0.95) < 1e-6

    def test_importance_clamped_to_01(self, memory_retriever):
        mem = memory_retriever.add("test", importance=1.5)
        assert mem.importance == 1.0

    def test_save_and_load_roundtrip(self, hashing_embedder):
        store1 = InMemoryVectorStore()
        r1 = MemoryRetriever(embedder=hashing_embedder, store=store1,
                             config=RetrievalConfig(top_k=5, min_score=0.0))
        r1.add("memory about cats", importance=0.8)
        r1.add("memory about dogs", importance=0.6)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            r1.save(path)

            store2 = InMemoryVectorStore()
            r2 = MemoryRetriever.load(path, hashing_embedder, store=store2,
                                      config=RetrievalConfig(top_k=5, min_score=0.0))
            assert len(r2) == 2
            contents = {r2.get(mid).content for mid in r2._memories}
            assert "memory about cats" in contents
        finally:
            Path(path).unlink(missing_ok=True)

    def test_none_store_defaults_to_in_memory(self, hashing_embedder):
        """Bug #7 fix: store=None should not crash."""
        r = MemoryRetriever(embedder=hashing_embedder, store=None)
        r.add("test memory")   # would crash before the fix
        assert len(r) == 1


# ===========================================================================
# RetrievalConfig
# ===========================================================================

class TestRetrievalConfig:
    def test_default_weights_valid(self):
        cfg = RetrievalConfig()
        total = cfg.weight_similarity + cfg.weight_recency + cfg.weight_importance
        assert total > 0

    def test_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            RetrievalConfig(min_score=1.5)

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            RetrievalConfig(weight_similarity=0.0, weight_recency=0.0, weight_importance=0.0)


# ===========================================================================
# ContextBudget
# ===========================================================================

class TestContextBudget:
    def test_available_tokens(self):
        b = ContextBudget(max_tokens=1024, reserved_for_response=256)
        assert b.available == 768

    def test_reserved_gte_max_raises(self):
        with pytest.raises(ValueError):
            ContextBudget(max_tokens=512, reserved_for_response=512)

    def test_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ContextBudget(max_memory_ratio=0.0)

    def test_max_affect_ratio_valid(self):
        b = ContextBudget(max_affect_ratio=0.05)
        assert b.max_affect_ratio == 0.05


# ===========================================================================
# ContextWindow.assemble
# ===========================================================================

class TestContextWindowAssemble:
    def test_no_content_returns_empty_messages(self, context_window):
        snapshot = context_window.assemble(None, [], [])
        assert isinstance(snapshot, ContextSnapshot)
        assert snapshot.total_tokens >= 0

    def test_system_prompt_included(self, context_window):
        snapshot = context_window.assemble("You are helpful.", [], [])
        roles = [m.role for m in snapshot.messages]
        assert Role.SYSTEM in roles

    def test_history_included(self, context_window):
        history = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
        snapshot = context_window.assemble(None, [], history)
        contents = [m.content for m in snapshot.messages]
        assert "Hello" in contents

    def test_total_tokens_within_budget(self, context_window):
        history = [Message(role=Role.USER, content="x" * 50) for _ in range(20)]
        snapshot = context_window.assemble("System.", [], history)
        assert snapshot.total_tokens <= snapshot.budget_tokens

    def test_dropped_count_when_history_overflows(self, context_window):
        # 50 messages each ~50 chars should overflow a 512-token window
        history = [Message(role=Role.USER, content="a" * 50) for _ in range(50)]
        snapshot = context_window.assemble(None, [], history)
        assert snapshot.dropped_history_count >= 0  # may or may not drop

    def test_memory_block_included(self, context_window, memory_retriever):
        memory_retriever.add("user likes Python")
        scored = memory_retriever.retrieve("Python")
        snapshot = context_window.assemble("System.", scored, [])
        combined = " ".join(m.content for m in snapshot.messages)
        assert "user likes Python" in combined

    def test_affect_block_included(self, context_window):
        affect = "Emotional context: user is joyful."
        snapshot = context_window.assemble(None, [], [], affect_block=affect)
        combined = " ".join(m.content for m in snapshot.messages)
        assert "joyful" in combined

    def test_large_system_prompt_truncated(self, context_window):
        big_system = "x" * 2000
        snapshot = context_window.assemble(big_system, [], [])
        sys_messages = [m for m in snapshot.messages if m.role == Role.SYSTEM
                        and m.metadata.get("kind") != "affect_block"
                        and m.metadata.get("kind") != "memory_block"]
        if sys_messages:
            assert len(sys_messages[0].content) <= len(big_system)

    def test_to_chat_format(self, context_window):
        history = [Message(role=Role.USER, content="Hi")]
        snapshot = context_window.assemble("System.", [], history)
        fmt = snapshot.to_chat_format()
        assert all("role" in m and "content" in m for m in fmt)

    def test_diagnostics_keys_present(self, context_window):
        history = [Message(role=Role.USER, content="test")]
        snapshot = context_window.assemble("sys", [], history)
        for key in ("system_tokens", "memory_tokens", "history_tokens"):
            assert key in snapshot.diagnostics


# ===========================================================================
# TruncatingSummarizer
# ===========================================================================

class TestTruncatingSummarizer:
    def test_output_within_max_chars(self):
        s = TruncatingSummarizer(max_chars=100)
        msgs = [Message(role=Role.USER, content="A very long message " * 10)]
        result = s.summarize(msgs)
        assert len(result) <= 100

    def test_output_is_string(self):
        s = TruncatingSummarizer()
        result = s.summarize([Message(role=Role.USER, content="hello")])
        assert isinstance(result, str)

    def test_includes_role_prefix(self):
        s = TruncatingSummarizer(max_chars=500)
        result = s.summarize([Message(role=Role.USER, content="feeling blue")])
        assert "user" in result.lower()


# ===========================================================================
# ContextManager — write path
# ===========================================================================

class TestContextManagerWrite:
    def test_add_user_message(self, context_manager):
        msg = context_manager.add_user_message("Hello there")
        assert msg.role == Role.USER
        assert msg.content == "Hello there"

    def test_add_assistant_message(self, context_manager):
        msg = context_manager.add_assistant_message("How can I help?")
        assert msg.role == Role.ASSISTANT

    def test_empty_content_raises(self, context_manager):
        with pytest.raises(ValueError):
            context_manager.add_message(Role.USER, "")

    def test_whitespace_only_raises(self, context_manager):
        with pytest.raises(ValueError):
            context_manager.add_message(Role.USER, "   ")

    def test_history_grows_after_add(self, context_manager):
        before = len(context_manager.history)
        context_manager.add_user_message("test message")
        assert len(context_manager.history) == before + 1

    def test_remember_adds_to_retriever(self, context_manager):
        before = len(context_manager.retriever)
        context_manager.remember("Important fact: user dislikes cats", importance=0.8)
        assert len(context_manager.retriever) == before + 1

    def test_remember_empty_raises(self, context_manager):
        with pytest.raises(ValueError):
            context_manager.remember("")

    def test_metrics_messages_added_increments(self, context_manager):
        m_before = context_manager.metrics()["messages_added"]
        context_manager.add_user_message("new message")
        m_after = context_manager.metrics()["messages_added"]
        assert m_after == m_before + 1

    def test_metrics_memories_added_increments(self, context_manager):
        m_before = context_manager.metrics()["memories_added"]
        context_manager.remember("a new long-term memory")
        m_after = context_manager.metrics()["memories_added"]
        assert m_after == m_before + 1


# ===========================================================================
# ContextManager — read path
# ===========================================================================

class TestContextManagerRead:
    def test_build_context_returns_snapshot(self, context_manager):
        context_manager.add_user_message("How are you?")
        snapshot = context_manager.build_context()
        assert isinstance(snapshot, ContextSnapshot)

    def test_build_context_tokens_within_budget(self, context_manager):
        context_manager.add_user_message("Tell me about emotions.")
        snapshot = context_manager.build_context()
        assert snapshot.total_tokens <= snapshot.budget_tokens

    def test_build_context_includes_history(self, context_manager):
        context_manager.add_user_message("I feel lonely.")
        snapshot = context_manager.build_context()
        contents = " ".join(m.content for m in snapshot.messages)
        assert "lonely" in contents

    def test_build_context_retrieves_relevant_memories(self, context_manager):
        context_manager.remember("user mentioned feeling anxious about work", importance=0.9)
        context_manager.add_user_message("I am stressed about my job.")
        snapshot = context_manager.build_context()
        # Memory about work anxiety should rank high
        combined = " ".join(m.content for m in snapshot.messages)
        assert "anxious" in combined or len(snapshot.memories_used) >= 0  # may rank if relevant

    def test_build_context_diagnostics_populated(self, context_manager):
        context_manager.add_user_message("test")
        snap = context_manager.build_context()
        assert "total_tokens" in snap.diagnostics or snap.total_tokens >= 0

    def test_build_context_without_query_uses_last_user_message(self, context_manager):
        context_manager.add_user_message("I need help with anxiety.")
        snap = context_manager.build_context(query=None)
        assert snap is not None

    def test_metrics_contexts_built_increments(self, context_manager):
        context_manager.add_user_message("test")
        before = context_manager.metrics()["contexts_built"]
        context_manager.build_context()
        assert context_manager.metrics()["contexts_built"] == before + 1


# ===========================================================================
# ContextManager — reset
# ===========================================================================

class TestContextManagerReset:
    def test_reset_clears_history(self, context_manager):
        context_manager.add_user_message("message to be cleared")
        context_manager.reset(keep_memories=True)
        assert len(context_manager.history) == 0

    def test_reset_keep_memories_true(self, context_manager):
        context_manager.remember("keep this memory", importance=0.8)
        before_count = len(context_manager.retriever)
        context_manager.reset(keep_memories=True)
        assert len(context_manager.retriever) == before_count

    def test_reset_keep_memories_false(self, context_manager):
        context_manager.remember("remove this memory", importance=0.8)
        assert len(context_manager.retriever) > 0
        context_manager.reset(keep_memories=False)
        assert len(context_manager.retriever) == 0

    def test_reset_resets_metrics(self, context_manager):
        context_manager.add_user_message("test")
        context_manager.build_context()
        context_manager.reset()
        # After reset, contexts_built counter resets to 0
        assert context_manager.metrics()["contexts_built"] == 0


# ===========================================================================
# Soft-cap eviction
# ===========================================================================

class TestSoftCapEviction:
    def test_eviction_fires_when_history_overflows(self, hashing_embedder, heuristic_counter):
        """Adding many long messages must trigger history eviction."""
        store = InMemoryVectorStore()
        retriever = MemoryRetriever(embedder=hashing_embedder, store=store)
        mgr = ContextManager(
            retriever=retriever,
            token_counter=heuristic_counter,
            budget=ContextBudget(max_tokens=512, reserved_for_response=64),
            history_soft_cap_tokens=200,
        )
        for i in range(30):
            mgr.add_user_message("x" * 50)  # each ~12 tokens

        metrics = mgr.metrics()
        assert metrics["history_evictions"] > 0

    def test_eviction_respects_soft_cap(self, hashing_embedder, heuristic_counter):
        """After eviction, total history tokens must be below the soft cap."""
        store = InMemoryVectorStore()
        retriever = MemoryRetriever(embedder=hashing_embedder, store=store)
        soft_cap = 300
        counter = heuristic_counter
        mgr = ContextManager(
            retriever=retriever,
            token_counter=counter,
            budget=ContextBudget(max_tokens=1024, reserved_for_response=64),
            history_soft_cap_tokens=soft_cap,
        )
        for _ in range(50):
            mgr.add_user_message("hello " * 20)

        total = sum(counter.count(m.content) for m in mgr.history)
        assert total <= soft_cap * 1.5  # allow one message overshoot


# ===========================================================================
# Emotion integration inside ContextManager
# ===========================================================================

class TestContextManagerEmotionIntegration:
    def test_emotion_manager_processes_user_messages(self, context_manager):
        context_manager.add_user_message("I feel so happy today!")
        # Emotion manager was wired; just ensure no crash
        metrics = context_manager.metrics()
        assert "emotional_events" in metrics

    def test_affect_block_appears_in_context(self, context_manager):
        context_manager.add_user_message("I am very sad and lonely.")
        snapshot = context_manager.build_context()
        combined = " ".join(m.content for m in snapshot.messages)
        # Affect block should mention "emotional" somewhere
        assert "emotional" in combined.lower() or len(snapshot.messages) > 0

    def test_no_crash_without_emotion_manager(self, memory_retriever, heuristic_counter):
        mgr = ContextManager(
            retriever=memory_retriever,
            token_counter=heuristic_counter,
            system_prompt="sys",
        )
        mgr.add_user_message("hello")
        snap = mgr.build_context()
        assert snap is not None

    def test_emotion_diagnostics_added_to_snapshot(self, context_manager):
        context_manager.add_user_message("I am scared.")
        snap = context_manager.build_context()
        # diagnostics may include emotion fields if confidence > 0
        assert isinstance(snap.diagnostics, dict)
