"""
conftest.py
===========
Shared pytest fixtures for the emotion-chatbot backend test suite.

All heavy ML objects (classifiers, LLMs, Redis, Mem0) are replaced with
lightweight fakes so tests run without any external services or GPU.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path fix — make sure `backend/` is importable as the root package
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# Fake / stub primitives
# ===========================================================================

class FakeMessage:
    """Minimal message object accepted by EmotionManager.process_turn()."""
    def __init__(self, content: str, message_id: str = "test-msg-id"):
        self.content = content
        self.message_id = message_id


class DummyClassifier:
    """
    Dependency-free emotion classifier.

    Returns deterministic EmotionSignals keyed on simple keyword matching
    so unit tests can assert on known outputs without loading any model.
    """

    def classify(self, turns: Sequence[Any]):
        from core.emotion_engine import EmotionSignal, VAD, probs_to_vad

        text = str(getattr(turns[-1], "content", turns[-1])).lower() if turns else ""

        if any(w in text for w in ("happy", "excited", "great", "wonderful", "joy")):
            probs = {"joy": 0.85, "excitement": 0.62, "optimism": 0.38}
        elif any(w in text for w in ("sad", "depressed", "cry", "grief")):
            probs = {"sadness": 0.80, "grief": 0.35, "disappointment": 0.30}
        elif any(w in text for w in ("angry", "furious", "rage", "anger")):
            probs = {"anger": 0.82, "annoyance": 0.55}
        elif any(w in text for w in ("scared", "fear", "worried", "anxious")):
            probs = {"fear": 0.78, "nervousness": 0.50}
        else:
            probs = {"neutral": 0.90}

        vad = probs_to_vad(probs)
        top = max(probs.values())
        second = sorted(probs.values(), reverse=True)[1] if len(probs) > 1 else 0.0
        confidence = min(1.0, 0.5 * top + 0.5 * (top - second))
        return EmotionSignal(vad=vad, confidence=confidence, label_probs=probs, source="dummy")


class LowConfidenceClassifier:
    """Always returns a near-neutral, sub-threshold signal."""

    def classify(self, turns: Sequence[Any]):
        from core.emotion_engine import EmotionSignal, VAD
        return EmotionSignal(vad=VAD(0.05, 0.05, 0.05), confidence=0.10, label_probs={}, source="low-conf")


class NeutralClassifier:
    """Always returns exactly neutral VAD with moderate confidence."""

    def classify(self, turns: Sequence[Any]):
        from core.emotion_engine import EmotionSignal, VAD
        return EmotionSignal(vad=VAD.neutral(), confidence=0.50, label_probs={"neutral": 0.90}, source="neutral")


# ===========================================================================
# Core fixtures
# ===========================================================================

@pytest.fixture
def dummy_classifier():
    return DummyClassifier()


@pytest.fixture
def low_conf_classifier():
    return LowConfidenceClassifier()


@pytest.fixture
def neutral_classifier():
    return NeutralClassifier()


@pytest.fixture
def emotion_manager(dummy_classifier):
    """Full EmotionManager wired with the dummy classifier."""
    from core.emotion_engine import EmotionManager, EmotionManagerConfig
    cfg = EmotionManagerConfig(
        sustained_support=2,
        ltm_support=3,
        context_turns=4,
        signal_window=12,
        spike_delta=0.20,           # lower threshold so tests fire events reliably
        situational_min_confidence=0.10,
        stm_min_confidence=0.40,
        ltm_min_confidence=0.55,
    )
    return EmotionManager(classifier=dummy_classifier, config=cfg)


@pytest.fixture
def emotion_manager_neutral(neutral_classifier):
    from core.emotion_engine import EmotionManager, EmotionManagerConfig
    cfg = EmotionManagerConfig(sustained_support=1, ltm_support=2, context_turns=3)
    return EmotionManager(classifier=neutral_classifier, config=cfg)


@pytest.fixture
def fake_message():
    return FakeMessage


@pytest.fixture
def sample_history(fake_message):
    """Pre-built 4-message history for window-based tests."""
    return [
        fake_message("I felt great today!", f"m{i}")
        for i in range(4)
    ]


# ===========================================================================
# Context-engine fixtures (no heavy deps)
# ===========================================================================

@pytest.fixture
def hashing_embedder():
    from core.context_engine import HashingEmbedder
    return HashingEmbedder(dimension=64)


@pytest.fixture
def heuristic_counter():
    from core.context_engine import HeuristicTokenCounter
    return HeuristicTokenCounter(chars_per_token=4.0)


@pytest.fixture
def in_memory_store():
    from core.context_engine import InMemoryVectorStore
    return InMemoryVectorStore()


@pytest.fixture
def memory_retriever(hashing_embedder, in_memory_store):
    from core.context_engine import MemoryRetriever, RetrievalConfig
    return MemoryRetriever(
        embedder=hashing_embedder,
        store=in_memory_store,
        config=RetrievalConfig(top_k=3, min_score=0.0),
    )


@pytest.fixture
def context_window(heuristic_counter):
    from core.context_engine import ContextWindow, ContextBudget
    budget = ContextBudget(
        max_tokens=512,
        reserved_for_response=64,
        max_memory_ratio=0.30,
        max_system_ratio=0.25,
        max_affect_ratio=0.05,
    )
    return ContextWindow(token_counter=heuristic_counter, budget=budget)


@pytest.fixture
def context_manager(memory_retriever, heuristic_counter, emotion_manager):
    from core.context_engine import ContextManager, ContextBudget
    return ContextManager(
        retriever=memory_retriever,
        token_counter=heuristic_counter,
        budget=ContextBudget(max_tokens=512, reserved_for_response=64),
        system_prompt="You are a helpful assistant.",
        emotion_manager=emotion_manager,
    )


# ===========================================================================
# Redis fixtures (disabled / mocked by default)
# ===========================================================================

@pytest.fixture
def disabled_redis_cache():
    """RedisCache with enabled=False — no actual Redis needed."""
    from memory.redis_cache import RedisCache
    return RedisCache(enabled=False)


@pytest.fixture
def mock_redis_client():
    """A MagicMock that quacks like a redis.Redis client."""
    client = MagicMock()
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.setex.return_value = True
    client.delete.return_value = 1
    client.keys.return_value = []
    client.exists.return_value = 0
    client.ttl.return_value = -1
    client.incrby.return_value = 1
    client.info.return_value = {"keyspace_hits": 10, "keyspace_misses": 5}
    return client


@pytest.fixture
def live_redis_cache(mock_redis_client):
    """
    RedisCache with a mocked Redis client injected after construction.
    Acts like a "connected" cache without a real Redis server.
    """
    from memory.redis_cache import RedisCache
    cache = RedisCache(enabled=False)   # skip real connection
    cache._enabled = True
    cache._client = mock_redis_client
    return cache


# ===========================================================================
# Mem0 fixtures (fully mocked)
# ===========================================================================

@pytest.fixture
def mock_mem0():
    """MagicMock replacing the mem0.Memory class."""
    m = MagicMock()
    m.add.return_value = {"results": [{"id": "mem-1", "memory": "test memory"}]}
    m.search.return_value = {"results": [{"id": "mem-1", "memory": "test memory", "score": 0.9}]}
    m.get_all.return_value = {"results": [{"id": "mem-1", "memory": "test memory", "metadata": {"type": "general"}}]}
    m.update.return_value = {"id": "mem-1", "memory": "updated memory"}
    m.delete.return_value = None
    m.delete_all.return_value = None
    return m


@pytest.fixture
def mem0_manager(mock_mem0):
    """Mem0MemoryManager with the heavy mem0.Memory object replaced."""
    from memory.mem0_integration import Mem0MemoryManager, Mem0Config
    config = Mem0Config(
        vector_store="qdrant",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
    )
    with patch("memory.mem0_integration.Mem0MemoryManager._initialize_mem0"):
        mgr = Mem0MemoryManager.__new__(Mem0MemoryManager)
        mgr.config = config
        mgr._mem0 = mock_mem0
    return mgr


# ===========================================================================
# Ollama LLM fixtures (HTTP mocked)
# ===========================================================================

@pytest.fixture
def mock_ollama_response():
    """Requests response mock for a successful Ollama /api/chat call."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": "I hear you. That sounds tough."}}
    return resp


@pytest.fixture
def mock_ollama_tags_response():
    """Requests response mock for Ollama /api/tags (model list)."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"name": "phi4-mini"}]}
    return resp


@pytest.fixture
def ollama_llm(mock_ollama_tags_response):
    """OllamaLLM with the HTTP connectivity mocked out."""
    from response.ollama_llm import OllamaLLM
    with patch("response.ollama_llm.requests.get", return_value=mock_ollama_tags_response):
        llm = OllamaLLM(model="phi4-mini", base_url="http://localhost:11434")
    return llm


# ===========================================================================
# Response-generator fixtures
# ===========================================================================

@pytest.fixture
def response_generator():
    from response.response_generator import ResponseGenerator
    return ResponseGenerator()


# ===========================================================================
# FastAPI test client (integration tests)
# ===========================================================================

@pytest.fixture(scope="module")
def test_client():
    """
    FastAPI TestClient with all heavy dependencies patched at import time.

    Patches applied:
    - FinalAdapterClassifier → DummyClassifier (no torch/PEFT needed)
    - Mem0MemoryManager → fully mocked
    - OllamaLLM → fully mocked
    - Redis → disabled
    - SentenceTransformerEmbedder → HashingEmbedder
    - TiktokenCounter → HeuristicTokenCounter
    - ChromaVectorStore → InMemoryVectorStore
    """
    from unittest.mock import MagicMock, patch

    dummy_clf = DummyClassifier()

    # Build a mock mem0 manager
    mock_mem0_inst = MagicMock()
    mock_mem0_inst.config = MagicMock()
    mock_mem0_inst.config.vector_store = "qdrant"
    mock_mem0_inst.config.embedder_provider = "openai"
    mock_mem0_inst.config.embedder_model = "text-embedding-3-small"
    mock_mem0_inst.config.enable_graph = False
    mock_mem0_inst.search_memories.return_value = []
    mock_mem0_inst.add_emotional_event.return_value = {"results": []}
    mock_mem0_inst.add_conversation_turn.return_value = {"results": []}
    mock_mem0_inst.get_all_memories.return_value = []
    mock_mem0_inst.search_emotional_memories.return_value = []
    mock_mem0_inst.get_memory_summary.return_value = {"total_memories": 0}
    mock_mem0_inst.add_memory.return_value = {"results": []}
    mock_mem0_inst.delete_memory.return_value = True
    mock_mem0_inst.delete_all_memories.return_value = True

    with (
        patch("core.emotion_engine.FinalAdapterClassifier", return_value=dummy_clf),
        patch("response.response_generator.FinalAdapterClassifier", return_value=dummy_clf),
        patch("memory.mem0_integration.Mem0MemoryManager._initialize_mem0"),
        patch("memory.redis_cache.RedisCache.__init__", lambda self, **kw: (
            setattr(self, "_enabled", False) or
            setattr(self, "_client", None) or
            setattr(self, "_lock", threading.RLock())
        )),
        patch("core.context_manager_integration.SentenceTransformerEmbedder",
              side_effect=lambda: __import__(
                  "core.context_engine", fromlist=["HashingEmbedder"]
              ).HashingEmbedder(dimension=64)),
        patch("core.context_manager_integration.TiktokenCounter",
              side_effect=lambda model="gpt-4o": __import__(
                  "core.context_engine", fromlist=["HeuristicTokenCounter"]
              ).HeuristicTokenCounter()),
        patch("core.context_manager_integration.ChromaVectorStore",
              side_effect=lambda **kw: __import__(
                  "core.context_engine", fromlist=["InMemoryVectorStore"]
              ).InMemoryVectorStore()),
    ):
        # Import main inside the patches so module-level init uses fakes
        import importlib
        import main as app_module
        # Inject mock mem0
        app_module.mem0_manager = mock_mem0_inst

        from fastapi.testclient import TestClient
        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client
