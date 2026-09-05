"""
final_context_engine.py

Production-grade Context Window + Memory Retriever + Emotion-Aware System for LLM applications.

Architecture:
    ContextManager            -> high-level orchestrator (write path + read path)
        ├── ContextWindow     -> token-budgeted prompt assembly with eviction
        ├── MemoryRetriever   -> hybrid-scored long-term memory (semantic + recency + importance)
        │       ├── Embedder (Protocol)       -> pluggable embedding model
        │       └── VectorStore (ABC)         -> pluggable ANN backend
        ├── EmotionManager    -> IMPORTED from emotion_engine.py (VAD-based affect tracking)
        │       ├── Situational (seconds)     -> immediate turn-level emotion
        │       ├── Short-term (minutes)      -> recent conversational mood
        │       └── Long-term (days/weeks)    -> persistent user trait baseline
        ├── EmotionalMemoryWriter -> IMPORTED from emotion_engine.py
        └── TokenCounter (Protocol)           -> pluggable tokenizer

Design goals:
    - Thread-safe, dependency-free core (stdlib only)
    - Deterministic fallbacks (heuristic tokenizer, hashing embedder)
    - Bounded memory: hard token budgets, soft history caps, summarization-on-evict
    - Observable: structured diagnostics + metrics on every context build
    - Emotion-aware: affect state injected into every context assembly
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

# ═══════════════════════════════════════════════════════════════════════
# IMPORT EMOTION COMPONENTS FROM emotion_engine.py (SINGLE SOURCE OF TRUTH)
# ═══════════════════════════════════════════════════════════════════════
from core.emotion_engine import (
    VAD,
    EmotionLabel,
    AffectState,
    EmotionManager,
    EmotionalMemoryWriter,
    EmotionSignal,
    render_affect_line,
    project_label,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id", uuid.uuid4().hex),
        )


@dataclass
class Memory:
    """A unit of long-term memory."""
    content: str
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    embedding: Optional[list[float]] = None
    importance: float = 0.5                       # [0, 1], set at write time
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.importance = min(1.0, max(0.0, float(self.importance)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "memory_id": self.memory_id,
            "embedding": self.embedding,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            content=data["content"],
            memory_id=data.get("memory_id", uuid.uuid4().hex),
            embedding=data.get("embedding"),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", time.time()),
            last_accessed_at=data.get("last_accessed_at", time.time()),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    similarity: float
    recency: float
    importance: float


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """Accurate counting via tiktoken (pip install tiktoken)."""

    def __init__(self, model: str = "gpt-4o") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken is required for TiktokenCounter") from exc
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class HeuristicTokenCounter:
    """Zero-dependency fallback (~4 chars/token). Conservative by default."""

    def __init__(self, chars_per_token: float = 3.5) -> None:
        self._cpt = chars_per_token

    def count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self._cpt))


def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Binary-search truncation that works with any TokenCounter."""
    if counter.count(text) <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + " …[truncated]"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """
    Deterministic n-gram hashing embedder (feature hashing + L2 norm).

    REMOVED FROM PRODUCTION USE. Retained only for unit tests that need
    a dependency-free embedder. Do NOT use in ChatContextManager.
    Use SentenceTransformerEmbedder instead.
    """

    def __init__(self, dimension: int = 512, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self._dim = dimension
        self._ngram_range = ngram_range

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                idx = h % self._dim
                sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec


class SentenceTransformerEmbedder:
    """
    Production semantic embedder using sentence-transformers.

    Understands meaning — 'happy' and 'joyful' will be close in embedding
    space. Required for vector search to return semantically relevant results.

    HashingEmbedder must NOT be used as a fallback: it hashes strings without
    understanding meaning, causing vector search to return garbage results
    silently.

    Install: pip install sentence-transformers

    Args:
        model_name: HuggingFace model ID. Defaults to a fast, high-quality
                    384-dim model. Override via EMBEDDER_MODEL env var.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        resolved = model_name or os.environ.get("EMBEDDER_MODEL", self.DEFAULT_MODEL)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(resolved)
            self._dim: int = self._model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required but not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{resolved}'. "
                f"Check EMBEDDER_MODEL env var or model availability. "
                f"Original error: {e}"
            ) from e

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), show_progress_bar=False)
        return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class VectorStore(ABC):
    """Pluggable ANN backend (swap for FAISS / pgvector / Qdrant in production)."""

    @abstractmethod
    def upsert(self, memory_id: str, embedding: Sequence[float], metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[tuple[str, float]]:
        """Return [(memory_id, cosine_similarity)] sorted by similarity desc."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """Thread-safe brute-force store. O(n) scan — fine up to ~100k vectors."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def upsert(self, memory_id: str, embedding: Sequence[float], metadata: dict[str, Any]) -> None:
        with self._lock:
            self._vectors[memory_id] = (list(embedding), dict(metadata))

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            snapshot = list(self._vectors.items())
        scored: list[tuple[str, float]] = []
        for memory_id, (vec, meta) in snapshot:
            if filter_fn is not None and not filter_fn(meta):
                continue
            scored.append((memory_id, _cosine_similarity(query_embedding, vec)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._vectors.pop(memory_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._vectors)


class ChromaVectorStore(VectorStore):
    """
    Production-ready vector store using ChromaDB.
    
    ChromaDB provides:
    - Persistent storage to disk
    - Efficient ANN search with HNSW
    - Built-in metadata filtering
    - Automatic indexing
    
    Parameters:
    - persist_directory: Path to store ChromaDB data (default: ./chroma_db)
    - collection_name: Name of the collection (default: emotion_memories)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "emotion_memories",
        embedding_dimension: Optional[int] = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            ) from exc

        # BUG FIX #6: chromadb >= 0.4.0 removed the legacy Client(Settings(...))
        # API.  Use PersistentClient(path=...) for on-disk storage and
        # EphemeralClient() for in-memory (test) usage instead.
        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
            )
        except AttributeError:
            # Fallback for very old chromadb versions still using the old API.
            from chromadb.config import Settings  # type: ignore[import]
            self._client = chromadb.Client(  # type: ignore[attr-defined]
                Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False,
                )
            )

        # Get or create collection
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # If collection exists with different settings, get it as-is.
            self._collection = self._client.get_collection(name=collection_name)

        self._lock = threading.RLock()
        logger.info(
            f"ChromaVectorStore initialized: collection='{collection_name}', "
            f"persist_dir='{persist_directory}', count={self._collection.count()}"
        )

    def upsert(self, memory_id: str, embedding: Sequence[float], metadata: dict[str, Any]) -> None:
        with self._lock:
            # ChromaDB requires string metadata values
            sanitized_metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in metadata.items()
            }
            
            self._collection.upsert(
                ids=[memory_id],
                embeddings=[list(embedding)],
                metadatas=[sanitized_metadata],
            )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            # ChromaDB returns distances, not similarities
            # For cosine space, distance = 1 - similarity (or 2 - 2*similarity for normalized)
            # We need to convert back to similarity
            results = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k if filter_fn is None else top_k * 3,  # Overfetch if filtering
                include=["distances", "metadatas"],
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            # Extract results
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            
            # Convert distances to similarities (for cosine: similarity = 1 - distance)
            scored: list[tuple[str, float]] = []
            for memory_id, distance, metadata in zip(ids, distances, metadatas):
                # Apply filter if provided
                if filter_fn is not None and not filter_fn(metadata):
                    continue
                
                # Convert distance to similarity
                similarity = 1.0 - distance
                scored.append((memory_id, similarity))
            
            # Sort by similarity (descending) and limit to top_k
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[:top_k]

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            try:
                # Check if exists
                existing = self._collection.get(ids=[memory_id])
                if not existing["ids"]:
                    return False
                
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False

    def __len__(self) -> int:
        with self._lock:
            return self._collection.count()


# ---------------------------------------------------------------------------
# Memory retriever
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.05                    # drop results below this hybrid score
    weight_similarity: float = 0.7
    weight_recency: float = 0.2
    weight_importance: float = 0.1
    recency_half_life_seconds: float = 86_400.0  # recency halves every 24h
    overfetch_factor: int = 3                  # ANN overfetch before hybrid re-rank

    def __post_init__(self) -> None:
        total = self.weight_similarity + self.weight_recency + self.weight_importance
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")


class MemoryRetriever:
    """
    Long-term memory with hybrid scoring:

        score = w_sim * cosine_similarity
              + w_rec * 0.5 ** (age / half_life)
              + w_imp * importance

    Retrieved memories are reinforced (access_count++, last_accessed bumped),
    enabling frequency-aware downstream policies (e.g., decay pruning).
    """

    def __init__(
        self,
        embedder: Embedder,
        store: Optional[VectorStore] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._embedder = embedder
        # BUG FIX #7: accept None and default to InMemoryVectorStore so that
        # MemoryRetriever.load(..., store=None) doesn't crash on first upsert.
        self._store: VectorStore = store if store is not None else InMemoryVectorStore()
        self._config = config or RetrievalConfig()
        self._memories: dict[str, Memory] = {}
        self._lock = threading.RLock()

    # -- write path ---------------------------------------------------------

    def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Memory:
        content = content.strip()
        if not content:
            raise ValueError("Cannot store empty memory")
        embedding = self._embedder.embed([content])[0]
        memory = Memory(content=content, importance=importance, metadata=metadata or {})
        memory.embedding = embedding
        with self._lock:
            self._memories[memory.memory_id] = memory
            self._store.upsert(memory.memory_id, embedding, memory.metadata)
        logger.debug("memory stored id=%s importance=%.2f", memory.memory_id, memory.importance)
        return memory

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            existed = self._memories.pop(memory_id, None) is not None
            self._store.delete(memory_id)
        return existed

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            return self._memories.get(memory_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._memories)

    # -- read path ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> list[ScoredMemory]:
        cfg = self._config
        top_k = top_k or cfg.top_k
        if not query.strip() or len(self) == 0:
            return []

        query_vec = self._embedder.embed([query])[0]
        candidates = self._store.search(
            query_vec, top_k=top_k * cfg.overfetch_factor, filter_fn=filter_fn
        )

        now = time.time()
        total_w = cfg.weight_similarity + cfg.weight_recency + cfg.weight_importance
        results: list[ScoredMemory] = []

        with self._lock:
            for memory_id, similarity in candidates:
                memory = self._memories.get(memory_id)
                if memory is None:
                    continue
                similarity = max(0.0, similarity)  # clamp negative cosine
                age = max(0.0, now - memory.created_at)
                recency = 0.5 ** (age / cfg.recency_half_life_seconds)
                score = (
                    cfg.weight_similarity * similarity
                    + cfg.weight_recency * recency
                    + cfg.weight_importance * memory.importance
                ) / total_w
                if score < cfg.min_score:
                    continue
                # Access reinforcement: frequently retrieved memories stay "hot"
                memory.access_count += 1
                memory.last_accessed_at = now
                results.append(
                    ScoredMemory(
                        memory=memory,
                        score=score,
                        similarity=similarity,
                        recency=recency,
                        importance=memory.importance,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize memories (incl. embeddings) to JSON."""
        with self._lock:
            payload = [m.to_dict() for m in self._memories.values()]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("saved %d memories -> %s", len(payload), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder,
        store: Optional[VectorStore] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> "MemoryRetriever":
        retriever = cls(embedder=embedder, store=store, config=config)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        with retriever._lock:
            for data in payload:
                memory = Memory.from_dict(data)
                if memory.embedding is None:  # re-embed legacy entries
                    memory.embedding = embedder.embed([memory.content])[0]
                retriever._memories[memory.memory_id] = memory
                retriever._store.upsert(memory.memory_id, memory.embedding, memory.metadata)
        logger.info("loaded %d memories <- %s", len(payload), path)
        return retriever


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 8192                 # model context window size
    reserved_for_response: int = 1024      # headroom for the model's answer
    max_memory_ratio: float = 0.30         # cap on retrieved-memory block
    max_system_ratio: float = 0.25         # guard rail on system prompt size
    max_affect_ratio: float = 0.05         # cap on emotional-state block
    per_message_overhead: int = 4          # chat-format framing tokens / message

    def __post_init__(self) -> None:
        if self.reserved_for_response >= self.max_tokens:
            raise ValueError("reserved_for_response must be < max_tokens")
        for name in ("max_memory_ratio", "max_system_ratio", "max_affect_ratio"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")

    @property
    def available(self) -> int:
        return self.max_tokens - self.reserved_for_response


@dataclass
class ContextSnapshot:
    """Assembled context plus full diagnostics for observability."""
    messages: list[Message]
    total_tokens: int
    budget_tokens: int
    memories_used: list[ScoredMemory]
    dropped_history_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_chat_format(self) -> list[dict[str, str]]:
        """Ready to pass to OpenAI-style chat APIs."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]


class ContextWindow:
    """
    Token-budgeted prompt assembly.

    Priority order (highest first):
        1. System prompt   — pinned, truncated only if it breaches its guard rail
        2. Affect block    — emotional state summary, capped at max_affect_ratio
        3. Memory block    — retrieved memories, capped at max_memory_ratio
        4. Conversation    — filled newest-first; the latest message is always
                             included (truncated if it alone exceeds the budget)
    """

    MEMORY_HEADER = "Relevant long-term memories (most relevant first):"

    def __init__(self, token_counter: TokenCounter, budget: Optional[ContextBudget] = None) -> None:
        self._counter = token_counter
        self._budget = budget or ContextBudget()

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    def _msg_tokens(self, msg: Message) -> int:
        return self._counter.count(msg.content) + self._budget.per_message_overhead

    def assemble(
        self,
        system_prompt: Optional[str],
        memories: Sequence[ScoredMemory],
        history: Sequence[Message],
        affect_block: Optional[str] = None,
    ) -> ContextSnapshot:
        budget = self._budget.available
        messages: list[Message] = []
        used = 0
        diagnostics: dict[str, Any] = {}

        # -- 1. system prompt (pinned) ---------------------------------------
        if system_prompt:
            sys_tokens = self._counter.count(system_prompt) + self._budget.per_message_overhead
            sys_cap = int(budget * self._budget.max_system_ratio)
            if sys_tokens > sys_cap:
                logger.warning("system prompt %d tokens exceeds cap %d; truncating", sys_tokens, sys_cap)
                system_prompt = truncate_to_tokens(system_prompt, sys_cap, self._counter)
                sys_tokens = self._counter.count(system_prompt) + self._budget.per_message_overhead
                diagnostics["system_truncated"] = True
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
            used += sys_tokens
        diagnostics["system_tokens"] = used

        # -- 2. affect block (pinned, small) ---------------------------------
        affect_tokens = 0
        if affect_block:
            affect_cap = int(budget * self._budget.max_affect_ratio)
            fitted = truncate_to_tokens(affect_block, affect_cap, self._counter)
            affect_tokens = self._counter.count(fitted) + self._budget.per_message_overhead
            if affect_tokens <= budget - used:
                messages.append(Message(role=Role.SYSTEM, content=fitted,
                                        metadata={"kind": "affect_block"}))
                used += affect_tokens
            else:
                diagnostics["affect_block_dropped"] = True
                affect_tokens = 0
        diagnostics["affect_tokens"] = affect_tokens

        # -- 3. memory block (budget-capped) ----------------------------------
        memory_budget = int(budget * self._budget.max_memory_ratio)
        memory_tokens = 0
        memories_used: list[ScoredMemory] = []
        if memories:
            lines: list[str] = []
            header_cost = self._counter.count(self.MEMORY_HEADER) + self._budget.per_message_overhead
            for scored in memories:
                line = f"- {scored.memory.content}"
                line_cost = self._counter.count(line) + 1
                if header_cost + memory_tokens + line_cost > memory_budget:
                    continue  # a shorter memory further down may still fit
                lines.append(line)
                memory_tokens += line_cost
                memories_used.append(scored)
            if lines:
                block = self.MEMORY_HEADER + "\n" + "\n".join(lines)
                block_msg = Message(role=Role.SYSTEM, content=block,
                                    metadata={"kind": "memory_block"})
                messages.append(block_msg)
                used += header_cost + memory_tokens
        diagnostics["memory_tokens"] = memory_tokens
        diagnostics["memories_considered"] = len(memories)
        diagnostics["memories_included"] = len(memories_used)

        # -- 4. conversation history (newest-first fill) ----------------------
        remaining = budget - used
        kept: list[Message] = []
        history_tokens = 0
        dropped = 0
        for msg in reversed(history):
            cost = self._msg_tokens(msg)
            if history_tokens + cost <= remaining:
                kept.append(msg)
                history_tokens += cost
            else:
                dropped += 1

        if not kept and history:
            # Guarantee the latest message is present, even if it must be truncated.
            last = history[-1]
            trunc_budget = max(0, remaining - self._budget.per_message_overhead)
            truncated = Message(role=last.role,
                                content=truncate_to_tokens(last.content, trunc_budget, self._counter),
                                metadata={**last.metadata, "truncated": True})
            kept.append(truncated)
            history_tokens = self._msg_tokens(truncated)
            dropped = len(history) - 1
            diagnostics["last_message_truncated"] = True

        kept.reverse()
        messages.extend(kept)
        used += history_tokens
        diagnostics["history_tokens"] = history_tokens
        diagnostics["history_messages_kept"] = len(kept)

        return ContextSnapshot(
            messages=messages,
            total_tokens=used,
            budget_tokens=budget,
            memories_used=memories_used,
            dropped_history_count=dropped,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Summarization (eviction hook)
# ---------------------------------------------------------------------------

@runtime_checkable
class Summarizer(Protocol):
    """Plug in an LLM-backed summarizer in production."""
    def summarize(self, messages: Sequence[Message]) -> str: ...


class TruncatingSummarizer:
    """Dependency-free fallback: concatenated excerpts."""

    def __init__(self, max_chars: int = 500) -> None:
        self._max_chars = max_chars

    def summarize(self, messages: Sequence[Message]) -> str:
        parts = [f"{m.role.value}: {m.content[:120]}" for m in messages]
        return ("Earlier conversation summary — " + " | ".join(parts))[: self._max_chars]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    contexts_built: int = 0
    messages_added: int = 0
    memories_added: int = 0
    memories_retrieved: int = 0
    history_evictions: int = 0
    summaries_created: int = 0
    emotional_events_logged: int = 0
    last_context_tokens: int = 0
    last_build_latency_ms: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "contexts_built": self.contexts_built,
            "messages_added": self.messages_added,
            "memories_added": self.memories_added,
            "memories_retrieved": self.memories_retrieved,
            "history_evictions": self.history_evictions,
            "summaries_created": self.summaries_created,
            "emotional_events_logged": self.emotional_events_logged,
            "last_context_tokens": self.last_context_tokens,
            "last_build_latency_ms": round(self.last_build_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Context manager (orchestrator)
# ---------------------------------------------------------------------------

class ContextManager:
    """
    High-level facade combining:
    - Short-term history (conversation)
    - Long-term memory (vector-backed retrieval)
    - Emotion tracking (VAD-based, three-tier) - IMPORTED from emotion_engine.py
    - Token-budgeted context assembly

    Write path:
        add_message()  -> append to history; detect emotion; evict if needed
        remember()     -> explicit long-term memory write

    Read path:
        build_context() -> retrieve memories + build affect block + assemble prompt
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        token_counter: TokenCounter,
        budget: Optional[ContextBudget] = None,
        system_prompt: Optional[str] = None,
        summarizer: Optional[Summarizer] = None,
        history_soft_cap_tokens: Optional[int] = None,
        emotion_manager: Optional[EmotionManager] = None,
        emotional_writer: Optional[EmotionalMemoryWriter] = None,
    ) -> None:
        self._retriever = retriever
        self._counter = token_counter
        self._budget = budget or ContextBudget()
        self._window = ContextWindow(token_counter, self._budget)
        self._system_prompt = system_prompt
        self._summarizer = summarizer
        self._soft_cap = history_soft_cap_tokens or self._budget.available * 2
        self._history: list[Message] = []
        self._metrics = Metrics()
        self._lock = threading.RLock()
        
        # Affective layer — optional so the class still works standalone
        self._emotion_manager = emotion_manager
        self._emotional_writer = emotional_writer

    # -- write path ---------------------------------------------------------

    def add_message(
        self,
        role: Role,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        content = content.strip()
        if not content:
            raise ValueError("Message content must be non-empty")
        msg = Message(role=role, content=content, metadata=metadata or {})
        
        with self._lock:
            self._history.append(msg)
            self._metrics.messages_added += 1
            self._enforce_soft_cap_locked()

        # Affective write path — deliberately OUTSIDE self._lock
        # The classifier call can be slow (network-bound model), and
        # EmotionManager guards its own state, so no need to hold two locks
        if role == Role.USER and self._emotion_manager is not None:
            k = self._emotion_manager.context_turns
            with self._lock:
                recent_turns = list(self._history[-k:])
            
            events = self._emotion_manager.process_turn(msg, recent_turns)
            
            if events:
                with self._lock:
                    self._metrics.emotional_events_logged += len(events)
                
                if self._emotional_writer is not None:
                    self._emotional_writer.maybe_write(events)

        return msg

    def add_user_message(self, content: str, **kwargs: Any) -> Message:
        return self.add_message(Role.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs: Any) -> Message:
        return self.add_message(Role.ASSISTANT, content, **kwargs)

    def remember(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Memory:
        memory = self._retriever.add(content, importance=importance, metadata=metadata)
        with self._lock:
            self._metrics.memories_added += 1
        return memory

    def _enforce_soft_cap_locked(self) -> None:
        """Evict oldest history beyond the soft cap; summarize into memory. Caller holds lock."""
        overhead = self._budget.per_message_overhead
        total = sum(self._counter.count(m.content) + overhead for m in self._history)
        if total <= self._soft_cap:
            return

        evicted: list[Message] = []
        while self._history and total > self._soft_cap:
            oldest = self._history.pop(0)
            total -= self._counter.count(oldest.content) + overhead
            evicted.append(oldest)

        self._metrics.history_evictions += len(evicted)
        logger.info("evicted %d history messages (soft cap %d tokens)", len(evicted), self._soft_cap)

        if self._summarizer and evicted:
            try:
                summary = self._summarizer.summarize(evicted)
                self._retriever.add(
                    summary,
                    importance=0.6,
                    metadata={"kind": "conversation_summary", "source_count": len(evicted)},
                )
                self._metrics.summaries_created += 1
            except Exception:
                logger.exception("summarizer failed; evicted history not persisted to memory")

    # -- read path ----------------------------------------------------------

    def build_context(
        self,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        memory_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> ContextSnapshot:
        """
        Assemble the LLM-ready context.

        Retrieval query defaults to the most recent user message.
        """
        start = time.perf_counter()
        
        with self._lock:
            history = list(self._history)

        if query is None:
            query = next(
                (m.content for m in reversed(history) if m.role == Role.USER),
                "",
            )

        memories = self._retriever.retrieve(query, top_k=top_k, filter_fn=memory_filter)

        # Build affect block if emotion manager is available
        affect_block = None
        if self._emotion_manager is not None:
            affect = self._emotion_manager.affect_state()
            affect_block = self._render_affect_block(affect)

        snapshot = self._window.assemble(
            self._system_prompt, memories, history, affect_block=affect_block
        )

        # Add emotion diagnostics if available
        if self._emotion_manager is not None:
            affect = self._emotion_manager.affect_state()
            stm_label, stm_intensity = project_label(affect.short_term_vad)
            ltm_label, _ = project_label(affect.long_term_vad)
            
            snapshot.diagnostics["emotion_situational"] = affect.stm_dominant.value
            snapshot.diagnostics["emotion_short_term"] = stm_label.value
            snapshot.diagnostics["emotion_trend"] = affect.trend
            snapshot.diagnostics["emotion_confidence"] = round(affect.confidence, 3)

        with self._lock:
            self._metrics.contexts_built += 1
            self._metrics.memories_retrieved += len(snapshot.memories_used)
            self._metrics.last_context_tokens = snapshot.total_tokens
            self._metrics.last_build_latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "context built: %d/%d tokens, %d memories, %d dropped",
            snapshot.total_tokens, snapshot.budget_tokens,
            len(snapshot.memories_used), snapshot.dropped_history_count,
        )
        return snapshot

    @staticmethod
    def _render_affect_block(affect: AffectState) -> str:
        """
        Compact prompt-ready emotional state summary using the production
        render_affect_line function from emotion_engine.py.
        """
        return (
            "Emotional context (model estimate, not certain fact — weigh accordingly):\n" +
            render_affect_line(affect)
        )

    # -- introspection --------------------------------------------------------

    @property
    def history(self) -> list[Message]:
        with self._lock:
            return list(self._history)

    @property
    def retriever(self) -> MemoryRetriever:
        return self._retriever

    @property
    def emotion_manager(self) -> Optional[EmotionManager]:
        return self._emotion_manager

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            snap = self._metrics.snapshot()
        snap["history_messages"] = len(self._history)
        snap["long_term_memories"] = len(self._retriever)
        if self._emotion_manager is not None:
            snap["emotional_events"] = len(self._emotion_manager.events())
        return snap

    def reset(self, keep_memories: bool = True, keep_emotions: bool = True) -> None:
        with self._lock:
            self._history.clear()
            self._metrics = Metrics()
        
        if not keep_memories:
            for m in list(self._retriever._memories.values()):
                self._retriever.remove(m.memory_id)
        
        if not keep_emotions and self._emotion_manager is not None:
            self._emotion_manager.reset_session()


# ---------------------------------------------------------------------------
# Demo / Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("\n" + "="*70)
    print("CONTEXT ENGINE - Using emotion_engine.py for emotion tracking")
    print("="*70 + "\n")
    
    # Initialize components
    # NOTE: Demo uses SentenceTransformerEmbedder + ChromaDB (production stack).
    # HashingEmbedder and HeuristicTokenCounter are NOT used here.
    embedder = SentenceTransformerEmbedder()
    token_counter = TiktokenCounter(model="gpt-4o")
    
    # Import and initialize emotion manager from emotion_engine.py
    from core.emotion_engine import build_emotion_manager
    
    emotion_manager = build_emotion_manager()
    
    # Initialize ChromaDB vector store (REQUIRED - no in-memory fallback)
    vector_store = ChromaVectorStore(
        persist_directory="./test_chroma_db",
        collection_name="test_memories",
    )
    print(f"✅ Using ChromaDB vector store (persistent)")
    
    retriever = MemoryRetriever(
        embedder=embedder,
        store=vector_store,
        config=RetrievalConfig(top_k=3, min_score=0.05),
    )
    
    emotional_writer = EmotionalMemoryWriter(
        retriever=retriever,
        min_magnitude=0.35,
        min_confidence=0.55,
    )
    
    # Create context manager with emotion tracking from emotion_engine.py
    manager = ContextManager(
        retriever=retriever,
        token_counter=token_counter,
        budget=ContextBudget(
            max_tokens=2048,
            reserved_for_response=512,
            max_memory_ratio=0.30,
            max_affect_ratio=0.05,
        ),
        system_prompt="You are an empathetic AI assistant.",
        summarizer=TruncatingSummarizer(),
        history_soft_cap_tokens=1000,
        emotion_manager=emotion_manager,
        emotional_writer=emotional_writer,
    )
    
    # Example conversation
    print("\nTurn 1: User expresses emotion")
    manager.add_user_message("I've been feeling really lonely lately. Nobody seems to understand me.")
    ctx1 = manager.build_context()
    print(f"Tokens: {ctx1.total_tokens}/{ctx1.budget_tokens}")
    print(f"Emotion: {ctx1.diagnostics.get('emotion_short_term', 'N/A')}")
    print(f"Confidence: {ctx1.diagnostics.get('emotion_confidence', 'N/A')}")
    print()
    
    # Show metrics
    print("="*70)
    print("METRICS")
    print("="*70)
    metrics = manager.metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    
    print("✅ Successfully using emotion_engine.py as single source of truth!")
    print(f"✅ Vector store: {type(vector_store).__name__} with {len(vector_store)} vectors")
