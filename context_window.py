"""

context_engine.py

Production-grade Context Window + Memory Retriever for LLM applications.

Architecture:
    ContextManager            -> high-level orchestrator (write path + read path)
        ├── ContextWindow     -> token-budgeted prompt assembly with eviction
        ├── MemoryRetriever   -> hybrid-scored long-term memory (semantic + recency + importance)
        │       ├── Embedder (Protocol)       -> pluggable embedding model
        │       └── VectorStore (ABC)         -> pluggable ANN backend
        └── EmotionManager    -> COMPULSORY. Every user turn circulates through
                                 emotion detection; every context build carries
                                 the affect block. Event log feeds MemoryRetriever
                                 via EmotionalMemoryWriter.
        └── TokenCounter (Protocol)           -> pluggable tokenizer

Design goals:
    - Thread-safe, dependency-free core (stdlib only).
    - Deterministic fallbacks (heuristic tokenizer, hashing embedder) so the
      system runs standalone; swap in real backends via dependency injection.
    - Bounded memory: hard token budgets, soft history caps, summarization-on-evict.
    - Observable: structured diagnostics + metrics on every context build.


This code implements a production-grade, thread-safe Context and Memory Management System for Large Language Model (LLM) applications. It is designed to solve the classic LLM problems of context window limits, memory retention, and token budgeting, while remaining completely standalone (zero third-party dependencies required for core functionality).
Here is a detailed breakdown of the architectural model and how it is coded.
1. High-Level Architecture
The system is built in layered, decoupled components using Dependency Injection (via Python Protocol and ABC). This allows you to swap in production-grade tools (like OpenAI embeddings, FAISS, or tiktoken) while keeping robust, zero-dependency fallbacks for testing or standalone operation.
The main orchestrator is the ContextManager, which coordinates:
Short-term memory: Conversation history (_history).
Long-term memory: MemoryRetriever (vector-backed).
Token budgeting: ContextWindow (assembles the final prompt).
2. Core Components Breakdown
A. Data Models
Message: Represents a single turn in a conversation (System, User, Assistant, Tool). Includes metadata and timestamps.
Memory: Represents a long-term memory unit. Tracks content, embedding, importance (0.0–1.0), created_at, last_accessed_at, and access_count (for frequency-based reinforcement).
ScoredMemory: A wrapper returned during retrieval, containing the original Memory plus its calculated score, similarity, recency, and importance.
B. Pluggable Interfaces & Zero-Dependency Fallbacks
The code defines Protocols/ABCs, ensuring the core logic doesn't care about the underlying implementation:
TokenCounter:
Production: TiktokenCounter (accurate, requires tiktoken package).
Fallback: HeuristicTokenCounter (estimates ~3.5 characters per token).
Utility: truncate_to_tokens uses binary search to safely truncate text to a strict token limit without over-counting.
Embedder:
Fallback: HashingEmbedder. A brilliant zero-dependency trick that uses n-gram tokenization + BLAKE2b hashing + feature hashing (with L2 normalization) to generate deterministic, fixed-dimension vectors without ML libraries.
VectorStore:
Fallback: InMemoryVectorStore. A thread-safe (threading.RLock) dictionary-based store that does brute-force cosine similarity. Fine for <100k vectors.
Summarizer:
Fallback: TruncatingSummarizer. Concatenates short excerpts of evicted messages.
C. The Memory Retrieval Model (MemoryRetriever)
This component handles long-term memory using a Hybrid Scoring Algorithm. When a query is made, it fetches candidates via the vector store, then re-ranks them using three factors:
Similarity: Cosine similarity between query and memory embedding.
Recency: Decays exponentially over time. recency = 0.5 ** (age_in_seconds / half_life_seconds). (Defaults to halving every 24 hours).
Importance: A static weight assigned at write time (e.g., critical user preferences get 0.9).
Formula:
score = (w_sim * similarity + w_rec * recency + w_imp * importance) / total_weights
Note: Every time a memory is retrieved, its access_count increments and last_accessed_at updates, naturally keeping frequently used memories "hot".
D. The Context Assembly Model (ContextWindow)
This is the engine that builds the final prompt for the LLM without exceeding the ContextBudget. It enforces a strict Priority Order:
System Prompt: Always included. If it exceeds max_system_ratio (e.g., 25% of budget), it is aggressively truncated.
Memory Block: Injected as a System message. Iterates through retrieved memories, adding them until the max_memory_ratio (e.g., 30% of budget) is hit.
Conversation History: Filled newest-first. It guarantees the absolute latest message is always included (truncating it if necessary), then works backward to fill the remaining token budget. Older messages are dropped.
E. The Orchestrator (ContextManager)
This is the facade the application interacts with.
Write Path:
add_message(): Appends to history. Checks if the total history exceeds history_soft_cap_tokens. If so, it evicts the oldest messages, passes them to the Summarizer, and saves the summary as a new long-term Memory.
remember(): Explicitly writes a fact to the MemoryRetriever.
Read Path:
build_context(): Takes the latest user message (or an explicit query), retrieves relevant long-term memories, and calls ContextWindow.assemble() to return a ContextSnapshot.
Observability: Tracks a Metrics dataclass (latency, tokens used, evictions, retrievals) and includes a diagnostics dict in every ContextSnapshot.


"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

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

    Zero-dependency fallback so the system runs standalone. For production,
    inject a real model (OpenAI text-embedding-3, sentence-transformers, ...).
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
        self._store = store or InMemoryVectorStore()
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
    per_message_overhead: int = 4          # chat-format framing tokens / message

    def __post_init__(self) -> None:
        if self.reserved_for_response >= self.max_tokens:
            raise ValueError("reserved_for_response must be < max_tokens")
        for name in ("max_memory_ratio", "max_system_ratio"):
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
        2. Memory block    — retrieved memories, capped at max_memory_ratio
        3. Conversation    — filled newest-first; the latest message is always
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

        # -- 2. memory block (budget-capped) ----------------------------------
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

        # -- 3. conversation history (newest-first fill) ----------------------
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
            "last_context_tokens": self.last_context_tokens,
            "last_build_latency_ms": round(self.last_build_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Context manager (orchestrator)
# ---------------------------------------------------------------------------

class ContextManager:
    """
    High-level facade combining short-term history, long-term memory,
    and token-budgeted context assembly.

    Write path:
        add_message()  -> append to history; evict oldest (optionally
                          summarized into long-term memory) past the soft cap
        remember()     -> explicit long-term memory write

    Read path:
        build_context() -> retrieve memories relevant to the current query,
                           assemble a token-bounded message list for the LLM
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        token_counter: TokenCounter,
        budget: Optional[ContextBudget] = None,
        system_prompt: Optional[str] = None,
        summarizer: Optional[Summarizer] = None,
        history_soft_cap_tokens: Optional[int] = None,
    ) -> None:
        self._retriever = retriever
        self._counter = token_counter
        self._budget = budget or ContextBudget()
        self._window = ContextWindow(token_counter, self._budget)
        self._system_prompt = system_prompt
        self._summarizer = summarizer
        # Raw history may exceed a single window; the soft cap bounds RAM and
        # triggers summarization-on-evict. Defaults to 2x the window budget.
        self._soft_cap = history_soft_cap_tokens or self._budget.available * 2
        self._history: list[Message] = []
        self._metrics = Metrics()
        self._lock = threading.RLock()

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
                # Summarization must never break the write path.
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

        Retrieval query defaults to the most recent user message — pass an
        explicit query for rewritten/expanded retrieval.
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
        snapshot = self._window.assemble(self._system_prompt, memories, history)

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

    # -- introspection --------------------------------------------------------

    @property
    def history(self) -> list[Message]:
        with self._lock:
            return list(self._history)

    @property
    def retriever(self) -> MemoryRetriever:
        return self._retriever

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            snap = self._metrics.snapshot()
        snap["history_messages"] = len(self._history)
        snap["long_term_memories"] = len(self._retriever)
        return snap

    def reset(self, keep_memories: bool = True) -> None:
        with self._lock:
            self._history.clear()
            self._metrics = Metrics()
        if not keep_memories:
            for m in list(self._retriever._memories.values()):
                self._retriever.remove(m.memory_id)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Small budget to exercise eviction + memory retrieval paths.
    manager = ContextManager(
        retriever=MemoryRetriever(
            embedder=HashingEmbedder(dimension=512),
            config=RetrievalConfig(top_k=3, min_score=0.05),
        ),
        token_counter=HeuristicTokenCounter(),
        budget=ContextBudget(max_tokens=512, reserved_for_response=128,
                             max_memory_ratio=0.35),
        system_prompt="You are a helpful support agent for Acme Corp.",
        summarizer=TruncatingSummarizer(),
        history_soft_cap_tokens=300,
    )


But the context should also include the emotion manager in the context window with short-term, long-term and emotion (Token budgeting: ContextWindow)


Here is emotionmanager model:
Affect Core (VAD vector) — single source of truth per tier: valence, arousal, dominance (continuous, 3 numbers) instead of 9 independent bars. Correlated emotions fall out naturally; blending/decay is just vector math (move toward baseline), not 9 separate heuristics.
Discrete Label Projector — maps the VAD point onto the 9 emotion labels (nearest-prototype/circumplex) only when you need bars to display or inject into a prompt. Storage stays continuous; presentation stays multi-label.
Context-aware Classifier Adapter — feeds the model the last k turns, not just the current message, so it can catch negation/sarcasm/tone shifts.
Confidence Gate — low-confidence reads affect Situational only; only confident, sustained signals propagate into Short-term, and only repeated confident signals move Long-term.
Trait Baseline — Long-term decays toward a learned per-user baseline, not zero.
Wall-clock half-life decay — intensity(t) = intensity₀ × 0.5^(Δt/half_life), with a short half-life for Situational/STM (minutes) and a long one for LTM (days), instead of per-turn multipliers.
Emotional Event Log — append-only record of spikes (cause, turn id, confidence, VAD delta), separate from the live bars. This is what the Memory Writer reads to decide what's worth persisting.
Emotion Manager stays a state-tracker, not a behavior-decider. A separate downstream policy consumes its output to decide response tone — keeping "what does the user feel" cleanly separate from "how should the assistant respond" (the latter also has to route through safety/wellbeing logic).

