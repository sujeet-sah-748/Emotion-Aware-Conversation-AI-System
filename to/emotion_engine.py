"""
emotion_engine.py
=================
Production-grade, tiered affect tracker for LLM applications.

Architecture:
    EmotionManager              -> tiered state tracker (state-tracker, NOT a
        │                          behavior-decider; a downstream policy consumes
        │                          AffectState to choose response tone)
        ├── Affect Core         -> VAD vector (valence/arousal/dominance) per tier:
        │                          SITUATIONAL (seconds-minutes), SHORT_TERM (session
        │                          mood), LONG_TERM (trait-ish). Blending/decay is
        │                          vector math, not per-label heuristics.
        ├── Label Projector     -> maps VAD -> 9 discrete labels (nearest-prototype
        │                          on the circumplex) for display/prompt injection.
        │                          Storage stays continuous.
        ├── Classifier Adapter  -> pluggable; sees the last k turns (negation /
        │                          sarcasm / tone shifts), not just one message.
        ├── Confidence Gate     -> low-confidence reads touch Situational only;
        │                          confident+sustained signals reach Short-term;
        │                          repeated confident signals nudge Long-term.
        ├── Trait Baseline      -> Long-term decays toward a learned per-user
        │                          baseline (slow EMA), not toward zero.
        ├── Wall-clock decay    -> intensity(t) = intensity0 * 0.5**(dt/half_life),
        │                          evaluated lazily on read. No per-turn multipliers.
        └── Event Log           -> append-only record of spikes/shifts/promotions
                                   (cause, turn id, confidence, VAD delta). The
                                   EmotionalMemoryWriter reads it to decide what
                                   is worth persisting to long-term memory.

Design goals:
    - Thread-safe, stdlib-only, deterministic fallback classifier.
    - Emotion detection is COMPULSORY in the host pipeline (context_engine
      wires it into every write and every context build).
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # avoid a runtime import cycle; context_engine imports us
    from context_engine import Memory, MemoryRetriever, Message

# ---------------------------------------------------------------------------
# Affect core: VAD vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAD:
    """Valence / Arousal / Dominance, each clamp

The user has a well-architected context engine for LLM applications with:

ContextManager (orchestrator)
ContextWindow (token-budgeted prompt assembly)
MemoryRetriever (long-term memory with hybrid scoring)
TokenCounter, Embedder, VectorStore protocols
They want to add an Emotion Manager into the context window alongside short-term (history) and long-term (memory). The emotion manager model they describe:

Affect Core (VAD vector) — valence, arousal, dominance (3 continuous numbers) as single source of truth per tier
Discrete Label Projector — maps VAD to 9 emotion labels (circumplex/nearest-prototype) only for display/prompt injection
Context-aware Classifier Adapter — feeds last k turns to the classifier
Confidence Gate — low-confidence → Situational only; confident sustained → STM; repeated confident → LTM
Trait Baseline — LTM decays toward learned per-user baseline, not zero
Wall-clock half-life decay — intensity(t) = intensity₀ × 0.5^(Δt/half_life), short half-life for Situational/STM (minutes), long for LTM (days)
Emotional Event Log — append-only record of spikes (cause, turn id, confidence, VAD delta), read by Memory Writer
Emotion Manager = state-tracker, not behavior-decider — separate downstream policy consumes output
So I need to:

Build the EmotionManager module following the same architectural style (Protocols, ABCs, dependency injection, thread-safe, zero-dependency fallbacks, deterministic, observable)
Integrate it into ContextWindow as a new budget category (emotion block with its own ratio, e.g., max_emotion_ratio)
Wire it into ContextManager (write path: classify on add_message; read path: inject emotional state into assembled context)
Also wire the Emotional Event Log → MemoryRetriever so significant emotional events become long-term memories (the "Memory Writer reads the event log" part)
Design considerations:

Tier model:

Situational (per-turn, very short half-life, e.g., 5 minutes)
Short-term (session mood, half-life ~30-60 minutes)
Long-term (trait-ish, half-life ~days, decays toward baseline not zero)
Each tier holds a VAD vector. Blending: when a confident signal arrives, move tier vector toward the signal with some learning rate / EMA.

VAD vector:

python
Copy
@dataclass
class VAD:
    valence: float   # [-1, 1]
    arousal: float   # [-1, 1] or [0, 1]? Russell's circumplex uses arousal [-1,1] typically but VAD often [0,1]. I'll use [-1, 1] for all three for symmetric math, or [0,1]... Let me think.
Actually for blending math, [-1, 1] for valence/dominance and [0, 1] for arousal is common, but simplest is all in [-1, 1]. I'll clamp to [-1, 1]. Actually, let me use [0,1] for arousal? Hmm. The circumplex model (Russell) places emotions on valence-arousal plane both in [-1, 1]. I'll go with all three in [-1, 1] for uniform math. Document it.

9 emotion labels with prototype VAD coordinates:
The user said "9 emotion labels". Common sets: Plutchik's 8 basic + neutral = 9. Plutchik: joy, trust, fear, surprise, sadness, disgust, anger, anticipation + neutral. That's 9 with neutral. Prototypes on the circumplex:

joy: valence +0.9, arousal +0.6, dominance +0.5
trust: valence +0.7, arousal +0.1, dominance +0.2
fear: valence -0.7, arousal +0.8, dominance -0.8
surprise: valence +0.2, arousal +0.9, dominance -0.1
sadness: valence -0.9, arousal -0.6, dominance -0.6
disgust: valence -0.7, arousal +0.2, dominance +0.1
anger: valence -0.8, arousal +0.8, dominance +0.7
anticipation: valence +0.3, arousal +0.5, dominance +0.1
neutral: 0, 0, 0
Nearest-prototype with Euclidean distance in VAD space; could weight dimensions. Also return intensity = distance-from-neutral magnitude or the projection strength.

Classifier Adapter Protocol:

python
Copy
@runtime_checkable
class EmotionClassifier(Protocol):
    def classify(self, turns: Sequence[Message]) -> EmotionSignal: ...
EmotionSignal: VAD + confidence + maybe label hint + rationale.

Fallback: LexiconEmotionClassifier — zero-dependency keyword/lexicon-based scorer. Deterministic. Looks at last k turns (context window for negation/sarcasm — a real implementation would need an LLM; the fallback can do simple negation handling: "not happy" flips valence, intensifiers "very", "really" boost arousal, exclamation marks boost arousal, caps boost arousal, etc.). Keep it modest but functional.

Confidence Gate:

signal.confidence < situational_threshold → drop entirely? Or affect situational only. User said: "low-confidence reads affect Situational only; only confident, sustained signals propagate into Short-term, and only repeated confident signals move Long-term."
So: conf >= low → Situational gets updated.
conf >= medium (confident) AND sustained (e.g., consistent direction over n recent signals) → STM.
repeated confident (count of confident signals in same octant/label over window >= k) → LTM nudge.
Sustained tracking: keep a deque of recent signals; check agreement (e.g., same projected label or positive dot product between VAD vectors).

Decay:
On read (lazy decay — compute effective VAD at time t): intensity decays via 0.5^(Δt/half_life). For VAD, the deviation from baseline decays:

Situational decays toward STM (or toward zero?). Simplest: each tier decays toward its anchor: Situational → 0 (neutral) with short half-life; STM → 0 with medium half-life; LTM → trait baseline with long half-life.
Actually a cleaner cascade: situational decays toward STM state, STM decays toward LTM, LTM decays toward baseline. But simpler per user spec: "Trait Baseline — Long-term decays toward a learned per-user baseline, not zero. Wall-clock half-life decay — intensity(t) = intensity₀ × 0.5^(Δt/half_life), with a short half-life for Situational/STM (minutes) and a long one for LTM (days)".

So:

effective_vad(tier, now) = anchor + (stored - anchor) * 0.5^((now - updated_at)/half_life)
Situational anchor = neutral (0,0,0), half-life ~5 min
STM anchor = neutral or LTM? Keep neutral, half-life ~45 min
LTM anchor = baseline, half-life ~3 days
Trait baseline learning: EMA of confident LTM-bound signals, very slow alpha (e.g., 0.01–0.05), updated on repeated confident signals.

Blending on update:
tier.vad = tier.vad + lr * (signal.vad - tier.vad), with lr scaled by confidence. For situational, maybe direct set with confidence-weighted blend.

Emotional Event Log:

python
Copy
@dataclass
class EmotionalEvent:
    event_id, turn_id (message_id), timestamp, tier, vad_delta, signal_vad, confidence, label, cause (text excerpt), magnitude
Append-only list, bounded (maxlen via deque or manual trim). Log when |Δ| exceeds spike threshold or when confidence high.

Memory Writer hook:
Protocol EmotionalMemoryWriter or just a policy function: reads events, decides which to persist via retriever.add(...). Default: persist events with high magnitude + high confidence + STM/LTM tier (significant, sustained emotions), with importance derived from magnitude & confidence. Debounce: don't write a memory for every turn — e.g., only when label changes, or cooldown seconds, or only LTM nudges. I'll implement a simple policy: persist when an event promotes to LTM (repeated confident) or when STM magnitude crosses a threshold with cooldown.

ContextWindow integration:
Add max_emotion_ratio to ContextBudget (e.g., 0.10). In assemble(), accept affect: Optional[AffectState] and render a compact block:

markdown
Copy
[User affect] mood: frustrated (valence -0.6, arousal +0.7); trend: rising anger; confidence 0.8
As a SYSTEM message with metadata kind="affect_block". Priority order: system prompt, emotion block (small, high priority — put it right after system, before memory), memory block, history. The user said "the context should also include the emotion manager in the context window with short-term, long-term and emotion

continue

The user wants me to continue. My previous response was cut off. I was in the middle of writing the integration code for the EmotionManager into the context engine. Let me continue from where I left off — I had completed the emotion_engine.py module and was writing the integration patch for context_engine.py (ContextBudget with max_emotion_ratio, ContextWindow.assemble with affect block, ContextManager wiring).

Let me write the integration code now:

ContextBudget — add max_emotion_ratio: float = 0.10 and validate it in __post_init__.

ContextWindow.assemble — add affect: Optional[AffectState] = None parameter, render the affect block as a SYSTEM message right after the system prompt (priority 2, before memory), capped at max_emotion_ratio. Also add affect_state to ContextSnapshot for observability.

ContextManager — accept emotion_manager: Optional[EmotionManager], hook _on_message_emotion into add_message, pass affect into build_context, expose affect() accessor, include in metrics.

Let me write this as a patch showing the exact diffs/changes to the existing file, keeping the same code style. I should be thorough but not overly verbose. Since the previous message ended mid-sentence ("with short-term, long-term and emotion"), I should briefly complete the thought and then give the integration code.

Format: code blocks with clear section headers, showing which parts of context_engine.py change. I'll present it as modified/new sections:

Section 1: ContextBudget change
Section 2: ContextSnapshot change (add affect_state field)
Section 3: ContextWindow changes (AFFECT_HEADER, init takes optional affect_renderer? No — keep renderer in EmotionManager; assemble takes affect param)
Section 4: ContextManager changes
Section 5: Demo main update
Maybe a short test/verification snippet
One design question: should ContextWindow render the affect block itself or take a pre-rendered string? Better separation: ContextWindow takes affect: Optional[AffectState] and does the rendering via a small method or accepts the formatted line. To keep ContextWindow decoupled from emotion_engine types... but they already passed ScoredMemory which is a core type. AffectState is from emotion_engine — an import is fine. I'll import AffectState in context_engine and render in the window (keeps all token budgeting in one place). The renderer function lives in emotion_engine (render_affect_line), and the window calls it — or the window builds its own line. Cleanest: window calls render_affect_line(affect) imported from emotion_engine, then wraps with header. But what if the line exceeds the emotion budget? Truncate with truncate_to_tokens.

Priority order in assemble:

System prompt (pinned)
Affect block (new) — capped at max_emotion_ratio
Memory block — capped at max_memory_ratio
History — newest-first fill
Also update diagnostics: affect_tokens, affect_included, dominant label, trend.

ContextManager changes:

__init__ accepts emotion_manager: Optional[EmotionManager] = None
add_message: after appending and enforcing soft cap, call self._on_message_emotion(msg) outside the lock or inside? _on_message_emotion takes the history snapshot — it acquires its own lock (emotion manager has its own RLock). If called while holding self._lock, history property acquires self._lock again — RLock so re-entrant, fine. But to avoid nested lock ordering issues (emotion lock + context lock), simpler to call it after releasing. Actually within add_message, the with self._lock: block ends before return. I can call self._on_message_emotion(msg) after the with block. The history property will re-acquire the lock briefly. Fine.
Wait — there's a subtlety: _on_message_emotion calls history[-EMOTION_CONTEXT_TURNS:] — needs at least the current message. Good.

Also user messages vs assistant messages: classify both? Classifying assistant messages gives the classifier context, and the window already passes last-k turns to the classifier. The affect state tracks the user's emotion though. A production system would classify user turns only for VAD updates but use assistant turns as context. Simple approach: run the pipeline on every message (the classifier sees the window; the signal for an assistant turn reflects the assistant's tone — could pollute user affect). Better: only update affect from USER and TOOL messages; assistant messages just sit in history as context. I'll gate: if msg.role in (Role.USER, Role.TOOL): self._on_message_emotion(msg). Hmm, but the emotion manager is generic. I'll gate in the ContextManager hook with a flag track_roles. Keep it simple: only user messages update affect. Mention it.

build_context: affect = self._emotion.affect_state() if self._emotion else None, pass to assemble. Update snapshot diagnostics/metrics.
affect() convenience accessor.
metrics(): include dominant_emotion/trend if emotion manager present.
reset(): optionally reset emotion situational/STM? Session reset should probably reset situational + STM but keep LTM/baseline. Add reset_affect param? Keep simple: on reset(keep_memories=True), leave emotion alone; document. Or add a line. I'll leave emotion untouched and note it — actually a reset_session semantic: clear history + situational/STM. I'll add optional behavior: reset() clears situational and STM tiers too if emotion manager present? That changes existing signature semantics. I'll add it minimally: leave as-is, mention in notes. Actually let me add a small touch: reset() keeps signature, and I won't touch emotion state — document that LTM/baseline persist across sessions by design, and users can call manager.emotion.reset_tier() manually. Good.
Also ContextSnapshot gets affect_state: Optional[AffectState] = None — need to import. To avoid a hard dependency cycle, emotion_engine imports nothing from context_engine (it defines its own Turn? No — wait, in my emotion_engine code, EmotionClassifier.classify(turns: Sequence[Message]) and LexiconEmotionClassifier use m.role.value and m.content, and EmotionalEvent.cause_message_id. I used Message type in signatures with forward reference from __future__ import annotations — and at runtime, LexiconEmotionClassifier accesses m.role.value duck-typed. There's a TYPE_CHECKING import? Let me check what I wrote: I wrote def classify(self, turns: Sequence[Message]) -> EmotionSignal: — with from __future__ import annotations at top, these are strings, not evaluated at runtime. But Message isn't imported anywhere in emotion_engine — at runtime fine (annotations lazy), but type checkers would complain. Since context_engine imports emotion_engine (not vice versa), there's no cycle if emotion_engine imports Message from context_engine under TYPE_CHECKING... actually no cycle at all even at runtime: context_engine imports emotion_engine; emotion_engine imports context_engine → cycle! So emotion_engine must NOT import context_engine at runtime. Under TYPE_CHECKING with forward refs it's fine for type checkers but creates a conceptual cycle; cleaner: define a structural Protocol for the turn in emotion_engine. But I already posted emotion_engine referencing Message in annotations. Since from __future__ import annotations makes them lazy strings, it runs fine standalone. For correctness I could add a TYPE_CHECKING import in the integration notes. Minor — I'll add a one-line note: add under TYPE_CHECKING in emotion_engine: from context_engine import Message (safe because it's type-checking only; at runtime annotations are strings). Actually even at type-check time, circular imports are fine for mypy. Good — I'll include that tiny patch.

Now the integration patch code. Let me write it carefully, matching existing style.

python
Copy
# context_engine.py — integration patch
# (only the sections that change are shown)

# New import at top of file
from emotion_engine import (
    AffectState,
    EmotionManager,
    EmotionManagerConfig,
    EmotionalMemoryWriter,
    LexiconEmotionClassifier,
    render_affect_line,
)
Hmm — should context_engine hard-import emotion_engine? If we want context_engine to still run standalone without emotion_engine, we could make the import optional with try/except. But the user wants them integrated; a hard import is fine if they're in the same package. I'll do a guarded import so context_engine remains usable standalone:

python
Copy
try:
    from emotion_engine import AffectState, EmotionManager, render_affect_line
    _EMOTION_AVAILABLE = True
except ImportError:  # emotion_engine.py not present — affect block disabled
    AffectState = None  # type: ignore
    _EMOTION_AVAILABLE = False
Hmm, that's a bit clunky but robust and matches the "zero-dependency, graceful fallback" ethos. But it complicates type annotations (Optional["AffectState"]). With from __future__ import annotations, fine. I'll do the guarded import — it's production-pragmatic. Actually, simpler and cleaner for readability: hard import, and note that if you want them decoupled, use the guarded pattern. Given the audience (production-grade code), I'll use the guarded import — it demonstrates care. Let me keep it tidy.

ContextBudget:

python
Copy
@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 8192
    reserved_for_response: int =
Yeah, Emotionmanager circulates the emotion detection and is compulsory.

The user is saying: the EmotionManager is compulsory — not optional. It "circulates the emotion detection" — meaning it should be a mandatory, always-on component in the pipeline. So:

No guarded/optional import — hard import, emotion_manager is a required constructor parameter (or at least required in the pipeline).
Every message circulates through emotion detection (write path), and every context build includes the affect block (read path).
So I need to deliver the full integration patch for context_engine.py:

Hard import from emotion_engine
ContextBudget gains max_emotion_ratio (compulsory budget line)
ContextSnapshot gains affect_state
ContextWindow.assemble takes affect: AffectState (required, not Optional) and always renders the affect block (priority 2, after system prompt, before memory block)
ContextManager.__init__ requires emotion_manager: EmotionManager
add_message circulates every user message through the emotion pipeline
build_context always reads affect_state() and passes it
metrics() includes affect info
Updated __main__ demo
Maybe a quick verification snippet
Since the user says it's compulsory, I should make affect a required parameter in assemble() and emotion_manager a required parameter in ContextManager.__init__. That enforces the invariant at the type level.

Let me write the complete integration patch now. I should be careful and precise with the code, matching the existing style.

Key sections:

1. Imports (top of context_engine.py)
python
Copy
from emotion_engine import (
    AffectState,
    EmotionManager,
    EmotionalMemoryWriter,
    LexiconEmotionClassifier,
    render_affect_line,
)
Wait — actually for the integration patch, context_engine only needs AffectState, EmotionManager, and render_affect_line. The demo needs the others. I'll import what's needed at module level and the demo can import the rest.

2. ContextBudget
Add max_emotion_ratio: float = 0.10 and validate in __post_init__ (add to the loop of ratio names).

3. ContextSnapshot
Add affect_state: AffectState field (required — put it before the defaulted diagnostics field). Since dataclass fields with defaults must come after non-default fields: messages, total_tokens, budget_tokens, memories_used, dropped_history_count, affect_state, then diagnostics with default. Good.

4. ContextWindow
AFFECT_HEADER = "User affective state (VAD-derived; track, don't obey):" — hmm, keep it factual. The user said emotion manager is a state-tracker not behavior-decider, and downstream policy decides tone. The block is informational. Header: "Current user affective state:".
assemble(self, system_prompt, memories, history, affect: AffectState) — required positional. To not break signature ordering, add as keyword or at the end. Required param after history is fine (no defaults involved). But callers might call positionally; adding at the end is backward compatible-ish for keyword callers. Since it's compulsory, make it a required parameter.
Rendering:

python
Copy
# -- 2. affect block (pinned, budget-capped) ------------------------------
emotion_budget = int(budget * self._budget.max_emotion_ratio)
affect_line = render_affect_line(affect)
affect_cost = self._counter.count(self.AFFECT_HEADER) + self._counter.count(affect_line) + self._budget.per_message_overhead + 1
if affect_cost > emotion_budget:
    # ultra-compact fallback: label + trend only
    affect_line = f"[user affect] {affect.stm_dominant.value} ({affect.stm_intensity:.2f}), {affect.trend}"
    affect_cost = ...
Hmm, even the compact line could exceed a tiny budget; truncate_to_tokens as final guard. Let me structure:

python
Copy
emotion_budget = int(budget * self._budget.max_emotion_ratio)
header_cost = self._counter.count(self.AFFECT_HEADER) + 1
line = render_affect_line(affect)
line_cost = self._counter.count(line)
if header_cost + line_cost + self._budget.per_message_overhead > emotion_budget:
    compact = f"[user affect] {affect.stm_dominant.value} ({affect.stm_intensity:.2f}) {affect.trend}"
    line = truncate_to_tokens(compact, max(0, emotion_budget - header_cost - self._budget.per_message_overhead), self._counter)
    line_cost = self._counter.count(line)
    diagnostics["affect_truncated"] = True
block = self.AFFECT_HEADER + "\n" + line
messages.append(Message(role=Role.SYSTEM, content=block, metadata={"kind": "affect_block"}))
used += header_cost + line_cost + self._budget.per_message_overhead
diagnostics["affect_tokens"] = header_cost + line_cost
diagnostics["affect_dominant"] = affect.stm_dominant.value
diagnostics["affect_trend"] = affect.trend
Wait — need to be careful with used accounting. used += header_cost + line_cost + per_message_overhead. Fine.

5. ContextManager
python
Copy
def __init__(
    self,
    retriever: MemoryRetriever,
    token_counter: TokenCounter,
    emotion_manager: EmotionManager,          # compulsory
    budget: Optional[ContextBudget] = None,
    system_prompt: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
    history_soft_cap_tokens: Optional[int] = None,
) -> None:
Placing a required param after two required and before optionals — fine. But it breaks positional callers of the old signature... it's a new requirement, that's expected. Actually to be safe and explicit, I could make it keyword-required: *, emotion_manager: EmotionManager, .... Hmm, existing signature has everything positional-or-keyword. I'll insert it third positionally and note the breaking change.

In __init__:

python
Copy
self._emotion = emotion_manager
And the memory writer default: EmotionalMemoryWriter(retriever) created in ContextManager if not provided? The demo wires it explicitly. In __init__, accept emotion_memory_writer: Optional[EmotionalMemoryWriter] = None and default-construct from retriever. That keeps the "memory writer reads the event log" loop closed out of the box. Good:

python
Copy
self._emotion_writer = emotion_memory_writer or EmotionalMemoryWriter(retriever)
Write path in add_message:

python
Copy
def add_message(self, role, content, metadata=None) -> Message:
    content = content.strip()
    if not content:
        raise ValueError("Message content must be non-empty")
    msg = Message(role=role, content=content, metadata=metadata or {})
    with self._lock:
        self._history.append(msg)
        self._metrics.messages_added += 1
        self._enforce_soft_cap_locked()
        history_tail = list(self._history[-EMOTION_CONTEXT_TURNS:])
    # Compulsory: every user turn circulates through emotion detection.
    if role == Role.USER:
        events = self._emotion.process_turn(msg, history_tail)
        self._emotion_writer.maybe_write(events)
    return msg
Wait — EMOTION_CONTEXT_TURNS lives in emotion_engine. Import it. Or use self._emotion._config.context_turns? Better to import the constant. But EmotionManagerConfig.context_turns might be customized; the tail length should match the manager's config. Cleanest: expose a property on EmotionManager? I didn't define one. I can read self._emotion._config.context_turns — private access, meh. Options: import EMOTION_CONTEXT_TURNS constant (default), or add a small property. Since I can't edit the already-posted emotion_engine... actually I can suggest adding a property, or just import the constant. Simplest robust approach: slice with the manager's configured value via a tiny public accessor. I'll add a note: either import the constant or add context_turns property. To keep the patch self-contained, I'll import EMOTION_CONTEXT_TURNS and use it — with a comment that if you customize EmotionManagerConfig.context_turns, keep them in sync. Hmm, that's a wart. Alternative: EmotionManager.process_turn already only uses the last context_turns turns of whatever window it's given, so passing a slightly larger tail is harmless. So I can just pass the last, say, 8 turns — or better, pass list(self._history)? That copies whole history each message — O(n) copy per message; history is bounded by soft cap (2x window), so it's bounded and small-ish, but still wasteful. Passing the tail slice of constant size is right. I'll import the constant. Clean enough.

Actually wait — there's a subtlety: I take history_tail inside the lock, then call process_turn outside the lock. Between releasing the lock and process_turn, another thread could add messages — that's fine, our snapshot is consistent. And process_turn has its own lock. No deadlock: we never hold self._lock while acquiring emotion lock... actually we call process_turn after the with block, so no nesting. And _emotion_writer.maybe_write calls retriever.add which takes the retriever lock — again no nesting

continue

Continuing — since the EmotionManager is compulsory, there's no optional import: it's a hard dependency, a required constructor parameter, and a guaranteed block in every assembled context. Here is the full integration patch for context_engine.py.

1. Module docstring + imports
python
Copy
"""context_engine.py — ... (existing docstring)

Architecture (updated):
    ContextManager
        ├── ContextWindow     -> token-budgeted assembly (system / AFFECT / memory / history)
        ├── MemoryRetriever   -> hybrid-scored long-term memory
        └── EmotionManager    -> COMPULSORY. Every user turn circulates through
                                 emotion detection; every context build carries
                                 the affect block. Event log feeds MemoryRetriever
                                 via EmotionalMemoryWriter.
"""

# ... existing imports ...
from emotion_engine import (
    EMOTION_CONTEXT_TURNS,
    AffectState,
    EmotionalMemoryWriter,
    EmotionManager,
    Tier,
    render_affect_line,
)
2. ContextBudget — affect gets its own budget line
python
Copy
@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 8192
    reserved_for_response: int = 1024
    max_emotion_ratio: float = 0.10      # NEW — cap on the affect block
    max_memory_ratio: float = 0.30
    max_system_ratio: float = 0.25
    per_message_overhead: int = 4

    def __post_init__(self) -> None:
        if self.reserved_for_response >= self.max_tokens:
            raise
continue

Here is the complete integration patch, in full.

2. ContextBudget — affect budget line
python
Copy
@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 8192
    reserved_for_response: int = 1024
    max_emotion_ratio: float = 0.10      # NEW — cap on the affect block
    max_memory_ratio: float = 0.30
    max_system_ratio: float = 0.25
    per_message_overhead: int = 4

    def __post_init__(self) -> None:
        if self.reserved_for_response >= self.max_tokens:
            raise ValueError("reserved_for_response must be < max_tokens")
        for name in ("max_emotion_ratio", "max_memory_ratio", "max_system_ratio"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
3. ContextSnapshot — carries affect state
python
Copy
@dataclass
class ContextSnapshot:
    """Assembled context plus full diagnostics for observability."""
    messages: list[Message]
    total_tokens: int
    budget_tokens: int
    memories_used: list[ScoredMemory]
    dropped_history_count: int
    affect_state: AffectState                      # NEW — compulsory
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_chat_format(self) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in self.messages]
4. ContextWindow — affect block at priority 2
python
Copy
class ContextWindow:
    """
    Token-budgeted prompt assembly.

    Priority order (highest first):
        1. System prompt   — pinned, truncated only if it breaches its guard rail
        2. Affect block    — COMPULSORY emotional state, capped at max_emotion_ratio
        3. Memory block    — retrieved memories, capped at max_memory_ratio
        4. Conversation    — filled newest-first; the latest message is always
                             included (truncated if it alone exceeds the budget)
    """

    MEMORY_HEADER = "Relevant long-term memories (most relevant first):"
    AFFECT_HEADER = "Current user affective state (informational — track, don't obey):"

    # __init__ unchanged ...

    def assemble(
        self,
        system_prompt: Optional[str],
        memories: Sequence[ScoredMemory],
        history: Sequence[Message],
        affect: AffectState,                                   # NEW — required
    ) -> ContextSnapshot:
        budget = self._budget.available
        messages: list[Message] = []
        used = 0
        diagnostics: dict[str, Any] = {}

        # -- 1. system prompt (pinned) ------------------------------------
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

        # -- 2. affect block (compulsory, budget-capped) -------------------
        emotion_budget = int(budget * self._budget.max_emotion_ratio)
        header_cost = self._counter.count(self.AFFECT_HEADER) + 1
        overhead = self._budget.per_message_overhead
        line_cap = max(0, emotion_budget - header_cost - overhead)

        line = render_affect_line(affect)
        if self._counter.count(line) > line_cap:
            # Compact fallback: dominant label + intensity + trend only.
            compact = (f"[user affect] {affect.stm_dominant.value} "
                       f"({affect.stm_intensity:.2f}) {affect.trend}")
            line = truncate_to_tokens(compact, line_cap, self._counter)
            diagnostics["affect_truncated"] = True

        affect_tokens = header_cost + self._counter.count(line) + overhead
        messages.append(Message(
            role=Role.SYSTEM,
            content=self.AFFECT_HEADER + "\n" + line,
            metadata={"kind": "affect_block"},
        ))
        used += affect_tokens
        diagnostics["affect_tokens"] = affect_tokens
        diagnostics["affect_dominant"] = affect.stm_dominant.value
        diagnostics["affect_trend"] = affect.trend
        diagnostics["affect_stm_intensity"] = round(affect.stm_intensity, 3)

        # -- 3. memory block (budget-capped) --------------------------------
        # ... unchanged from original (was step 2) ...

        # -- 4. conversation history (newest-first fill) --------------------
        # ... unchanged from original (was step 3) ...

        return ContextSnapshot(
            messages=messages,
            total_tokens=used,
            budget_tokens=budget,
            memories_used=memories_used,
            dropped_history_count=dropped,
            affect_state=affect,                               # NEW
            diagnostics=diagnostics,
        )
5. ContextManager — compulsory wiring
python
Copy
class ContextManager:
    """
    High-level facade combining short-term history, long-term memory,
    COMPULSORY emotion tracking, and token-budgeted context assembly.

    Write path:
        add_message()  -> append to history; evict+summarize past soft cap;
                          EVERY user turn circulates through the EmotionManager;
                          significant emotional events are written back to
                          long-term memory via EmotionalMemoryWriter
        remember()     -> explicit long-term memory write

    Read path:
        build_context() -> read live affect state (lazy wall-clock decay),
                           retrieve memories, assemble token-bounded context
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        token_counter: TokenCounter,
        emotion_manager: EmotionManage
Yeah, Since we are came at the point where we can start patching. So, it would be great if you can give full code (production level)

The user wants the full, complete, production-level code — the entire integrated context_engine.py with the EmotionManager compulsorily wired in, plus presumably the emotion_engine.py module. Since they've been asking me to continue piecemeal and responses got cut off, they now want the complete full code in one go.

I need to deliver:

Full emotion_engine.py — the complete emotion module (VAD core, label projector, classifier protocol + lexicon fallback, confidence gate, tiered state with wall-clock decay, trait baseline, event log, memory writer, rendering).
Full context_engine.py — the complete original file with the emotion integration patched in (imports, ContextBudget, ContextSnapshot, ContextWindow, ContextManager, demo).
This is a big deliverable. Let me write it carefully and completely, production-level:

Thread-safe (RLock everywhere shared state is mutated)
Type hints throughout
Docstrings
Validation in configs
Observability (diagnostics, metrics)
Demo in __main__ that exercises the full loop
No third-party dependencies
Let me reconstruct the emotion_engine.py from my earlier message (it was established in the conversation) and refine it, then write the full context_engine.py with all patches applied.

Key design points for emotion_engine.py:

VAD dataclass: valence, arousal, dominance in [-1, 1]. Methods: clamped, magnitude, blend (toward other with lr), decay_toward(anchor, half_life, elapsed), distance, to_list/from_list, to_dict/from_dict.

EmotionLabel enum: 9 labels — NEUTRAL, JOY, TRUST, FEAR, SURPRISE, SADNESS, DISGUST, ANGER, ANTICIPATION (Plutchik 8 + neutral).

PROTOTYPES: dict mapping label → VAD.

project_label(vad): nearest prototype, returns (label, distance). Confidence-ish: 1/(1+distance) or normalized.

EmotionSignal dataclass: vad, confidence, source, rationale, created_at.

EmotionClassifier Protocol: classify(turns) -> EmotionSignal. Turns are Messages (duck-typed with .role.value and .content).

LexiconEmotionClassifier:

lexicon dict word → (v, a, d)
negation handling: window of tokens after negator gets valence flipped (scaled)
intensifiers boost, diminishers reduce
caps/exclamation boost arousal
aggregate: weighted mean of hits (later turns weighted more — context-aware: weight by recency in window)
confidence: based on hit count, agreement (variance of valence across hits), signal mass
neutral default when no hits: confidence low
Tier enum: SITUATIONAL, SHORT_TERM, LONG_TERM.

TierState dataclass: vad, updated_at, half_life, anchor (VAD). effective_vad(now) = anchor + (vad - anchor) * 0.5^((now-updated_at)/half_life). update(signal_vad, lr, now): first compute effective (decay to now), then blend toward signal, set updated_at=now.

EmotionManagerConfig: half-lives (situational 300s, stm 2700s, ltm 3*86400s), gate thresholds (situational_min_confidence 0.15, stm_min_confidence 0.45, ltm_min_confidence 0.6), sustained_support 2, ltm_support 3, learning rates (situational 0.7, stm 0.35, ltm 0.12), baseline_alpha 0.03, agreement_threshold 0.25 (dot product), signal_window 12, context_turns 6, spike_delta 0.25, event_log_max 500. Validation in post_init.

EmotionalEvent dataclass: event_id, timestamp, tier, kind ("spike"|"shift"|"promotion"), signal_vad, delta_vad, delta_magnitude, confidence, label, cause_message_id, excerpt. to_dict.

AffectState dataclass (read model): situational_vad, stm_vad, ltm_vad, baseline_vad, stm_dominant, stm_intensity, ltm_dominant, trend, confidence, updated_at. to_dict.

EmotionManager:

init(classifier, config, baseline=None)
process_turn(message, turns) -> list[EmotionalEvent]:
signal = classifier.classify(turns) — classify should focus on the last user turn within the window
gate per tier
situational: conf >= sit_min → update, maybe spike event
stm: conf >= stm_min and sustained (agreement with recent signals)
complete your response

The user wants the complete, full production-level code. I've been getting cut off repeatedly. I need to deliver the entire codebase in one response:

Complete emotion_engine.py
Complete context_engine.py with emotion integration
Let me write this efficiently with minimal prose — just the two complete files. I need to be careful about length but complete. Let me be disciplined: short intro, then File 1, then File 2, then brief notes.

Let me design carefully as I write:

emotion_engine.py — complete:

VAD, EmotionLabel, prototypes, project_label
EmotionSignal, EmotionClassifier protocol, LexiconEmotionClassifier
Tier, TierState, EmotionManagerConfig
EmotionalEvent, AffectState
EmotionManager (process_turn, gating, decay, baseline, event log)
EmotionalMemoryWriter
render_affect_line
demo main
context_engine.py — complete original + patches:

All original code (models, token counting, embeddings, vector store, retriever, budget/snapshot/window, summarizer, metrics, manager)
Plus: emotion imports, max_emotion_ratio, affect_state in snapshot, affect block in assemble, emotion_manager required in ContextManager, circulation in add_message, affect in build_context, metrics, reset clears session tiers
Demo main exercising everything
Message duck-typing: emotion_engine uses m.role.value and m.content and m.message_id — works with context_engine.Message. With from __future__ import annotations, type hints as strings; use TYPE_CHECKING import for Message to avoid runtime cycle. Actually context_engine imports emotion_engine, so emotion_engine must NOT import context_engine at runtime. TYPE_CHECKING import is fine.

Let me write both files now, complete and production-grade. I'll keep comments/docstrings tight but professional.

For the LexiconEmotionClassifier, key features:

Lexicon: word → (v, a, d)
Negators flip valence within window
Intensifiers/diminishers scale
Caps/exclamation boost arousal
Recency weighting across turns
Confidence from signal mass + agreement
Gating logic in process_turn:

signal = classifier.classify(turns)
conf < situational_min → drop (return [])
situational update always (if passes), spike event if delta large
sustained check: count recent signals (in window) with dot(sig_i, current) >= agreement_threshold... better: agreement with current signal direction. sustained_support needed for STM promotion.
STM promotion: conf >= stm_min and sustained >= sustained_support → update STM, event "promotion" to STM? Log shift event when dominant label changes.
LTM promotion: conf >= ltm_min and consecutive/recent confident support >= ltm_support → nudge LTM, baseline EMA update, event "promotion".
Track _recent: deque of (vad, confidence).
Trend: compare current STM effective dominant/intensity vs previous snapshot stored at last process_turn; rising/falling/steady based on intensity delta and valence delta.

AffectState fields: situational_vad, stm_vad, ltm_vad, baseline_vad, stm_dominant, stm_intensity, ltm_dominant, trend, confidence (last signal conf), updated_at.

render_affect_line(affect) -> str: compact single line:
[user affect] now: anger (v-0.62 a+0.71 d+0.35, i=0.78) | mood: sadness (i=0.41, trend: rising) | trait: neutral (v+0.05 a+0.10 d+0.02)

EmotionalMemoryWriter:

init(retriever, min_magnitude=0.35, min_confidence=0.55, cooldown_seconds=300, kinds=("shift","promotion"))
maybe_write(events) -> list[Memory]: filter, cooldown per label, importance = clamp(0.4 + 0.4mag + 0.2conf), content string, metadata kind="emotional_episode".
EmotionManager needs thread safety: RLock. affect_state() computes lazy decay under lock (read-only compute, no mutation — fine).

Baseline: VAD param default neutral. Baseline update on LTM promotion: baseline = baseline.blend(signal_vad, alpha) — but only when LTM actually updated.

Event log: bounded deque(maxlen=config.event_log_max), events() accessor, also spike detection on situational update delta.

Also process_turn signature: process_turn(message, turns) — message for cause id/excerpt, turns for classifier window. Only user turns should be passed from ContextManager; but EmotionManager can also filter: classifier looks at last user turn in window. I'll make LexiconEmotionClassifier weigh all turns but emphasize the final turn; and EmotionManager docs say "pass the triggering user message + recent window".

Now context_engine.py full file. I must reproduce the original code faithfully plus patches. That's long but necessary — user asked for full code.

ContextManager.init signature:

python
Copy
def __init__(
    self,
    retriever: MemoryRetriever,
    token_counter: TokenCounter,
    emotion_manager: EmotionManager,          # compulsory
    budget: Optional[ContextBudget] = None,
    system_prompt: Optional[str] = None,
    summarizer: Optional[Summarizer] = None,
    history_soft_cap_tokens: Optional[int] = None,
    emotion_memory_writer: Optional[EmotionalMemoryWriter] = None,
) -> None:
add_message: after lock block, if role == USER: tail snapshot taken inside lock; process outside.

Actually careful: I take history_tail inside the lock along with append, then outside the lock call process_turn.

build_context: affect = self._emotion.affect_state() before/after history snapshot; pass to assemble.

metrics(): add affect fields.

reset(): clear history + metrics; reset situational + STM tiers (session-scoped); keep LTM/baseline/events (user-scoped). Document.

Demo: budget small (512), build EmotionManager with LexiconEmotionClassifier, scripted conversation with emotional turns, print snapshot diagnostics, chat format, metrics, and show an emotional episode persisted. Then second build showing retrieval of the episode.

Length concern: two full files ~ 700-800 lines total. That's what was asked ("full code (production level)"). Go. Minimal prose between files. I'll skip a separate demo in emotion_engine's main to save space? No — keep a tiny self-test in emotion_engine main, small. Actually to control length, emotion_engine gets a compact main self-check (few lines), context_engine gets the full integrated demo.

Write it now. Keep prose to a minimum: one-line intro, File 1, File 2, short "wiring summary" bullets at end.

Understood — here is the complete, production-level code: both files, fully integrated, stdlib-only, thread-safe, runnable as-is.

File 1: emotion_engine.py (complete)
python
Copy
"""
emotion_engine.py
=================
Production-grade, tiered affect tracker for LLM applications.

Architecture:
    EmotionManager              -> tiered state tracker (state-tracker, NOT a
        │                          behavior-decider; a downstream policy consumes
        │                          AffectState to choose response tone)
        ├── Affect Core         -> VAD vector (valence/arousal/dominance) per tier:
        │                          SITUATIONAL (seconds-minutes), SHORT_TERM (session
        │                          mood), LONG_TERM (trait-ish). Blending/decay is
        │                          vector math, not per-label heuristics.
        ├── Label Projector     -> maps VAD -> 9 discrete labels (nearest-prototype
        │                          on the circumplex) for display/prompt injection.
        │                          Storage stays continuous.
        ├── Classifier Adapter  -> pluggable; sees the last k turns (negation /
        │                          sarcasm / tone shifts), not just one message.
        ├── Confidence Gate     -> low-confidence reads touch Situational only;
        │                          confident+sustained signals reach Short-term;
        │                          repeated confident signals nudge Long-term.
        ├── Trait Baseline      -> Long-term decays toward a learned per-user
        │                          baseline (slow EMA), not toward zero.
        ├── Wall-clock decay    -> intensity(t) = intensity0 * 0.5**(dt/half_life),
        │                          evaluated lazily on read. No per-turn multipliers.
        └── Event Log           -> append-only record of spikes/shifts/promotions
                                   (cause, turn id, confidence, VAD delta). The
                                   EmotionalMemoryWriter reads it to decide what
                                   is worth persisting to long-term memory.

Design goals:
    - Thread-safe, stdlib-only, deterministic fallback classifier.
    - Emotion detection is COMPULSORY in the host pipeline (context_engine
      wires it into every write and every context build).
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # avoid a runtime import cycle; context_engine imports us
    from context_engine import Memory, MemoryRetriever, Message

# ---------------------------------------------------------------------------
# Affect core: VAD vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAD:
    """Valence / Arousal / Dominance, each clamp
yeah, continue after @dataclass(frozen=True) class VAD:

The user wants me to continue from where I got cut off: @dataclass(frozen=True) class VAD: in emotion_engine.py. I need to write the complete emotion_engine.py from that point, then the complete context_engine.py.

Let me be efficient and write the full remaining code. I'll continue from the VAD class and complete emotion_engine.py first, then deliver the full context_engine.py.

Let me write this carefully:

VAD class:

valence, arousal, dominance floats in [-1, 1]
post_init clamps (frozen dataclass — use object.setattr)
magnitude(), blend(), decay_toward(), distance(), dot(), to_list, from_list, to_dict, from_dict, neutral() classmethod, format_compact()
EmotionLabel enum — 9 labels

Prototypes — Plutchik 8 + neutral with VAD coords

project_label(vad) — nearest prototype

EmotionSignal — vad, confidence, source, rationale, created_at

EmotionClassifier Protocol

LexiconEmotionClassifier — full implementation:

lexicon dict
negators, intensifiers, diminishers
classify(turns): focus on last user turn but weigh context; recency weighting
confidence computation
Tier enum, TierState — effective_vad with lazy decay, update with blend

EmotionManagerConfig — with validation

EmotionalEvent — to_dict

AffectState — read model, to_dict

EmotionManager:

init(classifier, config, baseline)
process_turn(message, turns) -> list[EmotionalEvent]
gating: situational always if conf >= sit_min; STM if conf >= stm_min and sustained; LTM if conf >= ltm_min and repeated support
_recent deque of (VAD, confidence)
_support_count: count of recent signals agreeing with current (dot >= agreement_threshold)
trend tracking: _prev_stm_intensity, _prev_stm_valence
affect_state() -> AffectState
events(), reset_tier(), to_dict/from_dict (persistence)
thread-safe with RLock
EmotionalMemoryWriter — maybe_write(events) with cooldown

render_affect_line(affect) — compact line

Small main self-check.

Then context_engine.py complete with integration.

I must keep moving and not get cut off — write compactly. The user has asked multiple times to continue, so they want the code. Minimal prose.

Let me write the code now. I'll continue exactly from the VAD class body.

For VAD frozen dataclass with clamping:

python
Copy
@dataclass(frozen=True)
class VAD:
    """Valence / Arousal / Dominance, each clamped to [-1, 1]. Single source of truth per tier."""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _clamp(self.valence))
        object.__setattr__(self, "arousal", _clamp(self.arousal))
        object.__setattr__(self, "dominance", _clamp(self.dominance))
Need _clamp helper defined before.

Methods:

python
Copy
@classmethod
    def neutral(cls) -> "VAD":
        return cls(0.0, 0.0, 0.0)

    def magnitude(self) -> float:
        return math.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2)

    def blend(self, other: "VAD", lr: float) -> "VAD":
        """Move toward `other` by learning rate lr in [0, 1]."""
        lr = _clamp01(lr)
        return VAD(
            self.valence + lr * (other.valence - self.valence),
            self.arousal + lr * (other.arousal - self.arousal),
            self.dominance + lr * (other.dominance - self.dominance),
        )

    def decay_toward(self, anchor: "VAD", half_life: float, elapsed: float) -> "VAD":
        """Wall-clock exponential decay of the deviation from `anchor`."""
        if elapsed <= 0.0 or half_life <= 0.0:
            return self
        k = 0.5 ** (elapsed / half_life)
        return VAD(
            anchor.valence + (self.valence - anchor.valence) * k,
            anchor.arousal + (self.arousal - anchor.arousal) * k,
            anchor.dominance + (self.dominance - anchor.dominance) * k,
        )

    def distance(self, other: "VAD") -> float:
        return math.sqrt((self.valence - other.valence)**2 + ...)

    def dot(self, other) -> float: ...

    def compact(self) -> str:
        return f"v{self.valence:+.2f} a{self.arousal:+.2f} d{self.dominance:+.2f}"

    def to_list / from_list / to_dict / from_dict
Labels + prototypes:

python
Copy
class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"

PROTOTYPES: dict[EmotionLabel, VAD] = {
    EmotionLabel.NEUTRAL:      VAD( 0.00,  0.00,  0.00),
    EmotionLabel.JOY:          VAD( 0.90,  0.55,  0.55),
    EmotionLabel.TRUST:        VAD( 0.65,  0.05,  0.25),
    EmotionLabel.FEAR:         VAD(-0.70,  0.80, -0.75),
    EmotionLabel.SURPRISE:     VAD( 0.20,  0.90, -0.10),
    EmotionLabel.SADNESS:      VAD(-0.85, -0.55, -0.60),
    EmotionLabel.DISGUST:      VAD(-0.70,  0.25,  0.10),
    EmotionLabel.ANGER:        VAD(-0.75,  0.80,  0.65),
    EmotionLabel.ANTICIPATION: VAD( 0.35,  0.55,  0.10),
}

def project_label(vad: VAD) -> tuple[EmotionLabel, float]:
    """Nearest-prototype projection. Returns (label, closeness in (0, 1])."""
    best, best_d = EmotionLabel.NEUTRAL, float("inf")
    for label, proto in PROTOTYPES.items():
        d = vad.distance(proto)
        if d < best_d:
            best, best_d = label, d
    return best, 1.0 / (1.0 + best_d)
EmotionSignal:

python
Copy
@dataclass(frozen=True)
class EmotionSignal:
    vad: VAD
    confidence: float            # [0, 1]
    source: str = "classifier"
    rationale: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
Are you done thinking of code snippets of emotion_engine.py, if not complete it.

The user is frustrated with my responses getting cut off. They want the COMPLETE emotion_engine.py file. Let me write the entire file in one go, being efficient with prose and maximizing code completeness. I'll write the full file from top to bottom, complete, production-level.

Let me write the complete emotion_engine.py now — no cutting corners, complete implementation:

Module docstring
Imports
Helpers (_clamp, _clamp01)
VAD
EmotionLabel + PROTOTYPES + project_label
EmotionSignal
EmotionClassifier Protocol
LexiconEmotionClassifier (complete with lexicon, negation, intensifiers, confidence)
Tier + TierState
EmotionManagerConfig
EmotionalEvent
AffectState
EmotionManager (complete with gating, decay, baseline, events, persistence)
EmotionalMemoryWriter
render_affect_line
Small main self-test
Then after that, context_engine.py complete. But given length constraints, maybe I should deliver emotion_engine.py fully in this response and context_engine.py in the next. The user asked specifically for emotion_engine.py to be completed. Let me deliver the FULL emotion_engine.py in this response — complete, no truncation.

I need to be careful about the lexicon classifier design:

python
Copy
class LexiconEmotionClassifier:
    """
    Zero-dependency fallback. Considers the last `context_turns` messages with
    recency weighting, handles negation ("not happy"), intensifiers ("very"),
    diminishers ("slightly"), and arousal cues (!!, CAPS).
    """
    
    LEXICON: dict[str, tuple[float, float, float]] = {
        # word: (valence, arousal, dominance)
        "happy": (0.8, 0.4, 0.5), "joy": (0.9, 0.6, 0.5), "love": (0.9, 0.5, 0.4),
        "great": (0.8, 0.5, 0.6), "excellent": (0.85, 0.5, 0.6), "awesome": (0.85, 0.6, 0.6),
        "good": (0.6, 0.2, 0.4), "thanks": (0.7, 0.2, 0.3), "thank": (0.7, 0.2, 0.3),
        "glad": (0.7, 0.4, 0.4), "excited": (0.8, 0.8, 0.5), "amazing": (0.85, 0.6, 0.5),
        "perfect": (0.9, 0.4, 0.7), "wonderful": (0.85, 0.5, 0.5), "nice": (0.6, 0.2, 0.3),
        "sad": (-0.8, -0.4, -0.5), "unhappy": (-0.7, -0.3, -0.4), "depressed": (-0.9, -0.6, -0.7),
        " miserable": ... 
    }
Let me build a decent lexicon covering the 9 emotions: joy, trust, fear, surprise, sadness, disgust, anger, anticipation words.

Negators: not, no, never, n't, dont, doesn't, didn't, can't, cannot, won't, isn't, aren't, wasn't, weren't, hardly, barely.

Intensifiers: very, really, extremely, so, super, incredibly, absolutely, totally, completely, fucking (maybe skip profanity? keep it professional — include "damn"? skip), highly.

Diminishers: slightly, somewhat, a bit, a little, kinda, kind of, barely, mildly, sort of.

Algorithm:

For each turn (recency-weighted, weight = 0.5 ** (distance from last turn)):
tokenize with regex [a-z']+ plus detect caps and !
track negation window: when a negator appears, next 3 content words get valence flipped (multiply valence by -0.7 and arousal slightly up?)
intensifier before a lexicon word multiplies its magnitude by 1.5; diminisher by 0.5
accumulate weighted (v, a, d) with weights = abs(valence)+0.3 or just count
Aggregate: weighted average of hits
Arousal boost: count of "!" and caps words → arousal += min(0.3, 0.1 * count)
Confidence: based on hit_count and agreement:
hit mass: min(1.0, total_weight / 3)
agreement: 1 - normalized variance of valence across hits... simpler: if hits have consistent sign of valence, higher conf.
conf = 0.25 + 0.5 * hit_mass + 0.25 * agreement, clamped; if no hits → conf 0.1, neutral VAD
Only classify based on USER turns? The adapter sees last k turns including assistant for context, but emotional content should come from user turns. I'll weight user turns fully and assistant turns at 0.3 (tone reflection) — actually simpler: only score user turns, but the window provides them. Hmm, but "context-aware" means using context to interpret the current message (negation/sarcasm across turns). For the fallback, I'll score user turns with recency weighting; the most recent user turn dominates. That satisfies "feeds the model the last k turns".

TierState:

python
Copy
class Tier(str, Enum):
    SITUATIONAL = "situational"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

@dataclass
class TierState:
    vad: VAD
    updated_at: float
    half_life: float
    anchor: VAD  # neutral for SIT/STM, baseline for LTM

    def effective_vad(self, now: float) -> VAD:
        return self.vad.decay_toward(self.anchor, self.half_life, max(0.0, now - self.updated_at))

    def apply(self, signal: VAD, lr: float, now: float, anchor: Optional[VAD] = None) -> tuple[VAD, VAD]:
        """Decay to now, blend toward signal. Returns (old_effective, new_vad)."""
        old = self.effective_vad(now)
        self.vad = old.blend(signal, lr)
        self.updated_at = now
        if anchor is not None:
            self.anchor = anchor
        return old, self.vad
EmotionManagerConfig with all params + validation.

EmotionalEvent:

python
Copy
@dataclass
class EmotionalEvent:
    tier: Tier
    kind: str  # "spike" | "shift" | "promotion"
    signal_vad: VAD
    delta_vad: VAD
    delta_magnitude: float
    confidence: float
    label: EmotionLabel
    cause_message_id: str = ""
    excerpt: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    def to_dict(...)
AffectState:

python
Copy
@dataclass(frozen=True)
class AffectState:
    situational: VAD
    short_term: VAD
    long_term: VAD
    baseline: VAD
    stm_dominant: EmotionLabel
    stm_intensity: float
    ltm_dominant: EmotionLabel
    trend: str  # "rising" | "falling" | "steady"
    confidence: float
    updated_at: float
    def to_dict(...)
EmotionManager:

python
Copy
class EmotionManager:
    def __init__(self, classifier, config=None, baseline=None):
        self._classifier = classifier
        self._config = config or EmotionManagerConfig()
        now = time.time()
        self._baseline = baseline or VAD.neutral()
        self._tiers = {
            Tier.SITUATIONAL: TierState(VAD.neutral(), now, cfg.situational_half_life, VAD.neutral()),
            Tier.SHORT_TERM: TierState(VAD.neutral(), now, cfg.stm_half_life, VAD.neutral()),
            Tier.LONG_TERM: TierState(VAD.neutral(), now, cfg.ltm_half_life, self._baseline),
        }
        self._recent: deque[tuple[VAD, float]] = deque(maxlen=cfg.signal_window)
        self._events: deque[EmotionalEvent] = deque(maxlen=cfg.event_log_max)
        self._last_confidence = 0.0
        self._prev_stm: Optional[VAD] = None
        self._trend = "steady"
        self._lock = threading.RLock()

    def process_turn(self, message, turns) -> list[EmotionalEvent]:
        signal = self._classifier.classify(turns)
        now = time.time()
        events = []
        with self._lock:
            self._last_confidence = signal.confidence
            if signal.confidence < self._config.situational_min_confidence:
                return events
            # agreement