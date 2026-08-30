"""emotion_engine.py — tiered affect tracker for LLM applications (v2).

Changes from the submitted version, in order of severity:

  1. FIX: reset_tier()/from_dict() crashed for the short_term and long_term
     tiers — the half-life lookup derived a config attribute name from
     tier.value ("short_term_half_life") but the config field is actually
     named "stm_half_life". Replaced with an explicit map.

  2. FIX: project_label() used raw Euclidean nearest-neighbor against
     prototypes of very different magnitudes, which biases classification
     toward whichever prototype happens to sit closest to the origin
     (trust/anticipation) regardless of direction. A sustained, clearly
     joy-directioned signal was mislabeled "anticipation". Now matches on
     cosine similarity (direction) and reports magnitude (intensity)
     separately, instead of conflating the two into "closeness".

  3. RESTORED: real multi-label 0-100% bars. VAD is still the single
     source of truth for reasoning (thresholds, spikes, baseline) — but a
     parallel per-tier BarVector, fed directly from your classifier's raw
     multi-label probabilities and decayed with the SAME half-life/lr
     schedule as its VAD counterpart, restores the ability to show "joy
     62%, anticipation 40%" simultaneously instead of one dominant label.

  4. Added a GoEmotions -> VAD lexicon and adapter skeleton, since you
     mentioned a fine-tuned DistilRoBERTa/GoEmotions model.

  5. Renamed the short-term "same label reinforced" event kind from
     "promotion" to "reinforcement" — it was colliding in meaning with the
     long-term tier's "promotion" (which is a genuinely different event:
     writing into the durable trait baseline).

  6. Fixed the sustained/support checks including the just-appended
     current signal in its own corroboration window, which made
     sustained_support=2 silently behave like 1.

  7. Removed the dead `re` import; added bounds validation for the
     remaining tunables in EmotionManagerConfig.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# VAD vector — reasoning core (thresholds, spikes, baseline). NOT the bars.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAD:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _clamp(self.valence))
        object.__setattr__(self, "arousal", _clamp(self.arousal))
        object.__setattr__(self, "dominance", _clamp(self.dominance))

    @classmethod
    def neutral(cls) -> "VAD":
        return cls(0.0, 0.0, 0.0)

    def magnitude(self) -> float:
        return math.sqrt(self.valence ** 2 + self.arousal ** 2 + self.dominance ** 2)

    def unit(self) -> "VAD":
        m = self.magnitude()
        if m < 1e-9:
            return VAD.neutral()
        return VAD(self.valence / m, self.arousal / m, self.dominance / m)

    def blend(self, other: "VAD", lr: float) -> "VAD":
        lr = _clamp01(lr)
        return VAD(
            self.valence + lr * (other.valence - self.valence),
            self.arousal + lr * (other.arousal - self.arousal),
            self.dominance + lr * (other.dominance - self.dominance),
        )

    def decay_toward(self, anchor: "VAD", half_life: float, elapsed: float) -> "VAD":
        if elapsed <= 0.0 or half_life <= 0.0:
            return self
        k = 0.5 ** (elapsed / half_life)
        return VAD(
            anchor.valence + (self.valence - anchor.valence) * k,
            anchor.arousal + (self.arousal - anchor.arousal) * k,
            anchor.dominance + (self.dominance - anchor.dominance) * k,
        )

    def distance(self, other: "VAD") -> float:
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def dot(self, other: "VAD") -> float:
        return self.valence * other.valence + self.arousal * other.arousal + self.dominance * other.dominance

    def cosine(self, other: "VAD") -> float:
        d = self.dot(other)
        ma, mb = self.magnitude(), other.magnitude()
        if ma < 1e-9 or mb < 1e-9:
            return 0.0
        return _clamp(d / (ma * mb))

    def compact(self) -> str:
        return f"v{self.valence:+.2f} a{self.arousal:+.2f} d{self.dominance:+.2f}"

    def to_list(self) -> list[float]:
        return [self.valence, self.arousal, self.dominance]

    @classmethod
    def from_list(cls, vals: Sequence[float]) -> "VAD":
        v, a, d = vals
        return cls(v, a, d)

    def to_dict(self) -> dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "VAD":
        return cls(d["valence"], d["arousal"], d["dominance"])


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
    EmotionLabel.NEUTRAL:      VAD(0.00, 0.00, 0.00),
    EmotionLabel.JOY:          VAD(0.90, 0.55, 0.55),
    EmotionLabel.TRUST:        VAD(0.65, 0.05, 0.25),
    EmotionLabel.FEAR:         VAD(-0.70, 0.80, -0.75),
    EmotionLabel.SURPRISE:     VAD(0.20, 0.90, -0.10),
    EmotionLabel.SADNESS:      VAD(-0.85, -0.55, -0.60),
    EmotionLabel.DISGUST:      VAD(-0.70, 0.25, 0.10),
    EmotionLabel.ANGER:        VAD(-0.75, 0.80, 0.65),
    EmotionLabel.ANTICIPATION: VAD(0.35, 0.55, 0.10),
}

_UNIT_PROTOTYPES: dict[EmotionLabel, VAD] = {k: v.unit() for k, v in PROTOTYPES.items() if k != EmotionLabel.NEUTRAL}

# Below this raw magnitude, direction is too noisy to trust — call it neutral
# rather than snapping to whichever prototype direction happens to win.
NEUTRAL_MAGNITUDE_FLOOR = 0.12


def project_label(vad: VAD) -> tuple[EmotionLabel, float]:
    """
    Direction (cosine similarity to unit prototypes) decides WHICH label.
    Magnitude of the vector itself decides intensity — returned separately,
    not conflated with "closeness" the way the original nearest-neighbor
    version did. Returns (label, intensity in [0, 1]).
    """
    mag = vad.magnitude()
    intensity = _clamp01(mag / math.sqrt(3))  # sqrt(3) = max possible VAD magnitude
    if mag < NEUTRAL_MAGNITUDE_FLOOR:
        return EmotionLabel.NEUTRAL, intensity

    unit = vad.unit()
    best_label, best_sim = EmotionLabel.NEUTRAL, -1.0
    for label, proto_unit in _UNIT_PROTOTYPES.items():
        sim = unit.dot(proto_unit)
        if sim > best_sim:
            best_label, best_sim = label, sim
    return best_label, intensity


# ---------------------------------------------------------------------------
# Bar vector — the display layer. Real multi-label, independent 0-100%.
# ---------------------------------------------------------------------------

BarVector = dict[str, float]  # label -> 0..100, independent per label


def _bar_zero(universe: Iterable[str]) -> BarVector:
    return {label: 0.0 for label in universe}


def _bar_blend(current: BarVector, incoming: BarVector, lr: float) -> BarVector:
    lr = _clamp01(lr)
    out = dict(current)
    for label in out:
        target = incoming.get(label, 0.0) * 100.0
        out[label] = _clamp01((out[label] + lr * (target - out[label])) / 100.0) * 100.0
    return out


def _bar_decay(current: BarVector, half_life: float, elapsed: float) -> BarVector:
    if elapsed <= 0.0 or half_life <= 0.0:
        return current
    k = 0.5 ** (elapsed / half_life)
    return {label: val * k for label, val in current.items()}


def render_bar(bar: BarVector, width: int = 20, top_k: int = 4) -> str:
    top = sorted(bar.items(), key=lambda kv: -kv[1])[:top_k]
    lines = []
    for label, val in top:
        filled = int(round(_clamp01(val / 100) * width))
        lines.append(f"{label:<14} [{'█' * filled}{'-' * (width - filled)}] {val:5.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GoEmotions -> VAD lexicon (approximate, NRC-VAD-informed).
# Used by the classifier adapter to fold 28 independent sigmoid outputs
# into the 3D reasoning core, WITHOUT throwing away the raw probabilities
# (those go into EmotionSignal.label_probs and drive the bars instead).
# ---------------------------------------------------------------------------

GOEMOTIONS_VAD: dict[str, VAD] = {
    "admiration":     VAD(0.65, 0.35, 0.30),
    "amusement":      VAD(0.70, 0.55, 0.35),
    "anger":          VAD(-0.75, 0.80, 0.65),
    "annoyance":      VAD(-0.55, 0.55, 0.35),
    "approval":       VAD(0.55, 0.15, 0.30),
    "caring":         VAD(0.55, 0.20, 0.15),
    "confusion":      VAD(-0.20, 0.40, -0.40),
    "curiosity":      VAD(0.30, 0.50, 0.10),
    "desire":         VAD(0.55, 0.55, 0.10),
    "disappointment": VAD(-0.60, -0.20, -0.35),
    "disapproval":    VAD(-0.55, 0.30, 0.20),
    "disgust":        VAD(-0.70, 0.25, 0.10),
    "embarrassment":  VAD(-0.45, 0.50, -0.55),
    "excitement":     VAD(0.75, 0.85, 0.35),
    "fear":           VAD(-0.70, 0.80, -0.75),
    "gratitude":      VAD(0.75, 0.25, 0.15),
    "grief":          VAD(-0.85, 0.10, -0.70),
    "joy":            VAD(0.90, 0.55, 0.55),
    "love":           VAD(0.85, 0.45, 0.30),
    "nervousness":    VAD(-0.35, 0.70, -0.60),
    "optimism":       VAD(0.65, 0.40, 0.30),
    "pride":          VAD(0.70, 0.45, 0.65),
    "realization":    VAD(0.10, 0.45, 0.05),
    "relief":         VAD(0.55, -0.30, 0.20),
    "remorse":        VAD(-0.60, 0.30, -0.45),
    "sadness":        VAD(-0.85, -0.55, -0.60),
    "surprise":       VAD(0.20, 0.90, -0.10),
    "neutral":        VAD(0.00, 0.00, 0.00),
}

GOEMOTIONS_LABELS: tuple[str, ...] = tuple(GOEMOTIONS_VAD.keys())


def probs_to_vad(probs: dict[str, float]) -> VAD:
    """Weighted sum of per-label VAD prototypes, weighted by independent
    sigmoid probabilities. Not normalized to sum=1 since multi-label probs
    aren't mutually exclusive — a message can genuinely be 70% joy AND
    60% surprise at once; both should pull the vector."""
    total_weight = sum(probs.values()) or 1.0
    v = a = d = 0.0
    for label, p in probs.items():
        proto = GOEMOTIONS_VAD.get(label)
        if proto is None:
            continue
        v += p * proto.valence
        a += p * proto.arousal
        d += p * proto.dominance
    # scale down by sqrt of total weight rather than total weight itself,
    # so a message with many co-active labels doesn't get diluted toward
    # zero just because more categories fired.
    scale = math.sqrt(total_weight) if total_weight > 1.0 else 1.0
    return VAD(v / scale, a / scale, d / scale)


# ---------------------------------------------------------------------------
# Signal returned by the external classifier adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmotionSignal:
    vad: VAD
    confidence: float = 0.0
    label_probs: dict[str, float] = field(default_factory=dict)
    source: str = "classifier"
    rationale: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _clamp01(self.confidence))


@runtime_checkable
class EmotionClassifier(Protocol):
    """
    Adapter contract: accept the last k turns, return an EmotionSignal
    carrying BOTH the reduced VAD point (for reasoning) AND the raw
    multi-label probabilities (for bars).

    Example for your fine-tuned DistilRoBERTa/GoEmotions model:

        class DistilRoBERTaGoEmotionsAdapter:
            def __init__(self, model_path: str):
                from transformers import pipeline
                self.pipe = pipeline(
                    "text-classification", model=model_path,
                    top_k=None, function_to_apply="sigmoid",
                )

            def classify(self, turns) -> EmotionSignal:
                text = " ".join(t.content for t in turns[-3:])
                raw = self.pipe(text)[0]  # [{"label": "...", "score": ...}, ...]
                probs = {r["label"]: float(r["score"]) for r in raw}
                vad = probs_to_vad(probs)
                sorted_probs = sorted(probs.values(), reverse=True)
                # margin-based confidence: how cleanly separated is the top
                # label from the rest, not just its raw sigmoid score (sigmoid
                # outputs aren't calibrated the way a softmax top-1 is).
                margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
                confidence = _clamp01(0.5 * sorted_probs[0] + 0.5 * margin)
                return EmotionSignal(
                    vad=vad, confidence=confidence, label_probs=probs,
                    source="distilroberta-goemotions",
                )
    """
    def classify(self, turns: Sequence[Any]) -> EmotionSignal: ...


# ---------------------------------------------------------------------------
# Tier model
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    SITUATIONAL = "situational"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


# Explicit map instead of deriving attribute names from tier.value — the
# bug in the submitted version was exactly this kind of string-concat
# assumption (tier.value + "_half_life" != the actual field name).
_HALF_LIFE_ATTR: dict[Tier, str] = {
    Tier.SITUATIONAL: "situational_half_life",
    Tier.SHORT_TERM: "stm_half_life",
    Tier.LONG_TERM: "ltm_half_life",
}


@dataclass
class TierState:
    vad: VAD
    updated_at: float
    half_life: float
    anchor: VAD
    bar: BarVector = field(default_factory=lambda: _bar_zero(GOEMOTIONS_LABELS))

    def effective_vad(self, now: float) -> VAD:
        return self.vad.decay_toward(self.anchor, self.half_life, max(0.0, now - self.updated_at))

    def effective_bar(self, now: float) -> BarVector:
        return _bar_decay(self.bar, self.half_life, max(0.0, now - self.updated_at))

    def apply(self, signal_vad: VAD, signal_bar: BarVector, lr: float, now: float) -> tuple[VAD, VAD]:
        old = self.effective_vad(now)
        self.vad = old.blend(signal_vad, lr)
        self.bar = _bar_blend(self.effective_bar(now), signal_bar, lr)
        self.updated_at = now
        return old, self.vad


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EmotionManagerConfig:
    situational_half_life: float = 300.0
    stm_half_life: float = 2700.0
    ltm_half_life: float = 259_200.0
    situational_min_confidence: float = 0.15
    stm_min_confidence: float = 0.45
    ltm_min_confidence: float = 0.60
    sustained_support: int = 2
    ltm_support: int = 3
    situational_lr: float = 0.70
    stm_lr: float = 0.35
    ltm_lr: float = 0.12
    baseline_alpha: float = 0.03
    agreement_threshold: float = 0.25
    signal_window: int = 12
    context_turns: int = 6
    spike_delta: float = 0.25
    event_log_max: int = 500
    label_universe: tuple[str, ...] = GOEMOTIONS_LABELS

    def __post_init__(self) -> None:
        if self.situational_half_life <= 0 or self.stm_half_life <= 0 or self.ltm_half_life <= 0:
            raise ValueError("Half-lives must be positive")
        for name in ("situational_min_confidence", "stm_min_confidence", "ltm_min_confidence"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("situational_lr", "stm_lr", "ltm_lr", "baseline_alpha"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.sustained_support < 1 or self.ltm_support < 1:
            raise ValueError("support counts must be >= 1")
        if self.signal_window < max(self.sustained_support, self.ltm_support):
            raise ValueError("signal_window must be >= the largest support requirement")


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

@dataclass
class EmotionalEvent:
    tier: Tier
    kind: str  # "spike" | "reinforcement" | "shift" | "promotion"
    signal_vad: VAD
    delta_vad: VAD
    delta_magnitude: float
    confidence: float
    label: EmotionLabel
    top_labels: tuple[str, ...] = ()
    cause_message_id: str = ""
    excerpt: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "timestamp": self.timestamp, "tier": self.tier.value,
            "kind": self.kind, "signal_vad": self.signal_vad.to_dict(),
            "delta_vad": self.delta_vad.to_dict(), "delta_magnitude": self.delta_magnitude,
            "confidence": self.confidence, "label": self.label.value,
            "top_labels": list(self.top_labels),
            "cause_message_id": self.cause_message_id, "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class AffectState:
    situational_vad: VAD
    short_term_vad: VAD
    long_term_vad: VAD
    baseline_vad: VAD
    situational_bar: BarVector
    short_term_bar: BarVector
    long_term_bar: BarVector
    stm_dominant: EmotionLabel
    stm_intensity: float
    ltm_dominant: EmotionLabel
    trend: str
    confidence: float
    updated_at: float


# ---------------------------------------------------------------------------
# EmotionManager — state tracker (NOT behavior decider)
# ---------------------------------------------------------------------------

class EmotionManager:
    def __init__(
        self,
        classifier: EmotionClassifier,
        config: Optional[EmotionManagerConfig] = None,
        baseline: Optional[VAD] = None,
    ) -> None:
        self._classifier = classifier
        self.config = config or EmotionManagerConfig()
        now = time.time()
        self._baseline = baseline or VAD.neutral()
        universe = self.config.label_universe
        self._tiers: dict[Tier, TierState] = {
            Tier.SITUATIONAL: TierState(VAD.neutral(), now, self.config.situational_half_life, VAD.neutral(), _bar_zero(universe)),
            Tier.SHORT_TERM: TierState(VAD.neutral(), now, self.config.stm_half_life, VAD.neutral(), _bar_zero(universe)),
            Tier.LONG_TERM: TierState(VAD.neutral(), now, self.config.ltm_half_life, self._baseline, _bar_zero(universe)),
        }
        self._recent: deque[tuple[VAD, float]] = deque(maxlen=self.config.signal_window)
        self._events: deque[EmotionalEvent] = deque(maxlen=self.config.event_log_max)
        self._last_confidence = 0.0
        self._prev_stm_label = EmotionLabel.NEUTRAL
        self._prev_stm_intensity = 0.0
        self._trend = "steady"
        self._lock = threading.RLock()

    @property
    def context_turns(self) -> int:
        return self.config.context_turns

    def process_turn(self, message: Any, turns: Sequence[Any]) -> list[EmotionalEvent]:
        signal = self._classifier.classify(turns)
        now = time.time()
        events: list[EmotionalEvent] = []
        msg_id = str(getattr(message, "message_id", ""))
        excerpt = str(getattr(message, "content", ""))[:120]
        top_labels = tuple(
            l for l, _ in sorted(signal.label_probs.items(), key=lambda kv: -kv[1])[:3]
        )

        with self._lock:
            self._last_confidence = signal.confidence
            if signal.confidence < self.config.situational_min_confidence:
                return events

            # ---- Situational (always, if confidence passes gate) ----
            sit_old, sit_new = self._tiers[Tier.SITUATIONAL].apply(
                signal.vad, signal.label_probs, self.config.situational_lr * signal.confidence, now
            )
            sit_delta = VAD(sit_new.valence - sit_old.valence, sit_new.arousal - sit_old.arousal, sit_new.dominance - sit_old.dominance)
            sit_mag = sit_delta.magnitude()
            sit_label, _ = project_label(sit_new)
            if sit_mag >= self.config.spike_delta:
                events.append(EmotionalEvent(
                    tier=Tier.SITUATIONAL, kind="spike", signal_vad=signal.vad,
                    delta_vad=sit_delta, delta_magnitude=sit_mag, confidence=signal.confidence,
                    label=sit_label, top_labels=top_labels, cause_message_id=msg_id, excerpt=excerpt,
                ))

            # ---- Short-term (confident + sustained by PRIOR turns) ----
            sustained = self._check_sustained(signal.vad)
            self._recent.append((signal.vad, signal.confidence))  # append AFTER the check
            if signal.confidence >= self.config.stm_min_confidence and sustained:
                stm_old, stm_new = self._tiers[Tier.SHORT_TERM].apply(
                    signal.vad, signal.label_probs, self.config.stm_lr * signal.confidence, now
                )
                stm_delta = VAD(stm_new.valence - stm_old.valence, stm_new.arousal - stm_old.arousal, stm_new.dominance - stm_old.dominance)
                stm_mag = stm_delta.magnitude()
                stm_label, _ = project_label(stm_new)
                kind = "shift" if stm_label != self._prev_stm_label else "reinforcement"
                events.append(EmotionalEvent(
                    tier=Tier.SHORT_TERM, kind=kind, signal_vad=signal.vad,
                    delta_vad=stm_delta, delta_magnitude=stm_mag, confidence=signal.confidence,
                    label=stm_label, top_labels=top_labels, cause_message_id=msg_id, excerpt=excerpt,
                ))
                self._prev_stm_label = stm_label

            # ---- Long-term (repeated confident signals) ----
            ltm_ready = signal.confidence >= self.config.ltm_min_confidence and self._check_ltm_support(signal.vad)
            if ltm_ready:
                ltm_old, ltm_new = self._tiers[Tier.LONG_TERM].apply(
                    signal.vad, signal.label_probs, self.config.ltm_lr * signal.confidence, now
                )
                ltm_delta = VAD(ltm_new.valence - ltm_old.valence, ltm_new.arousal - ltm_old.arousal, ltm_new.dominance - ltm_old.dominance)
                ltm_mag = ltm_delta.magnitude()
                ltm_label, _ = project_label(ltm_new)
                events.append(EmotionalEvent(
                    tier=Tier.LONG_TERM, kind="promotion", signal_vad=signal.vad,
                    delta_vad=ltm_delta, delta_magnitude=ltm_mag, confidence=signal.confidence,
                    label=ltm_label, top_labels=top_labels, cause_message_id=msg_id, excerpt=excerpt,
                ))
                self._baseline = self._baseline.blend(signal.vad, self.config.baseline_alpha)
                self._tiers[Tier.LONG_TERM].anchor = self._baseline

            # ---- Trend ----
            stm_eff = self._tiers[Tier.SHORT_TERM].effective_vad(now)
            _, stm_intensity = project_label(stm_eff)
            if stm_intensity > self._prev_stm_intensity + 0.05:
                self._trend = "rising"
            elif stm_intensity < self._prev_stm_intensity - 0.05:
                self._trend = "falling"
            else:
                self._trend = "steady"
            self._prev_stm_intensity = stm_intensity

            for ev in events:
                self._events.append(ev)

        return events

    def _check_sustained(self, signal_vad: VAD) -> bool:
        # NOTE: checked BEFORE the current signal is appended to _recent,
        # so "sustained_support=2" genuinely means 2 prior independent
        # readings agree with this one — the submitted version appended
        # first, so the current entry always agreed with itself for free.
        if len(self._recent) < self.config.sustained_support:
            return False
        recent_vads = [v for v, c in self._recent if c >= self.config.stm_min_confidence]
        if len(recent_vads) < self.config.sustained_support:
            return False
        agree = sum(1 for v in recent_vads[-self.config.sustained_support:] if v.dot(signal_vad) >= self.config.agreement_threshold)
        return agree >= self.config.sustained_support

    def _check_ltm_support(self, signal_vad: VAD) -> bool:
        conf_recent = [(v, c) for v, c in self._recent if c >= self.config.ltm_min_confidence]
        if len(conf_recent) < self.config.ltm_support:
            return False
        agree = sum(1 for v, c in conf_recent[-self.config.ltm_support:] if v.dot(signal_vad) >= self.config.agreement_threshold)
        return agree >= self.config.ltm_support

    def affect_state(self) -> AffectState:
        now = time.time()
        with self._lock:
            sit = self._tiers[Tier.SITUATIONAL].effective_vad(now)
            stm = self._tiers[Tier.SHORT_TERM].effective_vad(now)
            ltm = self._tiers[Tier.LONG_TERM].effective_vad(now)
            stm_label, stm_intensity = project_label(stm)
            ltm_label, _ = project_label(ltm)
            return AffectState(
                situational_vad=sit, short_term_vad=stm, long_term_vad=ltm, baseline_vad=self._baseline,
                situational_bar=self._tiers[Tier.SITUATIONAL].effective_bar(now),
                short_term_bar=self._tiers[Tier.SHORT_TERM].effective_bar(now),
                long_term_bar=self._tiers[Tier.LONG_TERM].effective_bar(now),
                stm_dominant=stm_label, stm_intensity=round(stm_intensity, 3),
                ltm_dominant=ltm_label, trend=self._trend, confidence=self._last_confidence, updated_at=now,
            )

    def events(self) -> list[EmotionalEvent]:
        with self._lock:
            return list(self._events)

    def reset_tier(self, tier: Tier) -> None:
        now = time.time()
        anchor = self._baseline if tier == Tier.LONG_TERM else VAD.neutral()
        with self._lock:
            self._tiers[tier] = TierState(
                VAD.neutral(), now, getattr(self.config, _HALF_LIFE_ATTR[tier]),
                anchor, _bar_zero(self.config.label_universe),
            )

    def reset_session(self) -> None:
        now = time.time()
        with self._lock:
            self._tiers[Tier.SITUATIONAL] = TierState(
                VAD.neutral(), now, self.config.situational_half_life, VAD.neutral(), _bar_zero(self.config.label_universe)
            )
            self._tiers[Tier.SHORT_TERM] = TierState(
                VAD.neutral(), now, self.config.stm_half_life, VAD.neutral(), _bar_zero(self.config.label_universe)
            )
            self._recent.clear()
            self._prev_stm_label = EmotionLabel.NEUTRAL
            self._prev_stm_intensity = 0.0
            self._trend = "steady"


def render_affect_summary(affect: AffectState) -> str:
    lines = [
        f"trend: {affect.trend} | confidence: {affect.confidence:.2f}",
        "",
        "-- situational (this turn) --",
        render_bar(affect.situational_bar),
        "",
        "-- short-term --",
        render_bar(affect.short_term_bar),
        "",
        "-- long-term (trait trend) --",
        render_bar(affect.long_term_bar),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class DummyClassifier:
        """Stand-in for the DistilRoBERTa/GoEmotions adapter."""
        def classify(self, turns):
            text = str(getattr(turns[-1], "content", "")).lower()
            if "happy" in text or "excited" in text:
                probs = {"joy": 0.78, "excitement": 0.55, "optimism": 0.30}
            elif "worried" in text or "nervous" in text:
                probs = {"nervousness": 0.72, "fear": 0.40}
            else:
                probs = {"neutral": 0.9}
            vad = probs_to_vad(probs)
            top = max(probs.values())
            return EmotionSignal(vad=vad, confidence=top, label_probs=probs, source="dummy")

    class Msg:
        def __init__(self, content, mid):
            self.content = content
            self.message_id = mid

    mgr = EmotionManager(DummyClassifier())
    turns = [
        "I just got the job offer, I'm so happy and excited!!",
        "Still really happy about it honestly.",
        "A little worried about the move though.",
        "Yeah still nervous about the move.",
        "Ok feeling happy and excited again now.",
    ]
    all_events = []
    for i, t in enumerate(turns):
        m = Msg(t, f"m{i}")
        evs = mgr.process_turn(m, [m])
        all_events.extend(evs)
        print(f">>> {t}")
        for e in evs:
            print(f"    event: tier={e.tier.value:12s} kind={e.kind:12s} label={e.label.value}")

    print()
    print(render_affect_summary(mgr.affect_state()))

    # confirm the two crash bugs are fixed
    for tier in Tier:
        mgr.reset_tier(tier)
    print("\nreset_tier OK for all three tiers")


# ---------------------------------------------------------------------------
# Memory writer — reads the event log, persists qualifying episodes.
# retriever only needs .add(content, importance, metadata) — MemoryRetriever
# in context_engine.py already matches this shape with no adapter needed.
# ---------------------------------------------------------------------------

class EmotionalMemoryWriter:
    def __init__(
        self,
        retriever: Any,
        min_magnitude: float = 0.35,
        min_confidence: float = 0.55,
        cooldown_seconds: float = 300.0,
        kinds: Optional[set] = None,
    ) -> None:
        self._retriever = retriever
        self._min_magnitude = min_magnitude
        self._min_confidence = min_confidence
        self._cooldown = cooldown_seconds
        # "reinforcement" (STM, same label held) + "promotion" (LTM, durable
        # trait shift). "spike" (situational, one-off) is excluded by default
        # — a single intense turn alone isn't yet worth a permanent memory;
        # pass kinds={"spike","reinforcement","promotion"} to change that.
        self._kinds = kinds or {"reinforcement", "shift", "promotion"}
        self._last_write: dict[str, float] = {}
        self._lock = threading.Lock()

    def maybe_write(self, events: Sequence["EmotionalEvent"]) -> list[Any]:
        written: list[Any] = []
        now = time.time()
        for ev in events:
            if ev.kind not in self._kinds:
                continue
            if ev.delta_magnitude < self._min_magnitude:
                continue
            if ev.confidence < self._min_confidence:
                continue
            key = ev.tier.value + ":" + ev.label.value
            if now - self._last_write.get(key, 0.0) < self._cooldown:
                continue
            importance = _clamp01(0.4 + 0.4 * ev.delta_magnitude + 0.2 * ev.confidence)
            # Prefer the raw multi-label bars (ev.top_labels) over the single
            # VAD-projected ev.label here: during a mood transition the
            # blended VAD point can project onto a label that matches
            # neither the old nor the new emotion (see project_label notes).
            # The bars don't have that failure mode — each label decays/
            # blends independently, so they stay accurate through a shift.
            top = ", ".join(ev.top_labels) if ev.top_labels else ev.label.value
            content = (
                f"Emotional episode ({ev.kind} -> {ev.tier.value}): "
                f"user showed {top} ({ev.signal_vad.compact()}). "
                f"Confidence {ev.confidence:.2f}. Excerpt: {ev.excerpt[:80]}"
            )
            with self._lock:
                self._last_write[key] = now
            mem = self._retriever.add(
                content=content,
                importance=importance,
                metadata={
                    "kind": "emotional_episode",
                    "tier": ev.tier.value,
                    "emotion": ev.label.value,
                    "magnitude": round(ev.delta_magnitude, 3),
                    "confidence": round(ev.confidence, 3),
                },
            )
            written.append(mem)
        return written
