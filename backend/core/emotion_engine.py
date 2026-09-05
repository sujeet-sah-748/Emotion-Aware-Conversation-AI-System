"""
emotion_engine_final.py
=======================
Production-grade tiered affect tracker — final version synthesised from
emotion_engine (1).py and (2).py, with the real LoRA/DistilRoBERTa
final_adapter wired as the classifier instead of the dummy stand-in.

What was taken from each source file
--------------------------------------
(1).py  — all bug fixes (half-life attr map, cosine projection, sustained
           check ordering, event kind naming, dead import removal, config
           validation).
(2).py  — everything in (1) PLUS: EmotionalMemoryWriter, context_turns
           property on EmotionManager.
emotion_engine.py (the design transcript) — the GoEmotions→VAD lexicon
           coefficients and the margin-based confidence formula for sigmoid
           outputs (documented in the EmotionClassifier docstring).

New in this file
----------------
FinalAdapterClassifier
    The real classifier.  Loads the LoRA adapter from ./final_adapter at
    construction time (same path resolution as main.py / response_generator.py)
    and runs inference on the last context_turns user messages.

    Key design decisions
    ~~~~~~~~~~~~~~~~~~~~~
    * Concatenates the last ≤3 user turns with [SEP] so the model sees
      conversational context, not just the most recent message.
    * Uses ONLY the raw sigmoid scores from the 28-label head — does NOT
      apply any remapping or threshold that would conflict with the model's
      own calibration.
    * Converts the 28 independent probabilities to a 3D VAD point via
      probs_to_vad() (weighted sum, sqrt-normalised to avoid dilution when
      many labels fire simultaneously).
    * Computes margin-based confidence: 0.5 * top_score + 0.5 * margin
      (top − second).  Raw sigmoid scores are NOT calibrated the way a
      softmax top-1 is, so top_score alone over-reports confidence when
      many labels are moderately active.
    * Raises ImportError at construction if transformers/peft are absent;
      falls back gracefully at classify() time if the model fails at
      runtime (returns a low-confidence neutral signal instead of crashing
      the conversation loop).

sensitivity_halflife.py compatibility
--------------------------------------
All names that sensitivity_halflife.py imports from emotion_engine are
preserved unchanged:
    EmotionManager, EmotionManagerConfig, EmotionSignal, VAD, project_label
It can be run as:
    import emotion_engine_final as ee
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# VAD — valence / arousal / dominance, each in [-1, 1].
# Single source of truth for reasoning (thresholds, spikes, baseline).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAD:
    """
    Valence / Arousal / Dominance, each clamped to [-1, 1].

    Coordinate convention (consistent with NRC-VAD and most affect literature):
        valence   : negative (-1) ... positive (+1)
        arousal   : calm     (-1) ... excited  (+1)
        dominance : weak     (-1) ... dominant (+1)
    """
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence",   _clamp(self.valence))
        object.__setattr__(self, "arousal",   _clamp(self.arousal))
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
        """Move self toward other by learning rate lr ∈ [0, 1]."""
        lr = _clamp01(lr)
        return VAD(
            self.valence   + lr * (other.valence   - self.valence),
            self.arousal   + lr * (other.arousal   - self.arousal),
            self.dominance + lr * (other.dominance - self.dominance),
        )

    def decay_toward(self, anchor: "VAD", half_life: float, elapsed: float) -> "VAD":
        """
        Wall-clock exponential decay of the deviation from anchor:
            effective = anchor + (self - anchor) * 0.5^(elapsed / half_life)
        Evaluated lazily on read; no per-turn multipliers.
        """
        if elapsed <= 0.0 or half_life <= 0.0:
            return self
        k = 0.5 ** (elapsed / half_life)
        return VAD(
            anchor.valence   + (self.valence   - anchor.valence)   * k,
            anchor.arousal   + (self.arousal   - anchor.arousal)   * k,
            anchor.dominance + (self.dominance - anchor.dominance) * k,
        )

    def distance(self, other: "VAD") -> float:
        return math.sqrt(
            (self.valence   - other.valence)   ** 2 +
            (self.arousal   - other.arousal)   ** 2 +
            (self.dominance - other.dominance) ** 2
        )

    def dot(self, other: "VAD") -> float:
        return (self.valence * other.valence +
                self.arousal * other.arousal +
                self.dominance * other.dominance)

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
        return cls(float(v), float(a), float(d))

    def to_dict(self) -> dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "VAD":
        return cls(d["valence"], d["arousal"], d["dominance"])


# ---------------------------------------------------------------------------
# Emotion labels and prototypes
# ---------------------------------------------------------------------------

class EmotionLabel(str, Enum):
    """9 Plutchik-inspired labels used for display and prompt injection.
    Storage and reasoning stay in continuous VAD space."""
    NEUTRAL      = "neutral"
    JOY          = "joy"
    TRUST        = "trust"
    FEAR         = "fear"
    SURPRISE     = "surprise"
    SADNESS      = "sadness"
    DISGUST      = "disgust"
    ANGER        = "anger"
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

# Pre-computed unit vectors for cosine matching (exclude NEUTRAL — the
# magnitude floor check handles it before we ever reach cosine matching).
_UNIT_PROTOTYPES: dict[EmotionLabel, VAD] = {
    k: v.unit() for k, v in PROTOTYPES.items() if k != EmotionLabel.NEUTRAL
}

# Below this magnitude, direction is too noisy; project to NEUTRAL rather
# than snapping to whichever prototype direction happens to win.
NEUTRAL_MAGNITUDE_FLOOR: float = 0.12


def project_label(vad: VAD) -> tuple[EmotionLabel, float]:
    """
    Direction (cosine similarity to unit prototypes) decides WHICH label.
    Magnitude of the input vector decides intensity — returned separately
    so the two are never conflated (the original nearest-neighbour version
    conflated them, mislabelling sustained joy as anticipation because the
    trust/anticipation prototypes sit closest to the origin).

    Returns
    -------
    label     : EmotionLabel
    intensity : float in [0, 1]  (0 = neutral magnitude, 1 = max possible)
    """
    mag = vad.magnitude()
    # sqrt(3) ≈ 1.732 is the maximum possible VAD magnitude (all three
    # dimensions simultaneously at ±1).
    intensity = _clamp01(mag / math.sqrt(3))
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
# Bar vector — display layer, real multi-label 0-100 % bars.
# VAD is the reasoning core; bars are the presentation layer.
# Both are maintained per tier with the same half-life schedule.
# ---------------------------------------------------------------------------

BarVector = dict[str, float]   # label -> percentage in [0, 100]


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
    """ASCII bar chart of the top_k emotions by percentage."""
    top = sorted(bar.items(), key=lambda kv: -kv[1])[:top_k]
    lines = []
    for label, val in top:
        filled = int(round(_clamp01(val / 100.0) * width))
        lines.append(
            f"{label:<14} [{'█' * filled}{'-' * (width - filled)}] {val:5.1f}%"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GoEmotions 28-label → VAD mapping
# Approximate NRC-VAD-informed coordinates for folding 28 independent
# sigmoid outputs into the 3D reasoning core WITHOUT discarding the raw
# probabilities (those drive the bars via EmotionSignal.label_probs).
# ---------------------------------------------------------------------------

GOEMOTIONS_VAD: dict[str, VAD] = {
    "admiration":     VAD( 0.65,  0.35,  0.30),
    "amusement":      VAD( 0.70,  0.55,  0.35),
    "anger":          VAD(-0.75,  0.80,  0.65),
    "annoyance":      VAD(-0.55,  0.55,  0.35),
    "approval":       VAD( 0.55,  0.15,  0.30),
    "caring":         VAD( 0.55,  0.20,  0.15),
    "confusion":      VAD(-0.20,  0.40, -0.40),
    "curiosity":      VAD( 0.30,  0.50,  0.10),
    "desire":         VAD( 0.55,  0.55,  0.10),
    "disappointment": VAD(-0.60, -0.20, -0.35),
    "disapproval":    VAD(-0.55,  0.30,  0.20),
    "disgust":        VAD(-0.70,  0.25,  0.10),
    "embarrassment":  VAD(-0.45,  0.50, -0.55),
    "excitement":     VAD( 0.75,  0.85,  0.35),
    "fear":           VAD(-0.70,  0.80, -0.75),
    "gratitude":      VAD( 0.75,  0.25,  0.15),
    "grief":          VAD(-0.85,  0.10, -0.70),
    "joy":            VAD( 0.90,  0.55,  0.55),
    "love":           VAD( 0.85,  0.45,  0.30),
    "nervousness":    VAD(-0.35,  0.70, -0.60),
    "optimism":       VAD( 0.65,  0.40,  0.30),
    "pride":          VAD( 0.70,  0.45,  0.65),
    "realization":    VAD( 0.10,  0.45,  0.05),
    "relief":         VAD( 0.55, -0.30,  0.20),
    "remorse":        VAD(-0.60,  0.30, -0.45),
    "sadness":        VAD(-0.85, -0.55, -0.60),
    "surprise":       VAD( 0.20,  0.90, -0.10),
    "neutral":        VAD( 0.00,  0.00,  0.00),
}

GOEMOTIONS_LABELS: tuple[str, ...] = tuple(GOEMOTIONS_VAD.keys())


def probs_to_vad(probs: dict[str, float]) -> VAD:
    """
    Weighted sum of per-label VAD prototypes using independent sigmoid probs.

    NOT normalised to sum=1 — multi-label probs are not mutually exclusive,
    so a message can be 70 % joy AND 60 % surprise simultaneously; both
    should pull the vector.  We divide by sqrt(total_weight) rather than
    total_weight so that a signal with many co-active labels doesn't get
    diluted toward neutral just because more categories fired.
    """
    total_weight = sum(probs.values()) or 1.0
    v = a = d = 0.0
    for label, p in probs.items():
        proto = GOEMOTIONS_VAD.get(label)
        if proto is None:
            continue
        v += p * proto.valence
        a += p * proto.arousal
        d += p * proto.dominance
    scale = math.sqrt(total_weight) if total_weight > 1.0 else 1.0
    return VAD(v / scale, a / scale, d / scale)


# ---------------------------------------------------------------------------
# EmotionSignal — returned by the classifier adapter
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
    Adapter contract.  Accepts the last k conversation turns (any objects
    with .content: str), returns an EmotionSignal carrying BOTH:
        vad         — 3D reasoning point derived from probs_to_vad()
        label_probs — raw per-label probabilities [0, 1] that drive the bars
    """
    def classify(self, turns: Sequence[Any]) -> EmotionSignal: ...


# ---------------------------------------------------------------------------
# FinalAdapterClassifier — the REAL classifier using your LoRA weights
# ---------------------------------------------------------------------------

class FinalAdapterClassifier:
    """
    Loads the fine-tuned DistilRoBERTa + LoRA adapter from ./final_adapter
    and classifies the last context_turns user messages.

    Confidence formula
    ------------------
    Raw sigmoid outputs are NOT calibrated like a softmax top-1 — the model
    can output 0.95 for several labels at once.  We use a margin-based score
    instead:
        confidence = 0.5 * top_score + 0.5 * (top_score - second_score)
    This rewards a clear winner and penalises a flat distribution.

    Text input
    ----------
    Concatenates the text of the last min(turns, max_context_turns) turns
    that carry meaningful content (non-empty after strip), joined by
    " [SEP] ", truncated to max_chars before tokenisation to avoid OOM.

    Runtime fallback
    ----------------
    If inference fails for any reason (CUDA OOM, corrupted input, etc.),
    returns a low-confidence neutral signal so the conversation continues.
    """

    # Exact 28 GoEmotions labels in the training order used by the adapter.
    # Must match the id2label in adapter_config.json — misalignment here
    # would silently produce wrong probabilities on every call.
    LABEL_NAMES: tuple[str, ...] = GOEMOTIONS_LABELS

    def __init__(
        self,
        adapter_dir: Optional[str] = None,
        max_context_turns: int = 3,
        max_chars: int = 512,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            import numpy as np
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "FinalAdapterClassifier requires torch, transformers, and peft. "
                f"Original error: {exc}"
            ) from exc

        self._torch = torch
        self._np = np
        self._max_context_turns = max_context_turns
        self._max_chars = max_chars

        if adapter_dir is None:
            # Adapter is in backend/final_adapter (parent of core/)
            adapter_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_adapter")

        base_model_name = "j-hartmann/emotion-english-distilroberta-base"
        id2label = {i: name for i, name in enumerate(self.LABEL_NAMES)}
        label2id = {name: i for i, name in enumerate(self.LABEL_NAMES)}

        self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(self.LABEL_NAMES),
            ignore_mismatched_sizes=True,
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )
        self._model = PeftModel.from_pretrained(base, adapter_dir)
        self._model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(self._device)
        self._lock = threading.Lock()  # one inference at a time (CPU-bound)

    @staticmethod
    def _sigmoid(x: Any) -> Any:
        import numpy as np
        return 1.0 / (1.0 + np.exp(-x))

    def classify(self, turns: Sequence[Any]) -> EmotionSignal:
        # Extract text from the last max_context_turns turns
        texts: list[str] = []
        for t in turns[-self._max_context_turns:]:
            content = str(getattr(t, "content", t)).strip()
            if content:
                texts.append(content)

        if not texts:
            return EmotionSignal(
                vad=VAD.neutral(), confidence=0.0,
                label_probs={}, source="final_adapter",
                rationale="empty input",
            )

        # RoBERTa's separator token is </s>, not [SEP] (that's BERT).
        # Using the tokenizer's own sep_token ensures proper segment marking.
        sep = self._tokenizer.sep_token or "</s>"
        combined = f" {sep} ".join(texts)
        if len(combined) > self._max_chars:
            combined = combined[-self._max_chars:]   # keep the most recent content

        try:
            with self._lock:
                enc = self._tokenizer(
                    combined,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)

                with self._torch.no_grad():
                    logits = self._model(**enc).logits.squeeze(0).cpu().numpy()

            probs_arr = self._sigmoid(logits)
            probs: dict[str, float] = {
                label: float(probs_arr[i])
                for i, label in enumerate(self.LABEL_NAMES)
            }

            sorted_scores = sorted(probs.values(), reverse=True)
            top_score  = sorted_scores[0]
            second     = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            margin     = top_score - second
            confidence = _clamp01(0.5 * top_score + 0.5 * margin)

            vad = probs_to_vad(probs)

            return EmotionSignal(
                vad=vad,
                confidence=confidence,
                label_probs=probs,
                source="final_adapter",
            )

        except Exception as exc:
            # Runtime fallback — don't crash the conversation loop
            return EmotionSignal(
                vad=VAD.neutral(),
                confidence=0.0,
                label_probs={},
                source="final_adapter",
                rationale=f"inference error: {exc}",
            )


# ---------------------------------------------------------------------------
# Tier model
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    SITUATIONAL = "situational"
    SHORT_TERM  = "short_term"
    LONG_TERM   = "long_term"


# Explicit map — deriving attribute names from tier.value at runtime was
# the crash bug in the original submission (tier.value + "_half_life"
# produced "short_term_half_life" but the config field is "stm_half_life").
_HALF_LIFE_ATTR: dict[Tier, str] = {
    Tier.SITUATIONAL: "situational_half_life",
    Tier.SHORT_TERM:  "stm_half_life",
    Tier.LONG_TERM:   "ltm_half_life",
}


@dataclass
class TierState:
    vad:        VAD
    updated_at: float
    half_life:  float
    anchor:     VAD
    bar:        BarVector = field(default_factory=lambda: _bar_zero(GOEMOTIONS_LABELS))

    def effective_vad(self, now: float) -> VAD:
        return self.vad.decay_toward(
            self.anchor, self.half_life, max(0.0, now - self.updated_at)
        )

    def effective_bar(self, now: float) -> BarVector:
        return _bar_decay(self.bar, self.half_life, max(0.0, now - self.updated_at))

    def apply(
        self,
        signal_vad: VAD,
        signal_bar: BarVector,
        lr: float,
        now: float,
    ) -> tuple[VAD, VAD]:
        """Decay to now, then blend toward the incoming signal.
        Returns (effective_vad_before_blend, new_vad)."""
        old      = self.effective_vad(now)
        self.vad = old.blend(signal_vad, lr)
        self.bar = _bar_blend(self.effective_bar(now), signal_bar, lr)
        self.updated_at = now
        return old, self.vad


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmotionManagerConfig:
    # Half-lives in seconds
    situational_half_life: float = 300.0       # ~5 min
    stm_half_life:         float = 2_700.0     # ~45 min
    ltm_half_life:         float = 259_200.0   # ~3 days

    # Confidence gates
    situational_min_confidence: float = 0.15
    stm_min_confidence:         float = 0.45
    ltm_min_confidence:         float = 0.60

    # Support requirements
    sustained_support: int = 2   # prior signals needed for STM update
    ltm_support:       int = 3   # prior signals needed for LTM nudge

    # Learning rates
    situational_lr: float = 0.70
    stm_lr:         float = 0.35
    ltm_lr:         float = 0.12

    # Trait baseline EMA alpha (very slow — long-term personality drift)
    baseline_alpha: float = 0.03

    # Sustained-agreement threshold (dot product between VAD vectors)
    agreement_threshold: float = 0.25

    # Rolling window for sustained/LTM checks
    signal_window: int = 12

    # How many turns to pass to the classifier adapter
    context_turns: int = 6

    # Minimum VAD delta to log as a "spike" event
    spike_delta: float = 0.25

    # Maximum events to retain in the in-memory event log
    event_log_max: int = 500

    # Label universe for bar vectors
    label_universe: tuple[str, ...] = GOEMOTIONS_LABELS

    def __post_init__(self) -> None:
        for attr in ("situational_half_life", "stm_half_life", "ltm_half_life"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be positive")
        for attr in ("situational_min_confidence", "stm_min_confidence", "ltm_min_confidence"):
            if not 0.0 <= getattr(self, attr) <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1]")
        for attr in ("situational_lr", "stm_lr", "ltm_lr", "baseline_alpha"):
            if not 0.0 < getattr(self, attr) <= 1.0:
                raise ValueError(f"{attr} must be in (0, 1]")
        if self.sustained_support < 1 or self.ltm_support < 1:
            raise ValueError("support counts must be >= 1")
        if self.signal_window < max(self.sustained_support, self.ltm_support):
            raise ValueError("signal_window must be >= the largest support requirement")


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

@dataclass
class EmotionalEvent:
    tier:            Tier
    kind:            str          # "spike" | "reinforcement" | "shift" | "promotion"
    signal_vad:      VAD
    delta_vad:       VAD
    delta_magnitude: float
    confidence:      float
    label:           EmotionLabel
    top_labels:      tuple[str, ...] = ()
    cause_message_id: str = ""
    excerpt:          str = ""
    event_id:  str   = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":         self.event_id,
            "timestamp":        self.timestamp,
            "tier":             self.tier.value,
            "kind":             self.kind,
            "signal_vad":       self.signal_vad.to_dict(),
            "delta_vad":        self.delta_vad.to_dict(),
            "delta_magnitude":  self.delta_magnitude,
            "confidence":       self.confidence,
            "label":            self.label.value,
            "top_labels":       list(self.top_labels),
            "cause_message_id": self.cause_message_id,
            "excerpt":          self.excerpt,
        }


@dataclass(frozen=True)
class AffectState:
    situational_vad: VAD
    short_term_vad:  VAD
    long_term_vad:   VAD
    baseline_vad:    VAD
    situational_bar: BarVector
    short_term_bar:  BarVector
    long_term_bar:   BarVector
    stm_dominant:    EmotionLabel
    stm_intensity:   float
    ltm_dominant:    EmotionLabel
    trend:           str
    confidence:      float
    updated_at:      float


# ---------------------------------------------------------------------------
# EmotionManager — state tracker (NOT a behavior decider)
# ---------------------------------------------------------------------------

class EmotionManager:
    """
    Tiered affect tracker.  Classifies each user turn and updates three
    independent VAD vectors with different time constants:

        Situational  — this turn (half-life ~5 min)
        Short-term   — session mood (half-life ~45 min)
        Long-term    — trait baseline (half-life ~3 days, decays toward
                        a learned per-user baseline, not toward zero)

    The resulting AffectState is a read-only snapshot consumed by
    downstream policy (response tone, memory retrieval weighting, etc.).
    EmotionManager decides NOTHING about response content.
    """

    def __init__(
        self,
        classifier: EmotionClassifier,
        config: Optional[EmotionManagerConfig] = None,
        baseline: Optional[VAD] = None,
    ) -> None:
        self._classifier = classifier
        self.config      = config or EmotionManagerConfig()
        now              = time.time()
        self._baseline   = baseline or VAD.neutral()
        universe         = self.config.label_universe

        self._tiers: dict[Tier, TierState] = {
            Tier.SITUATIONAL: TierState(
                VAD.neutral(), now, self.config.situational_half_life,
                VAD.neutral(), _bar_zero(universe),
            ),
            Tier.SHORT_TERM: TierState(
                VAD.neutral(), now, self.config.stm_half_life,
                VAD.neutral(), _bar_zero(universe),
            ),
            Tier.LONG_TERM: TierState(
                VAD.neutral(), now, self.config.ltm_half_life,
                self._baseline, _bar_zero(universe),
            ),
        }

        self._recent: deque[tuple[VAD, float]] = deque(maxlen=self.config.signal_window)
        self._events: deque[EmotionalEvent]    = deque(maxlen=self.config.event_log_max)
        self._last_confidence   = 0.0
        self._prev_stm_label    = EmotionLabel.NEUTRAL
        self._prev_stm_intensity = 0.0
        self._trend             = "steady"
        self._lock              = threading.RLock()

    @property
    def context_turns(self) -> int:
        """How many turns to slice from history before calling the classifier."""
        return self.config.context_turns

    def process_turn(self, message: Any, turns: Sequence[Any]) -> list[EmotionalEvent]:
        """
        Classify the turn window, gate by confidence, and update whichever
        tiers the signal qualifies for.  Returns the list of EmotionalEvents
        fired this turn (may be empty for low-confidence / non-emotional turns).

        Parameters
        ----------
        message : Any
            The triggering message.  Needs .message_id (str) and .content (str)
            for event logging; other attributes are ignored.
        turns : Sequence[Any]
            The sliding window of recent messages (including `message`) passed
            to the classifier.  Slicing to the right length is the caller's
            responsibility (typically history[-context_turns:]).
        """
        signal  = self._classifier.classify(turns)
        now     = time.time()
        events: list[EmotionalEvent] = []
        msg_id  = str(getattr(message, "message_id", ""))
        excerpt = str(getattr(message, "content",    ""))[:120]
        top_labels = tuple(
            lbl for lbl, _ in
            sorted(signal.label_probs.items(), key=lambda kv: -kv[1])[:3]
        )

        with self._lock:
            self._last_confidence = signal.confidence
            if signal.confidence < self.config.situational_min_confidence:
                # Signal too weak to affect any tier; still append to _recent
                # so the window reflects true signal cadence.
                self._recent.append((signal.vad, signal.confidence))
                return events

            # -- Situational (always passes if confidence gate is met) -------
            sit_old, sit_new = self._tiers[Tier.SITUATIONAL].apply(
                signal.vad, signal.label_probs,
                self.config.situational_lr * signal.confidence, now,
            )
            sit_delta = VAD(
                sit_new.valence   - sit_old.valence,
                sit_new.arousal   - sit_old.arousal,
                sit_new.dominance - sit_old.dominance,
            )
            sit_mag   = sit_delta.magnitude()
            sit_label, _ = project_label(sit_new)
            if sit_mag >= self.config.spike_delta:
                events.append(EmotionalEvent(
                    tier=Tier.SITUATIONAL, kind="spike",
                    signal_vad=signal.vad, delta_vad=sit_delta, delta_magnitude=sit_mag,
                    confidence=signal.confidence, label=sit_label, top_labels=top_labels,
                    cause_message_id=msg_id, excerpt=excerpt,
                ))

            # -- Short-term (confident + sustained by PRIOR turns) -----------
            # IMPORTANT: _check_sustained is called BEFORE appending the
            # current signal.  sustained_support=2 means 2 *prior* readings
            # agree — the original submission appended first, so the current
            # entry counted toward its own corroboration window.
            sustained = self._check_sustained(signal.vad)
            self._recent.append((signal.vad, signal.confidence))

            if signal.confidence >= self.config.stm_min_confidence and sustained:
                stm_old, stm_new = self._tiers[Tier.SHORT_TERM].apply(
                    signal.vad, signal.label_probs,
                    self.config.stm_lr * signal.confidence, now,
                )
                stm_delta = VAD(
                    stm_new.valence   - stm_old.valence,
                    stm_new.arousal   - stm_old.arousal,
                    stm_new.dominance - stm_old.dominance,
                )
                stm_mag   = stm_delta.magnitude()
                stm_label, _ = project_label(stm_new)
                kind = "shift" if stm_label != self._prev_stm_label else "reinforcement"
                events.append(EmotionalEvent(
                    tier=Tier.SHORT_TERM, kind=kind,
                    signal_vad=signal.vad, delta_vad=stm_delta, delta_magnitude=stm_mag,
                    confidence=signal.confidence, label=stm_label, top_labels=top_labels,
                    cause_message_id=msg_id, excerpt=excerpt,
                ))
                self._prev_stm_label = stm_label

            # -- Long-term (repeated high-confidence signals) ----------------
            if (signal.confidence >= self.config.ltm_min_confidence
                    and self._check_ltm_support(signal.vad)):
                ltm_old, ltm_new = self._tiers[Tier.LONG_TERM].apply(
                    signal.vad, signal.label_probs,
                    self.config.ltm_lr * signal.confidence, now,
                )
                ltm_delta = VAD(
                    ltm_new.valence   - ltm_old.valence,
                    ltm_new.arousal   - ltm_old.arousal,
                    ltm_new.dominance - ltm_old.dominance,
                )
                ltm_mag   = ltm_delta.magnitude()
                ltm_label, _ = project_label(ltm_new)
                events.append(EmotionalEvent(
                    tier=Tier.LONG_TERM, kind="promotion",
                    signal_vad=signal.vad, delta_vad=ltm_delta, delta_magnitude=ltm_mag,
                    confidence=signal.confidence, label=ltm_label, top_labels=top_labels,
                    cause_message_id=msg_id, excerpt=excerpt,
                ))
                # Slowly shift the trait baseline (EMA; very small alpha)
                self._baseline = self._baseline.blend(signal.vad, self.config.baseline_alpha)
                self._tiers[Tier.LONG_TERM].anchor = self._baseline

            # -- Trend (compare current STM intensity to previous) -----------
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

    # ---- Internal helpers --------------------------------------------------

    def _check_sustained(self, signal_vad: VAD) -> bool:
        """
        True if at least sustained_support prior confident readings agree
        with the current signal (dot product >= agreement_threshold).
        Called BEFORE appending the current signal to _recent.
        """
        if len(self._recent) < self.config.sustained_support:
            return False
        recent_conf = [v for v, c in self._recent if c >= self.config.stm_min_confidence]
        if len(recent_conf) < self.config.sustained_support:
            return False
        agree = sum(
            1 for v in recent_conf[-self.config.sustained_support:]
            if v.dot(signal_vad) >= self.config.agreement_threshold
        )
        return agree >= self.config.sustained_support

    def _check_ltm_support(self, signal_vad: VAD) -> bool:
        high_conf = [(v, c) for v, c in self._recent if c >= self.config.ltm_min_confidence]
        if len(high_conf) < self.config.ltm_support:
            return False
        agree = sum(
            1 for v, _ in high_conf[-self.config.ltm_support:]
            if v.dot(signal_vad) >= self.config.agreement_threshold
        )
        return agree >= self.config.ltm_support

    # ---- Public read interface ---------------------------------------------

    def affect_state(self) -> AffectState:
        """
        Compute and return the current AffectState with lazy wall-clock
        decay applied to all three tiers.  Thread-safe; read-only.
        """
        now = time.time()
        with self._lock:
            sit = self._tiers[Tier.SITUATIONAL].effective_vad(now)
            stm = self._tiers[Tier.SHORT_TERM].effective_vad(now)
            ltm = self._tiers[Tier.LONG_TERM].effective_vad(now)
            stm_label, stm_intensity = project_label(stm)
            ltm_label, _             = project_label(ltm)
            return AffectState(
                situational_vad=sit,
                short_term_vad=stm,
                long_term_vad=ltm,
                baseline_vad=self._baseline,
                situational_bar=self._tiers[Tier.SITUATIONAL].effective_bar(now),
                short_term_bar=self._tiers[Tier.SHORT_TERM].effective_bar(now),
                long_term_bar=self._tiers[Tier.LONG_TERM].effective_bar(now),
                stm_dominant=stm_label,
                stm_intensity=round(stm_intensity, 3),
                ltm_dominant=ltm_label,
                trend=self._trend,
                confidence=self._last_confidence,
                updated_at=now,
            )

    def events(self) -> list[EmotionalEvent]:
        """Return a snapshot of the event log (thread-safe copy)."""
        with self._lock:
            return list(self._events)

    # ---- Session management ------------------------------------------------

    def reset_tier(self, tier: Tier) -> None:
        """
        Reset a single tier to neutral.  LTM anchor is preserved as the
        current baseline (not zeroed), so trait memory survives tier resets.
        """
        now    = time.time()
        anchor = self._baseline if tier == Tier.LONG_TERM else VAD.neutral()
        with self._lock:
            self._tiers[tier] = TierState(
                VAD.neutral(), now,
                getattr(self.config, _HALF_LIFE_ATTR[tier]),
                anchor,
                _bar_zero(self.config.label_universe),
            )

    def reset_session(self) -> None:
        """
        Clear session-scoped state (Situational + STM + signal window +
        trend tracking).  LTM and the trait baseline are intentionally
        preserved — they represent the user's durable affect profile.
        """
        now = time.time()
        with self._lock:
            self._tiers[Tier.SITUATIONAL] = TierState(
                VAD.neutral(), now, self.config.situational_half_life,
                VAD.neutral(), _bar_zero(self.config.label_universe),
            )
            self._tiers[Tier.SHORT_TERM] = TierState(
                VAD.neutral(), now, self.config.stm_half_life,
                VAD.neutral(), _bar_zero(self.config.label_universe),
            )
            self._recent.clear()
            self._prev_stm_label     = EmotionLabel.NEUTRAL
            self._prev_stm_intensity = 0.0
            self._trend              = "steady"


# ---------------------------------------------------------------------------
# EmotionalMemoryWriter
# Reads the event log and persists qualifying emotional episodes to the
# retriever (any object with .add(content, importance, metadata) will do —
# MemoryRetriever in context_engine.py already satisfies this contract).
# ---------------------------------------------------------------------------

class EmotionalMemoryWriter:
    """
    Filters the event stream and writes significant emotional episodes to
    long-term memory.

    Default policy
    --------------
    * Only "shift" (STM label change), "reinforcement" (STM label held),
      and "promotion" (LTM update) events are candidates.  Single-turn
      "spike" events are excluded by default — one intense turn isn't
      worth a permanent memory entry; sustained or escalating patterns are.
    * delta_magnitude >= min_magnitude (coarse intensity gate)
    * confidence >= min_confidence (quality gate)
    * cooldown_seconds per (tier, label) pair (deduplication)
    * importance = clamp(0.4 + 0.4 * delta_magnitude + 0.2 * confidence)

    Memory content uses top_labels (the raw multi-label bars) rather than
    the single VAD-projected label: during a mood transition the blended
    VAD can project onto a third label that matches neither the old nor the
    new dominant — top_labels doesn't have that failure mode.
    """

    def __init__(
        self,
        retriever: Any,
        min_magnitude: float   = 0.35,
        min_confidence: float  = 0.55,
        cooldown_seconds: float = 300.0,
        kinds: Optional[set]   = None,
    ) -> None:
        self._retriever      = retriever
        self._min_magnitude  = min_magnitude
        self._min_confidence = min_confidence
        self._cooldown       = cooldown_seconds
        self._kinds          = kinds or {"reinforcement", "shift", "promotion"}
        self._last_write: dict[str, float] = {}
        self._lock = threading.Lock()

    def maybe_write(self, events: Sequence[EmotionalEvent]) -> list[Any]:
        """Write qualifying events to the retriever.  Returns persisted objects."""
        written: list[Any] = []
        now = time.time()
        for ev in events:
            if ev.kind          not in self._kinds:                         continue
            if ev.delta_magnitude < self._min_magnitude:                    continue
            if ev.confidence      < self._min_confidence:                   continue
            key = f"{ev.tier.value}:{ev.label.value}"
            if now - self._last_write.get(key, 0.0) < self._cooldown:      continue

            importance = _clamp01(0.4 + 0.4 * ev.delta_magnitude + 0.2 * ev.confidence)
            top = ", ".join(ev.top_labels) if ev.top_labels else ev.label.value
            content = (
                f"Emotional episode ({ev.kind} -> {ev.tier.value}): "
                f"user showed {top} ({ev.signal_vad.compact()}). "
                f"Confidence {ev.confidence:.2f}. "
                f"Excerpt: {ev.excerpt[:80]}"
            )
            with self._lock:
                self._last_write[key] = now
            mem = self._retriever.add(
                content=content,
                importance=importance,
                metadata={
                    "kind":       "emotional_episode",
                    "tier":       ev.tier.value,
                    "emotion":    ev.label.value,
                    "magnitude":  round(ev.delta_magnitude, 3),
                    "confidence": round(ev.confidence, 3),
                },
            )
            written.append(mem)
        return written


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_affect_line(affect: AffectState) -> str:
    """Single-line summary for context injection."""
    return (
        f"now: {affect.stm_dominant.value} "
        f"({affect.short_term_vad.compact()}, i={affect.stm_intensity:.2f}) | "
        f"trend: {affect.trend} | "
        f"trait: {affect.ltm_dominant.value} "
        f"({affect.long_term_vad.compact()})"
    )


def render_affect_summary(affect: AffectState) -> str:
    """Multi-line summary with ASCII bars for all three tiers."""
    lines = [
        f"trend: {affect.trend} | confidence: {affect.confidence:.2f}",
        "",
        "-- situational (this turn) --",
        render_bar(affect.situational_bar),
        "",
        "-- short-term (session mood) --",
        render_bar(affect.short_term_bar),
        "",
        "-- long-term (trait baseline) --",
        render_bar(affect.long_term_bar),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience constructor — builds the full stack with the real adapter
# ---------------------------------------------------------------------------

def build_emotion_manager(
    adapter_dir: Optional[str] = None,
    config: Optional[EmotionManagerConfig] = None,
    baseline: Optional[VAD] = None,
) -> EmotionManager:
    """
    One-liner that wires FinalAdapterClassifier → EmotionManager.

        mgr = build_emotion_manager()
        events = mgr.process_turn(msg, history[-mgr.context_turns:])
        state  = mgr.affect_state()
    """
    classifier = FinalAdapterClassifier(adapter_dir=adapter_dir)
    return EmotionManager(classifier=classifier, config=config, baseline=baseline)


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # Use real adapter if available, otherwise fall back to dummy so the
    # smoke test can run without the heavy ML stack.
    # ------------------------------------------------------------------
    class _DummyClassifier:
        def classify(self, turns: Sequence[Any]) -> EmotionSignal:
            text = str(getattr(turns[-1], "content", "")).lower()
            if any(w in text for w in ("happy", "excited", "great", "wonderful")):
                probs = {"joy": 0.82, "excitement": 0.60, "optimism": 0.35}
            elif any(w in text for w in ("worried", "nervous", "anxious", "scared")):
                probs = {"nervousness": 0.75, "fear": 0.45}
            elif any(w in text for w in ("angry", "furious", "frustrated")):
                probs = {"anger": 0.80, "annoyance": 0.55, "disapproval": 0.40}
            elif any(w in text for w in ("sad", "depressed", "miserable", "lonely")):
                probs = {"sadness": 0.78, "grief": 0.30}
            else:
                probs = {"neutral": 0.88}
            vad = probs_to_vad(probs)
            top = max(probs.values())
            return EmotionSignal(vad=vad, confidence=top, label_probs=probs, source="dummy")

    try:
        classifier: EmotionClassifier = FinalAdapterClassifier()
        print("Using real final_adapter classifier.")
    except Exception as exc:
        print(f"final_adapter not available ({exc}), using dummy classifier.")
        classifier = _DummyClassifier()

    class Msg:
        def __init__(self, content: str, mid: str) -> None:
            self.content    = content
            self.message_id = mid

    config = EmotionManagerConfig(
        sustained_support=2,
        ltm_support=3,
        context_turns=4,
    )
    mgr = EmotionManager(classifier, config=config)

    # Scripted emotional arc: excited → anxious → relieved → still positive
    script = [
        ("m0", "I just found out I got into my dream university! I'm so excited and happy!"),
        ("m1", "I'm still really excited but also getting nervous about moving away from home."),
        ("m2", "Honestly I'm quite worried about the transition. What if I struggle?"),
        ("m3", "My friends threw me a surprise party — feeling so loved and grateful."),
        ("m4", "Getting anxious again thinking about the workload. Nervous about deadlines."),
        ("m5", "Just had a great call with my future roommate. Feeling much better now!"),
        ("m6", "Really happy and optimistic about the whole thing now."),
        ("m7", "Still a bit nervous deep down, but mostly excited and grateful."),
    ]

    history: list[Msg] = []
    print("\n" + "=" * 60)
    print("EMOTION ENGINE FINAL — SMOKE TEST")
    print("=" * 60)

    for mid, text in script:
        msg = Msg(text, mid)
        history.append(msg)
        window = history[-mgr.context_turns:]
        events = mgr.process_turn(msg, window)

        print(f"\n[{mid}] {text[:70]}...")
        if events:
            for ev in events:
                top = ", ".join(ev.top_labels) or ev.label.value
                print(
                    f"  EVENT tier={ev.tier.value:<12} kind={ev.kind:<14} "
                    f"label={ev.label.value:<14} Δmag={ev.delta_magnitude:.3f}  "
                    f"top=[{top}]"
                )
        else:
            print("  (no events — below threshold)")

    print("\n" + "=" * 60)
    print("FINAL AFFECT STATE")
    print("=" * 60)
    state = mgr.affect_state()
    print(render_affect_summary(state))
    print(f"\nAffect line (for context injection):\n  {render_affect_line(state)}")

    # -- reset_tier sanity check (was crashing in the original) --
    print("\n-- reset_tier sanity check --")
    for tier in Tier:
        mgr.reset_tier(tier)
        print(f"  reset_tier({tier.value}) OK")

    mgr.reset_session()
    print("  reset_session() OK")

    print("\nAll checks passed.")
    sys.exit(0)
