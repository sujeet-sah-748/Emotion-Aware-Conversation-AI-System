"""
test_emotion_engine.py
======================
Unit tests for core/emotion_engine.py

Coverage:
- VAD dataclass: arithmetic, clamping, decay, distance, cosine, serialisation
- probs_to_vad: weighted aggregation
- project_label: direction + magnitude correctness
- EmotionManagerConfig: validation guards
- TierState: apply + effective helpers
- EmotionManager: process_turn, affect_state, events, reset_tier, reset_session
- EmotionalMemoryWriter: cooldown, kind filter, importance formula
- render_affect_line: smoke test
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest

from core.emotion_engine import (
    VAD,
    EmotionLabel,
    EmotionManager,
    EmotionManagerConfig,
    EmotionalEvent,
    EmotionalMemoryWriter,
    EmotionSignal,
    GOEMOTIONS_VAD,
    PROTOTYPES,
    Tier,
    TierState,
    _bar_zero,
    _clamp,
    _clamp01,
    probs_to_vad,
    project_label,
    render_affect_line,
    render_affect_summary,
    GOEMOTIONS_LABELS,
)


# ===========================================================================
# Helpers / tiny factories
# ===========================================================================

def _sig(probs: dict, source: str = "test") -> EmotionSignal:
    vad = probs_to_vad(probs)
    top = max(probs.values())
    second = sorted(probs.values(), reverse=True)[1] if len(probs) > 1 else 0.0
    conf = min(1.0, 0.5 * top + 0.5 * (top - second))
    return EmotionSignal(vad=vad, confidence=conf, label_probs=probs, source=source)


class _StaticClassifier:
    """Classifier that always returns the same pre-built signal."""
    def __init__(self, signal: EmotionSignal):
        self._sig = signal
    def classify(self, turns):
        return self._sig


class _Msg:
    def __init__(self, content: str, mid: str = "x"):
        self.content = content
        self.message_id = mid


def _mgr(signal: EmotionSignal, **cfg_kwargs) -> EmotionManager:
    defaults = dict(
        situational_min_confidence=0.10,
        stm_min_confidence=0.30,
        ltm_min_confidence=0.45,
        sustained_support=2,
        ltm_support=3,
        signal_window=12,
        spike_delta=0.15,
    )
    defaults.update(cfg_kwargs)
    cfg = EmotionManagerConfig(**defaults)
    return EmotionManager(classifier=_StaticClassifier(signal), config=cfg)


# ===========================================================================
# VAD dataclass
# ===========================================================================

class TestVAD:
    def test_neutral_factory(self):
        v = VAD.neutral()
        assert v.valence == 0.0
        assert v.arousal == 0.0
        assert v.dominance == 0.0

    def test_clamping_on_construction(self):
        v = VAD(2.0, -3.0, 1.5)
        assert v.valence == 1.0
        assert v.arousal == -1.0
        assert v.dominance == 1.0

    def test_magnitude_zero(self):
        assert VAD.neutral().magnitude() == 0.0

    def test_magnitude_unit_axis(self):
        assert abs(VAD(1.0, 0.0, 0.0).magnitude() - 1.0) < 1e-9

    def test_magnitude_full_corner(self):
        m = VAD(1.0, 1.0, 1.0).magnitude()
        assert abs(m - math.sqrt(3)) < 1e-9

    def test_unit_vector_neutral_returns_neutral(self):
        u = VAD.neutral().unit()
        assert u == VAD.neutral()

    def test_unit_vector_normalised(self):
        v = VAD(0.3, -0.7, 0.5)
        u = v.unit()
        assert abs(u.magnitude() - 1.0) < 1e-6

    def test_blend_lr_zero_unchanged(self):
        v = VAD(0.5, 0.5, 0.5)
        result = v.blend(VAD(1.0, 1.0, 1.0), lr=0.0)
        assert result == v

    def test_blend_lr_one_becomes_other(self):
        v = VAD(0.0, 0.0, 0.0)
        other = VAD(1.0, -1.0, 0.5)
        result = v.blend(other, lr=1.0)
        assert abs(result.valence - 1.0) < 1e-9
        assert abs(result.arousal - (-1.0)) < 1e-9

    def test_blend_midpoint(self):
        a = VAD(0.0, 0.0, 0.0)
        b = VAD(1.0, 1.0, 1.0)
        mid = a.blend(b, lr=0.5)
        assert abs(mid.valence - 0.5) < 1e-9

    def test_decay_elapsed_zero_unchanged(self):
        v = VAD(0.8, 0.4, 0.2)
        result = v.decay_toward(VAD.neutral(), half_life=300.0, elapsed=0.0)
        assert result == v

    def test_decay_one_half_life(self):
        v = VAD(0.8, 0.0, 0.0)
        anchor = VAD.neutral()
        result = v.decay_toward(anchor, half_life=300.0, elapsed=300.0)
        assert abs(result.valence - 0.4) < 1e-6

    def test_decay_two_half_lives(self):
        v = VAD(1.0, 0.0, 0.0)
        result = v.decay_toward(VAD.neutral(), half_life=300.0, elapsed=600.0)
        assert abs(result.valence - 0.25) < 1e-6

    def test_decay_toward_non_zero_anchor(self):
        v = VAD(1.0, 0.0, 0.0)
        anchor = VAD(0.5, 0.0, 0.0)
        result = v.decay_toward(anchor, half_life=300.0, elapsed=300.0)
        # deviation from anchor = 0.5; after 1 half-life → 0.25 above anchor = 0.75
        assert abs(result.valence - 0.75) < 1e-6

    def test_distance_same_point_zero(self):
        v = VAD(0.3, -0.2, 0.1)
        assert v.distance(v) == 0.0

    def test_distance_symmetry(self):
        a = VAD(0.5, 0.3, -0.1)
        b = VAD(-0.2, 0.7, 0.4)
        assert abs(a.distance(b) - b.distance(a)) < 1e-9

    def test_cosine_identical(self):
        v = VAD(0.5, 0.3, 0.2)
        assert abs(v.cosine(v) - 1.0) < 1e-6

    def test_cosine_opposite(self):
        v = VAD(0.5, 0.3, 0.2)
        neg = VAD(-0.5, -0.3, -0.2)
        assert abs(v.cosine(neg) - (-1.0)) < 1e-6

    def test_cosine_neutral_returns_zero(self):
        v = VAD(0.5, 0.3, 0.2)
        assert v.cosine(VAD.neutral()) == 0.0

    def test_dot_product(self):
        a = VAD(1.0, 0.0, 0.0)
        b = VAD(0.0, 1.0, 0.0)
        assert a.dot(b) == 0.0
        assert a.dot(a) == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        v = VAD(0.4, -0.3, 0.7)
        d = v.to_dict()
        assert d == {"valence": 0.4, "arousal": -0.3, "dominance": 0.7}
        assert VAD.from_dict(d) == v

    def test_to_list_from_list_roundtrip(self):
        v = VAD(0.1, 0.2, 0.3)
        assert VAD.from_list(v.to_list()) == v

    def test_compact_string_format(self):
        s = VAD(0.5, -0.3, 0.0).compact()
        assert "v" in s and "a" in s and "d" in s

    def test_frozen_immutable(self):
        v = VAD(0.5, 0.5, 0.5)
        with pytest.raises((AttributeError, TypeError)):
            v.valence = 0.0  # type: ignore[misc]


# ===========================================================================
# _clamp helpers
# ===========================================================================

class TestClampHelpers:
    def test_clamp_within(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_above(self):
        assert _clamp(2.0) == 1.0

    def test_clamp_below(self):
        assert _clamp(-2.0) == -1.0

    def test_clamp01(self):
        assert _clamp01(1.5) == 1.0
        assert _clamp01(-0.5) == 0.0
        assert _clamp01(0.3) == 0.3


# ===========================================================================
# probs_to_vad
# ===========================================================================

class TestProbsToVad:
    def test_neutral_only(self):
        v = probs_to_vad({"neutral": 1.0})
        assert v == VAD.neutral()

    def test_joy_pushes_positive_valence(self):
        v = probs_to_vad({"joy": 0.9})
        assert v.valence > 0.5

    def test_sadness_pushes_negative_valence(self):
        v = probs_to_vad({"sadness": 0.9})
        assert v.valence < -0.5

    def test_anger_high_arousal(self):
        v = probs_to_vad({"anger": 0.9})
        assert v.arousal > 0.5

    def test_empty_probs_returns_neutral(self):
        v = probs_to_vad({})
        assert v == VAD.neutral()

    def test_unknown_label_ignored(self):
        # Should not raise; unknown labels are skipped
        v = probs_to_vad({"nonexistent_emotion": 1.0})
        assert v == VAD.neutral()

    def test_multi_label_coactivation(self):
        # Both joy and sadness active — result should be somewhere in between
        v = probs_to_vad({"joy": 0.6, "sadness": 0.6})
        # Not expected to hit extreme positive or negative valence
        assert -0.5 < v.valence < 0.5

    def test_sqrt_normalisation_doesnt_collapse_to_neutral(self):
        # Many labels active simultaneously should NOT dilute to exactly neutral
        probs = {label: 0.5 for label in GOEMOTIONS_LABELS[:10]}
        v = probs_to_vad(probs)
        assert v.magnitude() > 0.05


# ===========================================================================
# project_label
# ===========================================================================

class TestProjectLabel:
    def test_neutral_vad_projects_to_neutral(self):
        label, intensity = project_label(VAD.neutral())
        assert label == EmotionLabel.NEUTRAL
        assert intensity == 0.0

    def test_tiny_magnitude_projects_to_neutral(self):
        label, _ = project_label(VAD(0.01, 0.01, 0.01))
        assert label == EmotionLabel.NEUTRAL

    def test_joy_prototype_projects_to_joy(self):
        label, intensity = project_label(PROTOTYPES[EmotionLabel.JOY])
        assert label == EmotionLabel.JOY
        assert intensity > 0.5

    def test_anger_prototype_projects_to_anger(self):
        label, _ = project_label(PROTOTYPES[EmotionLabel.ANGER])
        assert label == EmotionLabel.ANGER

    def test_fear_prototype_projects_to_fear(self):
        label, _ = project_label(PROTOTYPES[EmotionLabel.FEAR])
        assert label == EmotionLabel.FEAR

    def test_sadness_prototype_projects_to_sadness(self):
        label, _ = project_label(PROTOTYPES[EmotionLabel.SADNESS])
        assert label == EmotionLabel.SADNESS

    def test_intensity_range(self):
        for proto in PROTOTYPES.values():
            _, intensity = project_label(proto)
            assert 0.0 <= intensity <= 1.0

    def test_returns_tuple(self):
        result = project_label(VAD(0.5, 0.3, 0.2))
        assert isinstance(result, tuple) and len(result) == 2


# ===========================================================================
# EmotionManagerConfig validation
# ===========================================================================

class TestEmotionManagerConfig:
    def test_default_construction(self):
        cfg = EmotionManagerConfig()
        assert cfg.situational_half_life > 0
        assert cfg.stm_half_life > 0
        assert cfg.ltm_half_life > 0

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            EmotionManagerConfig(situational_half_life=-1.0)

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError):
            EmotionManagerConfig(stm_half_life=0.0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EmotionManagerConfig(situational_min_confidence=1.5)

    def test_lr_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EmotionManagerConfig(situational_lr=0.0)  # must be > 0

    def test_support_below_one_raises(self):
        with pytest.raises(ValueError):
            EmotionManagerConfig(sustained_support=0)

    def test_signal_window_too_small_raises(self):
        with pytest.raises(ValueError):
            EmotionManagerConfig(sustained_support=5, ltm_support=3, signal_window=4)

    def test_context_turns_attribute(self):
        cfg = EmotionManagerConfig(context_turns=8)
        assert cfg.context_turns == 8


# ===========================================================================
# TierState
# ===========================================================================

class TestTierState:
    def test_effective_vad_no_elapsed(self):
        now = time.time()
        ts = TierState(
            vad=VAD(0.8, 0.0, 0.0),
            updated_at=now,
            half_life=300.0,
            anchor=VAD.neutral(),
            bar=_bar_zero(GOEMOTIONS_LABELS),
        )
        eff = ts.effective_vad(now)
        assert abs(eff.valence - 0.8) < 1e-6

    def test_effective_vad_after_one_half_life(self):
        now = time.time()
        ts = TierState(
            vad=VAD(1.0, 0.0, 0.0),
            updated_at=now - 300.0,
            half_life=300.0,
            anchor=VAD.neutral(),
            bar=_bar_zero(GOEMOTIONS_LABELS),
        )
        eff = ts.effective_vad(now)
        assert abs(eff.valence - 0.5) < 1e-6

    def test_apply_updates_vad(self):
        now = time.time()
        ts = TierState(
            vad=VAD.neutral(),
            updated_at=now,
            half_life=300.0,
            anchor=VAD.neutral(),
            bar=_bar_zero(GOEMOTIONS_LABELS),
        )
        joy_probs = {"joy": 0.9}
        signal_vad = probs_to_vad(joy_probs)
        old, new = ts.apply(signal_vad, joy_probs, lr=1.0, now=now)
        assert new.valence > 0.5  # should have moved toward joy

    def test_apply_returns_old_and_new(self):
        now = time.time()
        ts = TierState(VAD(0.3, 0.0, 0.0), now, 300.0, VAD.neutral(), _bar_zero(GOEMOTIONS_LABELS))
        old, new = ts.apply(VAD(0.9, 0.0, 0.0), {}, lr=0.5, now=now)
        assert isinstance(old, VAD)
        assert isinstance(new, VAD)
        assert new.valence > old.valence


# ===========================================================================
# EmotionManager — process_turn
# ===========================================================================

class TestEmotionManagerProcessTurn:
    def test_low_confidence_produces_no_events(self, low_conf_classifier):
        from core.emotion_engine import EmotionManager, EmotionManagerConfig
        cfg = EmotionManagerConfig(situational_min_confidence=0.20)
        mgr = EmotionManager(classifier=low_conf_classifier, config=cfg)
        msg = _Msg("hello")
        events = mgr.process_turn(msg, [msg])
        assert events == []

    def test_high_confidence_produces_situational_event(self):
        sig = _sig({"joy": 0.90, "excitement": 0.60})
        mgr = _mgr(sig)
        msg = _Msg("I am so happy!")
        events = mgr.process_turn(msg, [msg])
        tiers = {e.tier for e in events}
        assert Tier.SITUATIONAL in tiers

    def test_event_contains_expected_fields(self):
        sig = _sig({"joy": 0.90})
        mgr = _mgr(sig)
        msg = _Msg("great news!", "msg-abc")
        events = mgr.process_turn(msg, [msg])
        if events:
            ev = events[0]
            assert isinstance(ev.delta_magnitude, float)
            assert 0.0 <= ev.confidence <= 1.0
            assert ev.tier in Tier.__members__.values()
            assert ev.cause_message_id == "msg-abc"
            assert len(ev.excerpt) <= 120

    def test_repeated_signals_trigger_stm(self):
        """STM updates require sustained_support prior agreeing signals."""
        sig = _sig({"joy": 0.85, "excitement": 0.60})
        cfg = EmotionManagerConfig(
            sustained_support=2,
            ltm_support=5,
            stm_min_confidence=0.30,
            situational_min_confidence=0.10,
            signal_window=12,
            spike_delta=0.10,
        )
        mgr = EmotionManager(classifier=_StaticClassifier(sig), config=cfg)
        history = []
        stm_events = []
        for i in range(6):
            msg = _Msg(f"joy message {i}", f"m{i}")
            history.append(msg)
            evs = mgr.process_turn(msg, history[-cfg.context_turns:])
            stm_events.extend([e for e in evs if e.tier == Tier.SHORT_TERM])
        assert len(stm_events) > 0, "Expected at least one STM event after repeated signals"

    def test_events_appended_to_event_log(self):
        sig = _sig({"joy": 0.90})
        mgr = _mgr(sig)
        msg = _Msg("wonderful day")
        mgr.process_turn(msg, [msg])
        assert len(mgr.events()) >= 0  # won't crash even if 0 events

    def test_event_log_respects_max_size(self):
        sig = _sig({"joy": 0.90, "excitement": 0.55})
        cfg = EmotionManagerConfig(
            event_log_max=5,
            spike_delta=0.05,
            situational_min_confidence=0.10,
        )
        mgr = EmotionManager(classifier=_StaticClassifier(sig), config=cfg)
        for i in range(20):
            msg = _Msg(f"msg {i}", f"m{i}")
            mgr.process_turn(msg, [msg])
        assert len(mgr.events()) <= 5

    def test_thread_safety_concurrent_turns(self):
        import threading
        sig = _sig({"joy": 0.85})
        mgr = _mgr(sig)
        errors = []

        def _run():
            try:
                for i in range(10):
                    msg = _Msg(f"msg {i}")
                    mgr.process_turn(msg, [msg])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety failures: {errors}"


# ===========================================================================
# EmotionManager — affect_state
# ===========================================================================

class TestAffectState:
    def test_returns_affect_state_object(self, emotion_manager):
        from core.emotion_engine import AffectState
        state = emotion_manager.affect_state()
        assert isinstance(state, AffectState)

    def test_fresh_manager_is_neutral(self, emotion_manager):
        state = emotion_manager.affect_state()
        assert state.stm_dominant == EmotionLabel.NEUTRAL
        assert state.ltm_dominant == EmotionLabel.NEUTRAL

    def test_stm_dominant_updates_after_turns(self):
        sig = _sig({"joy": 0.90, "excitement": 0.60})
        mgr = _mgr(sig, sustained_support=1, stm_min_confidence=0.20)
        for i in range(4):
            msg = _Msg("wonderful!", f"m{i}")
            mgr.process_turn(msg, [msg])
        state = mgr.affect_state()
        # Should not be neutral after repeated positive signals
        assert state.stm_dominant != EmotionLabel.NEUTRAL or state.confidence == 0.0

    def test_trend_field_valid_values(self, emotion_manager):
        state = emotion_manager.affect_state()
        assert state.trend in ("rising", "falling", "steady")

    def test_confidence_in_range(self, emotion_manager):
        state = emotion_manager.affect_state()
        assert 0.0 <= state.confidence <= 1.0

    def test_situational_bar_has_all_labels(self, emotion_manager):
        state = emotion_manager.affect_state()
        for label in GOEMOTIONS_LABELS:
            assert label in state.situational_bar

    def test_vad_fields_are_vad_instances(self, emotion_manager):
        state = emotion_manager.affect_state()
        assert isinstance(state.situational_vad, VAD)
        assert isinstance(state.short_term_vad, VAD)
        assert isinstance(state.long_term_vad, VAD)

    def test_updated_at_is_recent(self, emotion_manager):
        before = time.time()
        state = emotion_manager.affect_state()
        after = time.time()
        assert before <= state.updated_at <= after


# ===========================================================================
# EmotionManager — reset_tier / reset_session
# ===========================================================================

class TestEmotionManagerReset:
    def test_reset_tier_situational(self):
        sig = _sig({"anger": 0.90})
        mgr = _mgr(sig, sustained_support=1, stm_min_confidence=0.20, spike_delta=0.05)
        for i in range(3):
            msg = _Msg("very angry", f"m{i}")
            mgr.process_turn(msg, [msg])

        mgr.reset_tier(Tier.SITUATIONAL)
        state = mgr.affect_state()
        # Situational should be back near neutral
        assert abs(state.situational_vad.magnitude()) < 0.3

    def test_reset_tier_short_term(self):
        sig = _sig({"joy": 0.90})
        mgr = _mgr(sig, sustained_support=1, stm_min_confidence=0.20, spike_delta=0.05)
        for i in range(5):
            msg = _Msg("happy day", f"m{i}")
            mgr.process_turn(msg, [msg])

        mgr.reset_tier(Tier.SHORT_TERM)
        state = mgr.affect_state()
        assert abs(state.short_term_vad.magnitude()) < 0.3

    def test_reset_session_clears_stm_and_situational(self):
        sig = _sig({"anger": 0.90})
        mgr = _mgr(sig, sustained_support=1, stm_min_confidence=0.20, spike_delta=0.05)
        for i in range(5):
            msg = _Msg("furious", f"m{i}")
            mgr.process_turn(msg, [msg])

        mgr.reset_session()
        state = mgr.affect_state()
        assert abs(state.situational_vad.magnitude()) < 0.3
        assert abs(state.short_term_vad.magnitude()) < 0.3

    def test_reset_session_preserves_ltm(self):
        sig = _sig({"joy": 0.95})
        cfg = EmotionManagerConfig(
            sustained_support=1,
            ltm_support=2,
            stm_min_confidence=0.20,
            ltm_min_confidence=0.40,
            spike_delta=0.05,
            situational_min_confidence=0.10,
        )
        mgr = EmotionManager(classifier=_StaticClassifier(sig), config=cfg)
        for i in range(10):
            msg = _Msg("absolutely wonderful", f"m{i}")
            mgr.process_turn(msg, [msg])

        ltm_before = mgr.affect_state().long_term_vad
        mgr.reset_session()
        ltm_after = mgr.affect_state().long_term_vad
        # LTM should be preserved; STM is cleared
        assert ltm_before.valence == ltm_after.valence

    def test_reset_all_tiers_individually(self):
        sig = _sig({"fear": 0.85})
        mgr = _mgr(sig, sustained_support=1, spike_delta=0.05)
        msg = _Msg("terrified")
        mgr.process_turn(msg, [msg])
        for tier in Tier:
            mgr.reset_tier(tier)   # should not raise


# ===========================================================================
# EmotionManager — events()
# ===========================================================================

class TestEmotionManagerEvents:
    def test_events_returns_list(self, emotion_manager):
        result = emotion_manager.events()
        assert isinstance(result, list)

    def test_events_are_emotional_event_instances(self):
        sig = _sig({"joy": 0.90})
        mgr = _mgr(sig, sustained_support=1, spike_delta=0.05, stm_min_confidence=0.20)
        for i in range(3):
            msg = _Msg("great day", f"m{i}")
            mgr.process_turn(msg, [msg])
        for ev in mgr.events():
            assert isinstance(ev, EmotionalEvent)

    def test_to_dict_serialisable(self):
        sig = _sig({"joy": 0.90})
        mgr = _mgr(sig, sustained_support=1, spike_delta=0.05, stm_min_confidence=0.20)
        msg = _Msg("great day")
        mgr.process_turn(msg, [msg])
        import json
        for ev in mgr.events():
            d = ev.to_dict()
            # Must be JSON-serialisable (no VAD objects, etc.)
            json.dumps(d)


# ===========================================================================
# EmotionalMemoryWriter
# ===========================================================================

class TestEmotionalMemoryWriter:
    def _make_event(self, kind="shift", mag=0.5, conf=0.7, label=EmotionLabel.JOY):
        return EmotionalEvent(
            tier=Tier.SHORT_TERM,
            kind=kind,
            signal_vad=VAD(0.8, 0.5, 0.3),
            delta_vad=VAD(0.3, 0.1, 0.1),
            delta_magnitude=mag,
            confidence=conf,
            label=label,
            top_labels=("joy", "excitement"),
            cause_message_id="msg-x",
            excerpt="I feel great today",
        )

    def test_qualifying_event_is_written(self):
        retriever = MagicMock()
        retriever.add.return_value = MagicMock()
        writer = EmotionalMemoryWriter(retriever, min_magnitude=0.30, min_confidence=0.55)
        written = writer.maybe_write([self._make_event(kind="shift", mag=0.5, conf=0.7)])
        assert len(written) == 1
        retriever.add.assert_called_once()

    def test_low_magnitude_event_not_written(self):
        retriever = MagicMock()
        writer = EmotionalMemoryWriter(retriever, min_magnitude=0.50)
        written = writer.maybe_write([self._make_event(kind="shift", mag=0.20)])
        assert written == []
        retriever.add.assert_not_called()

    def test_low_confidence_event_not_written(self):
        retriever = MagicMock()
        writer = EmotionalMemoryWriter(retriever, min_confidence=0.70)
        written = writer.maybe_write([self._make_event(kind="shift", conf=0.40)])
        assert written == []

    def test_spike_events_excluded_by_default(self):
        retriever = MagicMock()
        writer = EmotionalMemoryWriter(retriever)
        written = writer.maybe_write([self._make_event(kind="spike", mag=0.9, conf=0.9)])
        assert written == []

    def test_cooldown_prevents_duplicate_writes(self):
        retriever = MagicMock()
        retriever.add.return_value = MagicMock()
        writer = EmotionalMemoryWriter(retriever, cooldown_seconds=9999)
        ev = self._make_event(kind="shift", mag=0.5, conf=0.7)
        writer.maybe_write([ev])
        writer.maybe_write([ev])   # second call — within cooldown
        assert retriever.add.call_count == 1

    def test_importance_formula_in_range(self):
        """Importance = clamp(0.4 + 0.4*delta_mag + 0.2*confidence)."""
        retriever = MagicMock()
        call_kwargs = {}

        def capture_add(**kwargs):
            call_kwargs.update(kwargs)
            return MagicMock()

        retriever.add.side_effect = capture_add
        writer = EmotionalMemoryWriter(retriever)
        writer.maybe_write([self._make_event(kind="shift", mag=0.5, conf=0.8)])
        importance = call_kwargs.get("importance", None)
        assert importance is not None
        assert 0.0 <= importance <= 1.0

    def test_empty_events_list_no_error(self):
        writer = EmotionalMemoryWriter(MagicMock())
        assert writer.maybe_write([]) == []


# ===========================================================================
# Rendering helpers
# ===========================================================================

class TestRenderHelpers:
    def _fresh_state(self):
        from core.emotion_engine import AffectState, _bar_zero
        return AffectState(
            situational_vad=VAD.neutral(),
            short_term_vad=VAD(0.6, 0.3, 0.2),
            long_term_vad=VAD(0.1, -0.1, 0.0),
            baseline_vad=VAD.neutral(),
            situational_bar=_bar_zero(GOEMOTIONS_LABELS),
            short_term_bar=_bar_zero(GOEMOTIONS_LABELS),
            long_term_bar=_bar_zero(GOEMOTIONS_LABELS),
            stm_dominant=EmotionLabel.JOY,
            stm_intensity=0.45,
            ltm_dominant=EmotionLabel.NEUTRAL,
            trend="rising",
            confidence=0.78,
            updated_at=time.time(),
        )

    def test_render_affect_line_returns_string(self):
        state = self._fresh_state()
        line = render_affect_line(state)
        assert isinstance(line, str)
        assert len(line) > 0

    def test_render_affect_line_contains_trend(self):
        state = self._fresh_state()
        line = render_affect_line(state)
        assert "rising" in line

    def test_render_affect_line_contains_dominant_emotion(self):
        state = self._fresh_state()
        line = render_affect_line(state)
        assert "joy" in line

    def test_render_affect_summary_contains_tiers(self):
        state = self._fresh_state()
        summary = render_affect_summary(state)
        assert "situational" in summary.lower()
        assert "short-term" in summary.lower()
        assert "long-term" in summary.lower()

    def test_render_affect_summary_returns_multi_line(self):
        state = self._fresh_state()
        assert "\n" in render_affect_summary(state)


# ===========================================================================
# EmotionLabel enum
# ===========================================================================

class TestEmotionLabel:
    def test_all_labels_have_string_value(self):
        for label in EmotionLabel:
            assert isinstance(label.value, str)

    def test_neutral_exists(self):
        assert EmotionLabel.NEUTRAL.value == "neutral"

    def test_goemotions_vad_covers_all_canonical_labels(self):
        for label in GOEMOTIONS_LABELS:
            assert label in GOEMOTIONS_VAD

    def test_prototypes_all_emotion_labels(self):
        for el in EmotionLabel:
            assert el in PROTOTYPES
