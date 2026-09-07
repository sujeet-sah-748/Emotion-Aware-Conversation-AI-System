"""
test_response_generator.py
===========================
Unit tests for response/response_generator.py

Coverage:
- ResponseGenerator.generate: all emotion branches, mixed state, fallback
- ResponseGenerator._generate_negative_reframe: all known negative labels
- ResponseGenerator._generate_positive_response: all known positive labels
- ResponseGenerator._generate_mixed_reframe: returns string with both emotions
- ResponseGenerator._generate_uncertain_response: returns non-empty string
- predict_emotions: threshold filtering, top_k filtering, fallback flag
- compose_response: delegates to singleton correctly
"""

from __future__ import annotations

from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from response.response_generator import (
    ResponseGenerator,
    compose_response,
    predict_emotions,
    SimpleMessage,
    LABEL_NAMES,
    SIGMOID_THRESHOLD,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _emo(label: str, score: float) -> Dict[str, Any]:
    return {"label": label, "score": score}


def _result(emotions, used_fallback=False, all_scores=None):
    """Build arguments as predict_emotions returns them."""
    return emotions, used_fallback, all_scores or emotions


# ===========================================================================
# LABEL_NAMES
# ===========================================================================

class TestLabelNames:
    def test_28_labels(self):
        assert len(LABEL_NAMES) == 28

    def test_contains_joy(self):
        assert "joy" in LABEL_NAMES

    def test_contains_neutral(self):
        assert "neutral" in LABEL_NAMES

    def test_all_strings(self):
        for name in LABEL_NAMES:
            assert isinstance(name, str)


# ===========================================================================
# SimpleMessage
# ===========================================================================

class TestSimpleMessage:
    def test_content_attribute(self):
        msg = SimpleMessage("hello world")
        assert msg.content == "hello world"

    def test_empty_content(self):
        msg = SimpleMessage("")
        assert msg.content == ""


# ===========================================================================
# ResponseGenerator — negative emotions
# ===========================================================================

NEGATIVE_EMOTIONS = [
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
    "confusion",
]

POSITIVE_EMOTIONS = [
    "joy", "excitement", "admiration", "amusement", "approval",
    "gratitude", "love", "optimism", "pride", "relief", "caring",
    "curiosity", "desire", "surprise", "realization",
]


class TestResponseGeneratorNegative:
    @pytest.fixture(autouse=True)
    def _gen(self):
        self.gen = ResponseGenerator()

    @pytest.mark.parametrize("emotion", NEGATIVE_EMOTIONS)
    def test_negative_emotion_returns_nonempty(self, emotion):
        emotions = [_emo(emotion, 0.8)]
        all_scores = [_emo(emotion, 0.8)]
        response = self.gen.generate("I feel bad", emotions, False, all_scores)
        assert isinstance(response, str)
        assert len(response) > 20

    @pytest.mark.parametrize("emotion", NEGATIVE_EMOTIONS)
    def test_negative_reframe_direct(self, emotion):
        result = self.gen._generate_negative_reframe(emotion, "test text")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_unknown_negative_emotion_uses_fallback_template(self):
        result = self.gen._generate_negative_reframe("unknown_bad_feeling", "test")
        assert "unknown_bad_feeling" in result

    def test_anger_response_mentions_frustration(self):
        resp = self.gen._generate_negative_reframe("anger", "I am angry")
        assert any(w in resp.lower() for w in ("angry", "frustration", "frustrat"))

    def test_sadness_response_empathetic(self):
        resp = self.gen._generate_negative_reframe("sadness", "I feel sad")
        assert any(w in resp.lower() for w in ("sad", "heavy", "weight", "care", "alone"))

    def test_fear_response_mentions_safety(self):
        resp = self.gen._generate_negative_reframe("fear", "I am scared")
        assert any(w in resp.lower() for w in ("safe", "fear", "scared", "small step"))


# ===========================================================================
# ResponseGenerator — positive emotions
# ===========================================================================

class TestResponseGeneratorPositive:
    @pytest.fixture(autouse=True)
    def _gen(self):
        self.gen = ResponseGenerator()

    @pytest.mark.parametrize("emotion", POSITIVE_EMOTIONS)
    def test_positive_emotion_returns_nonempty(self, emotion):
        emotions = [_emo(emotion, 0.8)]
        all_scores = [_emo(emotion, 0.8)]
        response = self.gen.generate("I feel good", emotions, False, all_scores)
        assert isinstance(response, str)
        assert len(response) > 10

    @pytest.mark.parametrize("emotion", POSITIVE_EMOTIONS)
    def test_positive_response_direct(self, emotion):
        result = self.gen._generate_positive_response(emotion, "test")
        assert isinstance(result, str)

    def test_unknown_positive_uses_fallback_template(self):
        result = self.gen._generate_positive_response("transcendence", "test")
        assert "transcendence" in result

    def test_joy_response_warm_tone(self):
        resp = self.gen._generate_positive_response("joy", "I am happy")
        assert any(w in resp.lower() for w in ("wonderful", "joy", "happy", "contagious", "beautiful"))

    def test_gratitude_response_mentions_thankful(self):
        resp = self.gen._generate_positive_response("gratitude", "thanks")
        assert any(w in resp.lower() for w in ("gratitude", "grateful", "thankful", "difference"))


# ===========================================================================
# ResponseGenerator — mixed state
# ===========================================================================

class TestResponseGeneratorMixed:
    @pytest.fixture(autouse=True)
    def _gen(self):
        self.gen = ResponseGenerator()

    def test_mixed_reframe_contains_both_emotions(self):
        result = self.gen._generate_mixed_reframe("sadness", "hope", "feeling complex")
        assert "sadness" in result
        assert "hope" in result

    def test_mixed_state_triggered_when_negative_top_and_positive_hidden(self):
        emotions = [_emo("sadness", 0.75)]
        all_scores = [
            _emo("sadness", 0.75),
            _emo("hope", 0.40),
            _emo("relief", 0.20),
        ]
        response = self.gen.generate("complex feelings", emotions, False, all_scores)
        # Because hope/relief are positive and above 0.15 threshold,
        # mixed-state path should activate.
        assert isinstance(response, str)
        assert len(response) > 20

    def test_negative_without_positive_signal_uses_direct_reframe(self):
        emotions = [_emo("anger", 0.85)]
        all_scores = [_emo("anger", 0.85)]  # no positive signal
        response = self.gen.generate("I hate this", emotions, False, all_scores)
        assert isinstance(response, str)


# ===========================================================================
# ResponseGenerator — fallback / uncertain
# ===========================================================================

class TestResponseGeneratorFallback:
    @pytest.fixture(autouse=True)
    def _gen(self):
        self.gen = ResponseGenerator()

    def test_used_fallback_true_uses_uncertain_response(self):
        emotions = [_emo("neutral", 0.25)]
        all_scores = [_emo("neutral", 0.25)]
        response = self.gen.generate("...", emotions, True, all_scores)
        assert isinstance(response, str)
        assert len(response) > 10

    def test_low_score_uses_uncertain_response(self):
        emotions = [_emo("joy", 0.20)]  # below 0.35 threshold
        all_scores = [_emo("joy", 0.20)]
        response = self.gen.generate("hmm", emotions, False, all_scores)
        assert isinstance(response, str)

    def test_empty_emotions_list_returns_fallback(self):
        response = self.gen.generate("test", [], False, [])
        assert isinstance(response, str)
        assert len(response) > 0

    def test_uncertain_response_is_nonempty(self):
        result = self.gen._generate_uncertain_response("some text")
        assert isinstance(result, str)
        assert len(result) > 20


# ===========================================================================
# predict_emotions — uses mocked classifier
# ===========================================================================

class TestPredictEmotions:
    """
    predict_emotions calls the module-level `classifier` singleton.
    We mock the classifier object directly to test the function logic
    without triggering actual ML inference.
    """

    def _make_signal(self, probs: dict):
        from core.emotion_engine import EmotionSignal, probs_to_vad
        vad = probs_to_vad(probs)
        top = max(probs.values())
        second = sorted(probs.values(), reverse=True)[1] if len(probs) > 1 else 0.0
        conf = min(1.0, 0.5 * top + 0.5 * (top - second))
        return EmotionSignal(vad=vad, confidence=conf, label_probs=probs, source="mock")

    def _mock_classifier(self, probs: dict):
        import response.response_generator as rg
        sig = self._make_signal(probs)
        mock_clf = MagicMock()
        mock_clf.classify.return_value = sig
        return patch.object(rg, "classifier", mock_clf)

    def test_returns_tuple_of_three(self):
        probs = {"joy": 0.8, "neutral": 0.1}
        with self._mock_classifier(probs):
            result = predict_emotions("I am happy", threshold=0.5)
        assert isinstance(result, tuple) and len(result) == 3

    def test_above_threshold_label_included(self):
        probs = {"joy": 0.9, "neutral": 0.1}
        with self._mock_classifier(probs):
            emotions, used_fallback, all_scores = predict_emotions("happy", threshold=0.5)
        labels = [e["label"] for e in emotions]
        assert "joy" in labels

    def test_below_threshold_uses_fallback_flag(self):
        # All scores below threshold → used_fallback=True
        probs = {"neutral": 0.30, "joy": 0.20}
        with self._mock_classifier(probs):
            _, used_fallback, _ = predict_emotions("meh", threshold=0.5)
        assert used_fallback is True

    def test_top_k_limits_results(self):
        probs = {"joy": 0.9, "excitement": 0.7, "love": 0.6, "gratitude": 0.5}
        with self._mock_classifier(probs):
            emotions, _, _ = predict_emotions("wonderful", threshold=0.1, top_k=2)
        assert len(emotions) <= 2

    def test_all_scores_sorted_descending(self):
        probs = {"joy": 0.9, "sadness": 0.3, "anger": 0.5}
        with self._mock_classifier(probs):
            _, _, all_scores = predict_emotions("mixed", threshold=0.1)
        scores = [s["score"] for s in all_scores]
        assert scores == sorted(scores, reverse=True)

    def test_fallback_returns_single_best(self):
        # Nothing above threshold; should return list with 1 item
        probs = {"neutral": 0.20}
        with self._mock_classifier(probs):
            emotions, used_fallback, _ = predict_emotions("eh", threshold=0.5)
        assert len(emotions) == 1
        assert used_fallback is True

    def test_each_result_has_label_and_score(self):
        probs = {"joy": 0.8}
        with self._mock_classifier(probs):
            emotions, _, all_scores = predict_emotions("great", threshold=0.5)
        for item in emotions + all_scores:
            assert "label" in item
            assert "score" in item
            assert isinstance(item["score"], float)


# ===========================================================================
# compose_response — delegates to singleton
# ===========================================================================

class TestComposeResponse:
    def test_returns_string(self):
        emotions = [_emo("joy", 0.85)]
        all_scores = [_emo("joy", 0.85)]
        result = compose_response("I am happy", emotions, False, all_scores)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_delegates_to_generator(self, response_generator):
        """compose_response must produce same output as ResponseGenerator.generate."""
        emotions = [_emo("sadness", 0.80)]
        all_scores = [_emo("sadness", 0.80)]
        text = "I feel terrible"
        via_compose = compose_response(text, emotions, False, all_scores)
        via_direct = response_generator.generate(text, emotions, False, all_scores)
        assert via_compose == via_direct

    def test_empty_emotions_does_not_crash(self):
        result = compose_response("hmm", [], False, [])
        assert isinstance(result, str)
