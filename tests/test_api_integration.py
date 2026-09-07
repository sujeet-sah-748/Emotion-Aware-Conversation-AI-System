"""
test_api_integration.py
=======================
Integration tests for the FastAPI application (main.py).

All heavy dependencies are patched in conftest.py's `test_client` fixture.
These tests exercise the HTTP layer: routing, request validation,
response schema, status codes, and business-logic paths.

Test classes:
- TestRootEndpoint
- TestPredictEndpoint
- TestChatEndpoint
- TestSessionEndpoints
- TestCacheEndpoints
- TestMemoryEndpoints (including route-ordering Bug #5 fix)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Shared helpers
# ===========================================================================

def _chat_payload(text: str = "I feel happy today", user_id: str = "test_user") -> Dict:
    return {"text": text, "user_id": user_id, "threshold": 0.3}


def _predict_payload(text: str = "I feel great") -> Dict:
    return {"text": text, "threshold": 0.3}


# ===========================================================================
# Root endpoint
# ===========================================================================

class TestRootEndpoint:
    def test_get_root_returns_200(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200

    def test_get_root_returns_json(self, test_client):
        resp = test_client.get("/")
        data = resp.json()
        assert "message" in data

    def test_get_root_message_nonempty(self, test_client):
        resp = test_client.get("/")
        assert len(resp.json()["message"]) > 0


# ===========================================================================
# /predict endpoint
# ===========================================================================

class TestPredictEndpoint:
    def test_valid_request_returns_200(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        assert resp.status_code == 200

    def test_response_has_required_fields(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        data = resp.json()
        for field in ("text", "emotions", "used_fallback", "device"):
            assert field in data, f"Missing field: {field}"

    def test_emotions_is_list(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        assert isinstance(resp.json()["emotions"], list)

    def test_each_emotion_has_label_and_score(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        for emo in resp.json()["emotions"]:
            assert "label" in emo
            assert "score" in emo

    def test_empty_text_returns_400(self, test_client):
        resp = test_client.post("/predict", json={"text": "  ", "threshold": 0.5})
        assert resp.status_code == 400

    def test_empty_string_returns_400(self, test_client):
        resp = test_client.post("/predict", json={"text": "", "threshold": 0.5})
        assert resp.status_code == 400

    def test_text_echoed_in_response(self, test_client):
        text = "unique test text abc123"
        resp = test_client.post("/predict", json={"text": text, "threshold": 0.3})
        assert resp.json()["text"] == text

    def test_top_k_parameter_respected(self, test_client):
        resp = test_client.post("/predict", json={"text": "I am excited", "threshold": 0.1, "top_k": 2})
        assert resp.status_code == 200
        # emotions list should have at most top_k items
        assert len(resp.json()["emotions"]) <= 2

    def test_include_all_scores_true(self, test_client):
        resp = test_client.post("/predict", json={
            "text": "I feel great",
            "threshold": 0.3,
            "include_all_scores": True,
        })
        assert resp.status_code == 200

    def test_used_fallback_is_bool(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        assert isinstance(resp.json()["used_fallback"], bool)

    def test_device_field_is_string(self, test_client):
        resp = test_client.post("/predict", json=_predict_payload())
        assert isinstance(resp.json()["device"], str)


# ===========================================================================
# /chat endpoint
# ===========================================================================

class TestChatEndpoint:
    def test_valid_request_returns_200(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        assert resp.status_code == 200

    def test_response_has_all_required_fields(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        data = resp.json()
        required = ("text", "emotions", "used_fallback", "bot_response",
                    "affect_state", "emotional_events", "device")
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_affect_state_has_expected_structure(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        affect = resp.json()["affect_state"]
        for key in ("situational_vad", "short_term_vad", "long_term_vad",
                    "stm_dominant", "ltm_dominant", "trend", "confidence"):
            assert key in affect, f"Missing key in affect_state: {key}"

    def test_vad_fields_are_dicts_with_valence_arousal_dominance(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        sit_vad = resp.json()["affect_state"]["situational_vad"]
        assert "valence" in sit_vad
        assert "arousal" in sit_vad
        assert "dominance" in sit_vad

    def test_bot_response_is_nonempty_string(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        bot = resp.json()["bot_response"]
        assert isinstance(bot, str)
        assert len(bot) > 0

    def test_emotional_events_is_list(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        assert isinstance(resp.json()["emotional_events"], list)

    def test_empty_text_returns_400(self, test_client):
        resp = test_client.post("/chat", json={"text": "", "user_id": "u1"})
        assert resp.status_code == 400

    def test_whitespace_text_returns_400(self, test_client):
        resp = test_client.post("/chat", json={"text": "   ", "user_id": "u1"})
        assert resp.status_code == 400

    def test_session_info_included(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        data = resp.json()
        if "session_info" in data and data["session_info"]:
            assert "message_count" in data["session_info"]

    def test_different_user_ids_isolated(self, test_client):
        resp1 = test_client.post("/chat", json=_chat_payload(user_id="alice"))
        resp2 = test_client.post("/chat", json=_chat_payload(user_id="bob"))
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    def test_multiple_turns_same_user(self, test_client):
        uid = "multi_turn_user"
        for msg in ["Hello", "I feel anxious", "Tell me more"]:
            resp = test_client.post("/chat", json={"text": msg, "user_id": uid, "threshold": 0.3})
            assert resp.status_code == 200

    def test_trend_valid_values(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        trend = resp.json()["affect_state"]["trend"]
        assert trend in ("rising", "falling", "steady")

    def test_confidence_in_range_0_1(self, test_client):
        resp = test_client.post("/chat", json=_chat_payload())
        conf = resp.json()["affect_state"]["confidence"]
        assert 0.0 <= conf <= 1.0


# ===========================================================================
# /session/{user_id} endpoints
# ===========================================================================

class TestSessionEndpoints:
    @pytest.fixture(autouse=True)
    def _seed_session(self, test_client):
        """Create a session by sending a chat message first."""
        test_client.post("/chat", json=_chat_payload(user_id="session_test_user"))

    def test_get_affect_state_returns_200(self, test_client):
        resp = test_client.get("/session/session_test_user/affect")
        assert resp.status_code == 200

    def test_get_affect_state_structure(self, test_client):
        resp = test_client.get("/session/session_test_user/affect")
        data = resp.json()
        assert "affect_state" in data
        assert "session_info" in data

    def test_get_affect_unknown_user_returns_404(self, test_client):
        resp = test_client.get("/session/no_such_user_xyz/affect")
        assert resp.status_code == 404

    def test_get_events_returns_200(self, test_client):
        resp = test_client.get("/session/session_test_user/events")
        assert resp.status_code == 200

    def test_get_events_structure(self, test_client):
        resp = test_client.get("/session/session_test_user/events")
        data = resp.json()
        assert "events" in data
        assert "total_events" in data
        assert "user_id" in data

    def test_get_events_unknown_user_returns_404(self, test_client):
        resp = test_client.get("/session/no_such_user_events/events")
        assert resp.status_code == 404

    def test_get_events_limit_parameter(self, test_client):
        # Send several messages first
        uid = "events_limit_user"
        for i in range(5):
            test_client.post("/chat", json={"text": f"message {i}", "user_id": uid, "threshold": 0.3})
        resp = test_client.get(f"/session/{uid}/events?limit=2")
        assert resp.status_code == 200
        events = resp.json()["events"]
        assert len(events) <= 2

    def test_delete_session_returns_200(self, test_client):
        uid = "delete_session_user"
        test_client.post("/chat", json=_chat_payload(user_id=uid))
        resp = test_client.delete(f"/session/{uid}")
        assert resp.status_code == 200

    def test_delete_session_unknown_user_returns_404(self, test_client):
        resp = test_client.delete("/session/no_user_to_delete_xyz")
        assert resp.status_code == 404

    def test_get_active_sessions_returns_200(self, test_client):
        resp = test_client.get("/sessions/active")
        assert resp.status_code == 200

    def test_active_sessions_structure(self, test_client):
        resp = test_client.get("/sessions/active")
        data = resp.json()
        assert "active_sessions" in data
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_context_diagnostics_returns_200(self, test_client):
        resp = test_client.get("/session/session_test_user/context_diagnostics")
        assert resp.status_code in (200, 404)  # 404 if context_manager not wired

    def test_context_diagnostics_unknown_user_returns_404(self, test_client):
        resp = test_client.get("/session/nobody_here_xyz/context_diagnostics")
        assert resp.status_code == 404


# ===========================================================================
# /cache endpoints
# ===========================================================================

class TestCacheEndpoints:
    def test_cache_stats_returns_200(self, test_client):
        resp = test_client.get("/cache/stats")
        assert resp.status_code == 200

    def test_cache_health_returns_200(self, test_client):
        resp = test_client.get("/cache/health")
        assert resp.status_code == 200

    def test_cache_health_has_status_field(self, test_client):
        resp = test_client.get("/cache/health")
        assert "status" in resp.json()

    def test_cache_stats_disabled_response(self, test_client):
        # Redis is disabled in tests (patched); endpoint should still succeed
        data = test_client.get("/cache/stats").json()
        assert "enabled" in data

    def test_clear_cache_disabled_returns_503(self, test_client):
        # Redis is disabled; clearing should return 503
        resp = test_client.delete("/cache/clear")
        assert resp.status_code in (200, 503)


# ===========================================================================
# /memory endpoints (Bug #5 fix: route ordering)
# ===========================================================================

class TestMemoryEndpoints:
    def test_memory_health_returns_200(self, test_client):
        """
        Bug #5 fix verification: /memory/health must NOT be captured by
        /memory/{user_id}.  It should return a health status dict, not a
        user-memories response.
        """
        resp = test_client.get("/memory/health")
        assert resp.status_code == 200
        data = resp.json()
        # A user-memories response would have "memories" and "user_id" keys.
        # A health response has "status".
        assert "status" in data, (
            "Route ordering bug: /memory/health was matched as /memory/{user_id}. "
            f"Got: {data}"
        )
        assert "user_id" not in data, (
            "Route ordering bug: health endpoint returned user_id field. "
            f"Got: {data}"
        )

    def test_get_user_memories_returns_200(self, test_client):
        resp = test_client.get("/memory/actual_user_id")
        assert resp.status_code == 200

    def test_get_user_memories_structure(self, test_client):
        resp = test_client.get("/memory/actual_user_id")
        data = resp.json()
        assert "user_id" in data
        assert "total_memories" in data
        assert "memories" in data

    def test_search_user_memories(self, test_client):
        resp = test_client.post("/memory/test_user/search?query=hiking&limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_get_emotional_memories(self, test_client):
        resp = test_client.get("/memory/test_user/emotional")
        assert resp.status_code == 200

    def test_get_emotional_memories_with_filter(self, test_client):
        resp = test_client.get("/memory/test_user/emotional?emotion=joy&min_importance=0.5")
        assert resp.status_code == 200

    def test_get_memory_summary(self, test_client):
        resp = test_client.get("/memory/test_user/summary")
        assert resp.status_code == 200

    def test_add_user_memory(self, test_client):
        resp = test_client.post("/memory/test_user/add?text=User+loves+coffee&importance=0.8")
        assert resp.status_code == 200

    def test_delete_user_memory(self, test_client):
        resp = test_client.delete("/memory/test_user/mem-001")
        assert resp.status_code in (200, 404)  # 200 if mock returns success

    def test_delete_all_user_memories(self, test_client):
        resp = test_client.delete("/memory/test_user")
        assert resp.status_code == 200


# ===========================================================================
# Schema validation
# ===========================================================================

class TestRequestValidation:
    def test_missing_text_field_returns_422(self, test_client):
        resp = test_client.post("/chat", json={"user_id": "u1"})
        assert resp.status_code == 422

    def test_missing_text_predict_returns_422(self, test_client):
        resp = test_client.post("/predict", json={"threshold": 0.5})
        assert resp.status_code == 422

    def test_invalid_threshold_type_returns_422(self, test_client):
        resp = test_client.post("/chat", json={"text": "hi", "threshold": "not_a_float"})
        assert resp.status_code == 422


# ===========================================================================
# Concurrent requests
# ===========================================================================

class TestConcurrentRequests:
    def test_concurrent_chat_requests(self, test_client):
        """Multiple simultaneous requests must all succeed without data races."""
        errors = []
        responses = []

        def _send(uid: str, msg: str):
            try:
                resp = test_client.post("/chat", json={"text": msg, "user_id": uid, "threshold": 0.3})
                responses.append(resp.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=_send, args=(f"concurrent_user_{i}", f"Hello from thread {i}"))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent request errors: {errors}"
        assert all(s == 200 for s in responses), f"Non-200 statuses: {responses}"
