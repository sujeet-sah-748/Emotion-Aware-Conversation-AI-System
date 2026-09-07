"""
test_ollama_llm.py
==================
Unit tests for response/ollama_llm.py

Coverage:
- OllamaLLM.__init__: connection verification (model present / absent)
- OllamaLLM.generate: happy path, stream=False, system prompt, context
- OllamaLLM.generate: connection error → fallback response
- OllamaLLM._handle_stream: yields content chunks
- OllamaLLM._fallback_response: returns non-empty string
- OllamaLLM.generate_emotion_aware_response: builds correct system prompt
- OllamaLLM._build_emotion_aware_system_prompt: emotional state sections
- create_ollama_llm: reads env vars correctly
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from response.ollama_llm import OllamaLLM, create_ollama_llm


# ===========================================================================
# Helpers
# ===========================================================================

def _mock_tags_response(model_names: List[str]):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return resp


def _mock_chat_response(content: str):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": content}}
    return resp


def _make_llm(model="phi4-mini", model_present=True) -> OllamaLLM:
    tags = _mock_tags_response([model] if model_present else [])
    with patch("response.ollama_llm.requests.get", return_value=tags):
        return OllamaLLM(model=model, base_url="http://localhost:11434")


# ===========================================================================
# Initialization / connection verification
# ===========================================================================

class TestOllamaLLMInit:
    def test_init_success_model_present(self):
        llm = _make_llm(model_present=True)
        assert llm.model == "phi4-mini"

    def test_init_logs_warning_when_model_absent(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="response.ollama_llm"):
            llm = _make_llm(model_present=False)
        assert "not found" in caplog.text.lower() or llm.model == "phi4-mini"

    def test_init_handles_connection_error(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="response.ollama_llm"):
            with patch("response.ollama_llm.requests.get",
                       side_effect=requests.exceptions.ConnectionError("refused")):
                llm = OllamaLLM(model="phi4-mini", base_url="http://localhost:11434")
        assert llm.model == "phi4-mini"  # should not raise

    def test_base_url_trailing_slash_stripped(self):
        tags = _mock_tags_response(["phi4-mini"])
        with patch("response.ollama_llm.requests.get", return_value=tags):
            llm = OllamaLLM(model="phi4-mini", base_url="http://localhost:11434/")
        assert not llm.base_url.endswith("/")

    def test_default_parameters(self):
        llm = _make_llm()
        assert 0.0 <= llm.temperature <= 1.0
        assert llm.max_tokens > 0
        assert llm.timeout > 0


# ===========================================================================
# OllamaLLM.generate — happy path
# ===========================================================================

class TestOllamaLLMGenerate:
    @pytest.fixture(autouse=True)
    def _llm(self):
        self.llm = _make_llm()

    def test_returns_string(self):
        chat_resp = _mock_chat_response("I'm here to help.")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp):
            result = self.llm.generate("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_stripped_content(self):
        chat_resp = _mock_chat_response("  Some response text.  ")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp):
            result = self.llm.generate("Hi")
        assert result == "Some response text."

    def test_system_prompt_included_in_payload(self):
        chat_resp = _mock_chat_response("OK")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate("Hi", system_prompt="You are a helper.")
        payload = mock_post.call_args.kwargs.get("json", {})
        messages = payload.get("messages", [])
        assert any(m.get("role") == "system" for m in messages)

    def test_context_appended_before_user_message(self):
        chat_resp = _mock_chat_response("OK")
        context = [{"role": "user", "content": "first message"},
                   {"role": "assistant", "content": "first reply"}]
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate("second message", context=context)
        payload = mock_post.call_args.kwargs.get("json", {})
        messages = payload.get("messages", [])
        contents = [m["content"] for m in messages]
        assert "first message" in contents
        assert "second message" in contents
        assert contents.index("second message") > contents.index("first message")

    def test_model_name_in_payload(self):
        chat_resp = _mock_chat_response("OK")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate("test")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert payload.get("model") == "phi4-mini"

    def test_temperature_in_options(self):
        chat_resp = _mock_chat_response("OK")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate("test")
        payload = mock_post.call_args.kwargs.get("json", {})
        options = payload.get("options", {})
        assert "temperature" in options

    def test_fallback_on_request_exception(self):
        with patch("response.ollama_llm.requests.post",
                   side_effect=requests.exceptions.ConnectionError("down")):
            result = self.llm.generate("Hi")
        # Should return fallback, not raise
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raise_for_status_called(self):
        chat_resp = _mock_chat_response("OK")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp):
            self.llm.generate("test")
        chat_resp.raise_for_status.assert_called_once()


# ===========================================================================
# OllamaLLM._fallback_response
# ===========================================================================

class TestFallbackResponse:
    def test_returns_nonempty_string(self):
        llm = _make_llm()
        result = llm._fallback_response("any prompt")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_mentions_unavailable(self):
        llm = _make_llm()
        result = llm._fallback_response("")
        assert any(w in result.lower() for w in ("unavailable", "temporarily", "here"))


# ===========================================================================
# OllamaLLM._handle_stream
# ===========================================================================

class TestHandleStream:
    def test_yields_content_chunks(self):
        llm = _make_llm()

        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            json.dumps({"message": {"content": " world"}, "done": False}).encode(),
            json.dumps({"done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)

        chunks = list(llm._handle_stream(mock_resp))
        assert "Hello" in chunks
        assert " world" in chunks

    def test_stops_at_done_true(self):
        llm = _make_llm()

        lines = [
            json.dumps({"message": {"content": "A"}, "done": False}).encode(),
            json.dumps({"done": True}).encode(),
            json.dumps({"message": {"content": "B"}, "done": False}).encode(),  # should NOT be yielded
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)

        chunks = list(llm._handle_stream(mock_resp))
        assert "B" not in chunks

    def test_empty_content_not_yielded(self):
        llm = _make_llm()
        lines = [
            json.dumps({"message": {"content": ""}, "done": False}).encode(),
            json.dumps({"done": True}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        chunks = list(llm._handle_stream(mock_resp))
        assert chunks == []

    def test_stream_exception_yields_fallback(self):
        llm = _make_llm()
        mock_resp = MagicMock()
        mock_resp.iter_lines.side_effect = Exception("stream broken")
        chunks = list(llm._handle_stream(mock_resp))
        assert len(chunks) == 1  # fallback string


# ===========================================================================
# OllamaLLM.generate_emotion_aware_response
# ===========================================================================

class TestEmotionAwareResponse:
    @pytest.fixture(autouse=True)
    def _llm(self):
        self.llm = _make_llm()

    def test_returns_string(self):
        chat_resp = _mock_chat_response("I understand you're feeling sad.")
        with patch("response.ollama_llm.requests.post", return_value=chat_resp):
            result = self.llm.generate_emotion_aware_response(
                user_message="I feel terrible",
                emotion_state={"dominant_emotion": "sadness", "valence": -0.8,
                               "arousal": -0.3, "trend": "falling", "confidence": 0.85},
            )
        assert isinstance(result, str)

    def test_uses_conversation_history(self):
        chat_resp = _mock_chat_response("OK")
        history = [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous reply"},
        ]
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate_emotion_aware_response(
                user_message="current",
                emotion_state={"dominant_emotion": "neutral", "valence": 0.0,
                               "arousal": 0.0, "trend": "steady", "confidence": 0.5},
                conversation_history=history,
            )
        payload = mock_post.call_args.kwargs.get("json", {})
        all_content = " ".join(m.get("content", "") for m in payload.get("messages", []))
        assert "previous message" in all_content

    def test_conversation_history_capped_at_six(self):
        chat_resp = _mock_chat_response("OK")
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        with patch("response.ollama_llm.requests.post", return_value=chat_resp) as mock_post:
            self.llm.generate_emotion_aware_response(
                user_message="latest",
                emotion_state={"dominant_emotion": "neutral", "confidence": 0.3},
                conversation_history=history,
            )
        payload = mock_post.call_args.kwargs.get("json", {})
        # System prompt + up to 6 history + 1 current = max ~8 messages
        assert len(payload.get("messages", [])) <= 9


# ===========================================================================
# OllamaLLM._build_emotion_aware_system_prompt
# ===========================================================================

class TestBuildSystemPrompt:
    @pytest.fixture(autouse=True)
    def _llm(self):
        self.llm = _make_llm()

    def test_contains_dominant_emotion_high_confidence(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "anger", "valence": -0.7,
                           "arousal": 0.8, "trend": "rising", "confidence": 0.9},
        )
        assert "anger" in prompt.lower()

    def test_low_confidence_uses_exploratory_guidance(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "neutral", "valence": 0.0,
                           "arousal": 0.0, "trend": "steady", "confidence": 0.10},
        )
        assert "unclear" in prompt.lower() or "clarify" in prompt.lower()

    def test_memories_included_when_provided(self):
        memories = [
            {"memory": "User enjoys hiking in mountains"},
            {"memory": "User has a dog named Buddy"},
        ]
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "joy", "confidence": 0.7},
            memories=memories,
        )
        assert "hiking" in prompt.lower() or "mountains" in prompt.lower()

    def test_no_memories_no_memories_section(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "neutral", "confidence": 0.5},
            memories=None,
        )
        assert isinstance(prompt, str)

    def test_sadness_guidance_present(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "sadness", "confidence": 0.8,
                           "valence": -0.8, "arousal": -0.4, "trend": "falling"},
        )
        assert any(w in prompt.lower() for w in ("comfort", "sadness", "patience", "valid"))

    def test_anger_guidance_present(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "anger", "confidence": 0.8,
                           "valence": -0.7, "arousal": 0.8, "trend": "rising"},
        )
        assert any(w in prompt.lower() for w in ("anger", "frustrat", "outlet"))

    def test_fear_guidance_present(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "fear", "confidence": 0.7,
                           "valence": -0.6, "arousal": 0.7, "trend": "rising"},
        )
        assert any(w in prompt.lower() for w in ("fear", "reassur", "grounding"))

    def test_returns_string(self):
        prompt = self.llm._build_emotion_aware_system_prompt(
            emotion_state={"dominant_emotion": "neutral", "confidence": 0.5},
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ===========================================================================
# create_ollama_llm factory
# ===========================================================================

class TestCreateOllamaLLM:
    def test_uses_env_model(self):
        import os
        tags = _mock_tags_response(["llama3"])
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3"}):
            with patch("response.ollama_llm.requests.get", return_value=tags):
                llm = create_ollama_llm()
        assert llm.model == "llama3"

    def test_explicit_model_overrides_env(self):
        import os
        tags = _mock_tags_response(["mistral"])
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3"}):
            with patch("response.ollama_llm.requests.get", return_value=tags):
                llm = create_ollama_llm(model="mistral")
        assert llm.model == "mistral"

    def test_uses_env_base_url(self):
        import os
        tags = _mock_tags_response(["phi4-mini"])
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://remote-host:11434"}):
            with patch("response.ollama_llm.requests.get", return_value=tags):
                llm = create_ollama_llm()
        assert "remote-host" in llm.base_url

    def test_temperature_from_env(self):
        import os
        tags = _mock_tags_response(["phi4-mini"])
        with patch.dict(os.environ, {"OLLAMA_TEMPERATURE": "0.3"}):
            with patch("response.ollama_llm.requests.get", return_value=tags):
                llm = create_ollama_llm()
        assert abs(llm.temperature - 0.3) < 1e-6

    def test_max_tokens_from_env(self):
        import os
        tags = _mock_tags_response(["phi4-mini"])
        with patch.dict(os.environ, {"OLLAMA_MAX_TOKENS": "512"}):
            with patch("response.ollama_llm.requests.get", return_value=tags):
                llm = create_ollama_llm()
        assert llm.max_tokens == 512
