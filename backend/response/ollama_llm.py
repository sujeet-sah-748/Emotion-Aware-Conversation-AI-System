"""
ollama_llm.py
=============
Ollama integration for local LLM inference using phi4-mini.

Provides:
- Local LLM inference (no API costs)
- Emotion-aware response generation
- Memory-enhanced conversations
- Streaming support
- Context management
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Generator
import requests

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Ollama client for local LLM inference.
    
    Features:
    - Emotion-aware system prompts
    - Memory integration
    - Streaming responses
    - Conversation context management
    """
    
    def __init__(
        self,
        model: str = "phi4-mini",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        """
        Initialize Ollama LLM client.
        
        Parameters:
        - model: Model name (e.g., "phi4-mini", "llama2", "mistral")
        - base_url: Ollama server URL
        - temperature: Sampling temperature (0.0-1.0)
        - max_tokens: Maximum response length
        - timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama server is running and model is available."""
        try:
            # Check server
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model exists
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(
                    f"Model '{self.model}' not found. Available models: {model_names}. "
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama connected: {self.model} available")
                
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate response from Ollama.
        
        Parameters:
        - prompt: User prompt
        - system_prompt: Optional system prompt
        - context: Optional conversation history [{"role": "user/assistant", "content": "..."}]
        - stream: Enable streaming (returns generator if True)
        
        Returns:
        - Generated text response
        """
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.extend(context)
        
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result["message"]["content"].strip()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return self._fallback_response(prompt)
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        content = data["message"].get("content", "")
                        if content:
                            yield content
                    
                    if data.get("done", False):
                        break
        except Exception as e:
            logger.error(f"Stream handling error: {e}")
            yield self._fallback_response("")
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when Ollama fails."""
        return (
            "I'm here to listen and support you. Could you tell me more about "
            "what's on your mind? (Note: LLM service temporarily unavailable)"
        )
    
    def generate_emotion_aware_response(
        self,
        user_message: str,
        emotion_state: Dict[str, Any],
        memories: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate emotion-aware empathetic response.
        
        Parameters:
        - user_message: User's current message
        - emotion_state: Current emotional state from emotion engine
        - memories: Relevant memories from Mem0
        - conversation_history: Recent conversation turns
        
        Returns:
        - Empathetic response text
        """
        # Build emotion-aware system prompt
        system_prompt = self._build_emotion_aware_system_prompt(
            emotion_state, memories
        )
        
        # Build context with recent conversation
        context = conversation_history[-6:] if conversation_history else []
        
        # Generate response
        return self.generate(
            prompt=user_message,
            system_prompt=system_prompt,
            context=context,
            stream=False,
        )
    
    def _build_emotion_aware_system_prompt(
        self,
        emotion_state: Dict[str, Any],
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build system prompt with emotional context and memories."""
        
        # Extract emotional state
        dominant_emotion = emotion_state.get("dominant_emotion", "neutral")
        valence = emotion_state.get("valence", 0.0)
        arousal = emotion_state.get("arousal", 0.0)
        trend = emotion_state.get("trend", "steady")
        confidence = emotion_state.get("confidence", 0.5)
        
        # Build base prompt
        prompt_parts = [
            "You are an empathetic AI assistant specializing in emotional support and mental well-being.",
            "Your responses should be warm, understanding, and validating.",
            "",
            "CURRENT EMOTIONAL STATE (model estimate - treat as guidance, not fact):",
        ]
        
        # Add emotion details
        if confidence > 0.3:
            prompt_parts.append(f"- Dominant emotion: {dominant_emotion}")
            prompt_parts.append(f"- Emotional valence: {valence:.2f} (negative to positive)")
            prompt_parts.append(f"- Arousal level: {arousal:.2f} (calm to excited)")
            prompt_parts.append(f"- Trend: {trend}")
            prompt_parts.append(f"- Confidence: {confidence:.2f}")
        else:
            prompt_parts.append("- Emotional state unclear - listen carefully and ask clarifying questions")
        
        prompt_parts.append("")
        
        # Add memories if available
        if memories and len(memories) > 0:
            prompt_parts.append("RELEVANT MEMORIES FROM PAST CONVERSATIONS:")
            for i, mem in enumerate(memories[:5], 1):
                memory_text = mem.get("memory", mem.get("content", ""))
                if memory_text:
                    prompt_parts.append(f"{i}. {memory_text}")
            prompt_parts.append("")
        
        # Add response guidelines
        prompt_parts.extend([
            "RESPONSE GUIDELINES:",
            "1. Acknowledge and validate the user's emotions without judgment",
            "2. Show empathy and understanding",
            "3. Ask clarifying questions when needed",
            "4. Offer support and gentle guidance",
            "5. Keep responses concise but meaningful (2-4 sentences)",
            "6. Use memories to provide personalized, context-aware responses",
            "7. Match the emotional tone appropriately:",
        ])
        
        # Emotion-specific guidance
        if dominant_emotion in ["sadness", "grief", "disappointment"]:
            prompt_parts.append("   - For sadness: Offer comfort, validate their feelings, be patient")
        elif dominant_emotion in ["anger", "annoyance", "frustration"]:
            prompt_parts.append("   - For anger: Acknowledge frustration, help identify constructive outlets")
        elif dominant_emotion in ["fear", "nervousness", "anxiety"]:
            prompt_parts.append("   - For fear: Provide reassurance, help break down concerns, offer grounding")
        elif dominant_emotion in ["joy", "excitement", "gratitude"]:
            prompt_parts.append("   - For joy: Share in their happiness, encourage celebration")
        else:
            prompt_parts.append("   - Stay neutral and exploratory, help them articulate their feelings")
        
        prompt_parts.extend([
            "",
            "Remember: You're here to listen, validate, and support - not to diagnose or provide therapy.",
            "Be human, warm, and genuine in your responses.",
        ])
        
        return "\n".join(prompt_parts)


# =============================================================================
# Factory Function
# =============================================================================

def create_ollama_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaLLM:
    """
    Create Ollama LLM client from environment variables.
    
    Environment variables:
    - OLLAMA_MODEL: Model name (default: phi4-mini)
    - OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    - OLLAMA_TEMPERATURE: Sampling temperature (default: 0.7)
    - OLLAMA_MAX_TOKENS: Max response length (default: 1024)
    """
    return OllamaLLM(
        model=model or os.getenv("OLLAMA_MODEL", "phi4-mini"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "1024")),
    )


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("OLLAMA LLM - SMOKE TEST")
    print("="*70)
    
    # Create LLM
    llm = create_ollama_llm()
    
    # Test basic generation
    print("\n--- Basic Generation ---")
    response = llm.generate("Hello, how are you?")
    print(f"Response: {response}")
    
    # Test emotion-aware generation
    print("\n--- Emotion-Aware Generation ---")
    emotion_state = {
        "dominant_emotion": "sadness",
        "valence": -0.7,
        "arousal": -0.3,
        "trend": "falling",
        "confidence": 0.8,
    }
    
    memories = [
        {"memory": "User mentioned feeling lonely recently"},
        {"memory": "User enjoys playing tennis on weekends"},
    ]
    
    response = llm.generate_emotion_aware_response(
        user_message="I've been feeling really down lately. Nothing seems to help.",
        emotion_state=emotion_state,
        memories=memories,
    )
    print(f"Empathetic response: {response}")
    
    print("\n✅ All tests completed")
