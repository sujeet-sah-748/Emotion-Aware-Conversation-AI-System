import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

# =============================================================================
# REDUNDANCY FIX #1: Use classifier from emotion_engine (single model loading)
# =============================================================================
from core.emotion_engine import (
    FinalAdapterClassifier,
    GOEMOTIONS_LABELS,  # REDUNDANCY FIX #2: Use canonical label source
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
SIGMOID_THRESHOLD = 0.5

# REDUNDANCY FIX #2: Use canonical labels from emotion_engine
LABEL_NAMES = list(GOEMOTIONS_LABELS)

# =============================================================================
# MODEL INITIALIZATION (Single instance, shared across application)
# =============================================================================
print("Initializing emotion classifier from emotion_engine...")
try:
    # REDUNDANCY FIX #1: Use FinalAdapterClassifier (no duplicate model loading)
    classifier = FinalAdapterClassifier()
    device = classifier._device
    print(f"Emotion classifier initialized successfully on {device}!")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {e}")
    raise

# =============================================================================
# PYDANTIC MODELS (Keep only ones used by main.py)
# REDUNDANCY FIX #5: Removed unused ChatRequest and ChatResponse models
# =============================================================================
class TextRequest(BaseModel):
    text: str
    threshold: float = SIGMOID_THRESHOLD
    top_k: Optional[int] = None
    # When True, always returns every label's raw score regardless of threshold/top_k.
    include_all_scores: bool = False

class EmotionResponse(BaseModel):
    text: str
    emotions: List[Dict[str, Any]]
    used_fallback: bool
    all_scores: Optional[List[Dict[str, Any]]] = None
    device: str

# =============================================================================
# EMOTION PREDICTION (REDUNDANCY FIX #3: Unified with emotion_engine)
# =============================================================================
class SimpleMessage:
    """Lightweight message wrapper for classifier compatibility."""
    def __init__(self, content: str):
        self.content = content

def predict_emotions(
    text: str,
    threshold: float,
    top_k: Optional[int] = None,
) -> Tuple[list, bool, list]:
    """
    Predict emotions using the unified FinalAdapterClassifier.
    
    REDUNDANCY FIX #3: This now uses the same classifier as EmotionManager,
    ensuring consistent predictions across /predict and /chat endpoints.
    """
    # Use classifier from emotion_engine (single source of truth)
    msg = SimpleMessage(text)
    signal = classifier.classify([msg])
    
    # Convert EmotionSignal to expected format
    all_results = sorted(
        [{"label": label, "score": float(score)} 
         for label, score in signal.label_probs.items()],
        key=lambda x: -x["score"],
    )
    
    # Handle top_k filtering
    if top_k:
        return all_results[:top_k], False, all_results
    
    # Handle threshold filtering
    above_threshold = [r for r in all_results if r["score"] >= threshold]
    if above_threshold:
        return above_threshold, False, all_results
    
    # Fallback: nothing crossed threshold — return highest-scoring label
    # used_fallback=True lets the caller distinguish "genuinely neutral" from
    # "model unsure, picked best guess"
    return [all_results[0]], True, all_results

# =============================================================================
# RESPONSE GENERATION
# =============================================================================
class ResponseGenerator:
    def __init__(self):
        self.negative_emotions = {
            'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'confusion'
        }
        self.positive_emotions = {
            'joy', 'excitement', 'admiration', 'amusement', 'approval',
            'gratitude', 'love', 'optimism', 'pride', 'relief', 'caring',
            'curiosity', 'desire', 'surprise', 'realization'
        }

    def generate(
        self,
        text: str,
        emotions: List[Dict[str, Any]],
        used_fallback: bool,
        all_scores: List[Dict[str, Any]],
    ) -> str:
        """
        Generates a supportive response based on the emotion distribution.
        Decision order:
          1. Mixed state — negative top + hidden positive signal
          2. Low confidence / fallback
          3. Direct response for the top label
        """
        if not emotions:
            return "I'm here and I'm listening. Take your time, and tell me what's on your mind."

        top_emotion = emotions[0]["label"]
        top_score   = emotions[0]["score"]

        # 1. Mixed state: negative top emotion but a positive signal is also present
        hidden_positives = [
            e for e in all_scores
            if e["label"] in self.positive_emotions and e["score"] > 0.15
        ]
        if top_emotion in self.negative_emotions and hidden_positives:
            return self._generate_mixed_reframe(
                top_emotion, hidden_positives[0]["label"], text
            )

        # 2. Low confidence
        if used_fallback or top_score < 0.35:
            return self._generate_uncertain_response(text)

        # 3. Direct response
        if top_emotion in self.negative_emotions:
            return self._generate_negative_reframe(top_emotion, text)
        return self._generate_positive_response(top_emotion, text)

    def _generate_negative_reframe(self, emotion: str, text: str) -> str:
        templates = {
            "anger": (
                "It is completely understandable to feel angry when things feel unfair or out of "
                "your control. Your frustration is valid. While we can't always change the situation "
                "immediately, we can look at what boundaries or next steps might help you regain a "
                "sense of power. What do you think would be a fair resolution?"
            ),
            "annoyance": (
                "Those persistent, grinding frustrations are exhausting, and it makes total sense "
                "that you're feeling worn down by this. You've been patient for long enough. Let's "
                "take a breath — what is one small thing we can adjust right now to make this easier?"
            ),
            "disappointment": (
                "Disappointment carries a heavy, quiet sting, especially when you had genuine hope "
                "for a different outcome. I'm truly sorry it didn't go as planned. Sometimes "
                "acknowledging that hurt is the first step. What is one thing you can take forward?"
            ),
            "disapproval": (
                "It sounds like something crossed a real line for you, and your frustration is "
                "completely justified. It's hard when basic expectations of fairness or decency "
                "aren't met. Can you tell me more about what happened?"
            ),
            "disgust": (
                "It sounds like something genuinely disturbed or offended you, and that reaction "
                "is telling — our gut feelings about things like this usually matter. What happened, "
                "and what bothered you most about it?"
            ),
            "fear": (
                "It is completely okay to feel scared. Fear is your mind's way of trying to protect "
                "you, even when it feels overwhelming. You are safe here. Let's break this down "
                "together — what is the very first, smallest step we can take to make this more manageable?"
            ),
            "nervousness": (
                "That knotted, anxious feeling is so real, especially when something matters deeply "
                "to you. The fact that you care this much is a strength, not a weakness. Let's focus "
                "on what you can control. What is one thing that usually helps you feel grounded?"
            ),
            "sadness": (
                "I hear how heavy this feels, and I want to validate that sadness. You don't have to "
                "rush past it or force positivity right now. You don't have to carry it alone either — "
                "I'm here with you. Would it help to talk about what's weighing on you most?"
            ),
            "grief": (
                "Grief is one of the most profound and disorienting experiences, and there is "
                "absolutely no 'right' way to navigate it. Please be gentle with yourself right now. "
                "I am holding space for you, and I am listening without any judgment."
            ),
            "embarrassment": (
                "Those moments where we want the ground to swallow us whole are universally human, "
                "even if they feel deeply personal in the moment. The fact that you're reflecting on "
                "it shows great self-awareness. Try to offer yourself the same grace you'd give a friend."
            ),
            "remorse": (
                "It sounds like you're carrying a heavy weight of regret, and that shows how deeply "
                "you care about doing the right thing. Be careful not to let self-criticism overshadow "
                "your capacity to grow. What is one small, constructive action you can take today?"
            ),
            "confusion": (
                "It is completely normal to feel lost when things aren't making sense. That mental fog "
                "can be exhausting. Let's untangle this together, one piece at a time. What is the "
                "very first thing that feels unclear to you?"
            ),
        }
        return templates.get(
            emotion,
            f"I hear that you're feeling {emotion}, and that is completely valid. "
            f"Let's explore this together — what do you need most right now?"
        )

    def _generate_mixed_reframe(
        self, negative_emotion: str, positive_emotion: str, text: str
    ) -> str:
        return (
            f"It sounds like you're holding a really complex mix of feelings right now. "
            f"It is completely understandable to feel {negative_emotion} given the situation, "
            f"but I also notice a genuine sense of {positive_emotion} in what you're sharing. "
            f"Big, meaningful moments rarely come with just one emotion. "
            f"Both of these feelings are valid. Which of the two feels like it needs your attention most right now?"
        )

    def _generate_positive_response(self, emotion: str, text: str) -> str:
        templates = {
            "joy": (
                "That is genuinely wonderful to hear! Your joy is contagious. It sounds like "
                "something really beautiful is happening in your life. Tell me more about what's "
                "making you feel this way!"
            ),
            "excitement": (
                "That excitement is so palpable! New beginnings and milestones carry such an "
                "electric mix of anticipation and possibility. Tell me everything — what's happening?"
            ),
            "gratitude": (
                "Holding onto that kind of gratitude is a beautiful thing. It sounds like someone "
                "or something made a real, positive difference for you. What are you feeling most "
                "thankful for right now?"
            ),
            "pride": (
                "You absolutely should feel proud. What you've accomplished clearly took real effort, "
                "courage, and dedication. That feeling is hard-earned and completely deserved. "
                "How are you going to celebrate this?"
            ),
            "relief": (
                "Oh, that sense of relief when a heavy weight finally lifts — there's nothing quite "
                "like it. It sounds like you've been carrying something stressful for a while. "
                "How does it feel to finally have this behind you?"
            ),
            "admiration": (
                "It's wonderful when someone or something genuinely moves you like that. That kind "
                "of admiration says a lot about your own values. What stood out to you the most?"
            ),
            "optimism": (
                "That sense of hope is a powerful thing. It sounds like you're genuinely looking "
                "forward to what's ahead, and that mindset will serve you well. "
                "What are you most excited to see unfold?"
            ),
            "love": (
                "That warmth and love you're feeling is something really special. Whether it's for "
                "a person, a place, or something you're passionate about, it clearly means a great "
                "deal to you. Tell me more about it."
            ),
            "amusement": (
                "Ha! Those little moments of unexpected absurdity are the best. It's great that you "
                "can find the humour in it. What happened?"
            ),
            "curiosity": (
                "That kind of genuine curiosity is infectious — it sounds like your mind just lit up. "
                "There's something energising about going down a rabbit hole on something that "
                "fascinates you. What is it you want to explore first?"
            ),
            "caring": (
                "The depth of care you have for this comes through clearly, and that kind of "
                "investment matters. It takes a lot to show up for people the way you do. "
                "What's on your mind?"
            ),
            "surprise": (
                "That sounds like it came completely out of nowhere! Surprises — good or unsettling "
                "— can really knock you off balance. How are you sitting with it now?"
            ),
            "realization": (
                "Those moments when something suddenly clicks into place can be really powerful. "
                "It sounds like something just landed for you. What did you realise, and how does "
                "it feel now that you see it?"
            ),
            "approval": (
                "It sounds like something really resonated with you, and that feeling of alignment "
                "is worth paying attention to. What is it that clicked?"
            ),
            "desire": (
                "It's clear there's something you really want, and that longing is worth taking "
                "seriously. Sometimes just naming what we want is the first step. "
                "What would it mean to you if you got it?"
            ),
        }
        return templates.get(
            emotion,
            f"That's wonderful to hear! It sounds like you're experiencing {emotion}. "
            f"Tell me more about what's making you feel this way!"
        )

    def _generate_uncertain_response(self, text: str) -> str:
        return (
            "I want to make sure I'm truly understanding how you're feeling, because what you're "
            "going through clearly matters. It sounds like there might be a lot going on beneath "
            "the surface. Can you tell me a bit more about what's been happening and how it's "
            "been affecting you?"
        )


# Module-level singleton — imported and used by main.py
response_generator = ResponseGenerator()


def compose_response(
    text: str,
    emotions: List[Dict[str, Any]],
    used_fallback: bool,
    all_scores: List[Dict[str, Any]],
) -> str:
    """
    Public function called by main.py.
    Delegates to the ResponseGenerator singleton.
    """
    return response_generator.generate(text, emotions, used_fallback, all_scores)
