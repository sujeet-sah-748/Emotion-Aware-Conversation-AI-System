import random
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Absolute path derived from this file's location — works regardless of where
# uvicorn is launched from (project root, backend/, etc.)
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "final_adapter")

MAX_LENGTH = 128
SIGMOID_THRESHOLD = 0.5

# Exact 28 GoEmotions labels in confirmed training order (index 0-27).
# Must match id2label used during training or predictions silently misalign.
LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
label2id = {name: i for i, name in enumerate(LABEL_NAMES)}


# =============================================================================
# MODEL LOADING (Runs once at startup)
# =============================================================================
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL_NAMES),
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)

try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
except Exception as e:
    raise RuntimeError(
        f"Failed to load LoRA adapter from '{ADAPTER_DIR}'. "
        f"If predictions always return 'neutral' with no adapter loaded, this is why -- "
        f"an untrained/base head defaults to the majority class. Original error: {e}"
    )

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on {device}!")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class TextRequest(BaseModel):
    text: str
    threshold: float = SIGMOID_THRESHOLD
    top_k: Optional[int] = None
    # When True, always returns every label's raw score regardless of threshold/top_k.
    # Use this to check whether e.g. "confusion" is sitting just under threshold
    # (model is learning it, just needs a lower threshold or more training) vs
    # near-zero (model hasn't learned it at all, needs class-weighted loss / more data).
    include_all_scores: bool = False

class EmotionResponse(BaseModel):
    text: str
    emotions: List[Dict[str, Any]]
    used_fallback: bool
    all_scores: Optional[List[Dict[str, Any]]] = None
    device: str

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@torch.no_grad()
def predict_emotions(
    text: str,
    threshold: float,
    top_k: Optional[int] = None,
) -> Tuple[list, bool, list]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits.squeeze(0).cpu().numpy()
    probs = sigmoid(logits)

    all_results = sorted(
        [{"label": id2label[i], "score": float(p)} for i, p in enumerate(probs)],
        key=lambda x: -x["score"],
    )

    if top_k:
        return all_results[:top_k], False, all_results

    above_threshold = [r for r in all_results if r["score"] >= threshold]
    if above_threshold:
        return above_threshold, False, all_results

    # Fallback: nothing crossed threshold. Returning the single highest-scoring label.
    # used_fallback=True lets the caller distinguish "genuinely neutral" from
    # "model unsure, picked best guess".
    return [all_results[0]], True, all_results

class ResponseGenerator:
    def __init__(self):
        # Define which emotions are considered "negative" and need gentle reframing
        self.negative_emotions = {
            'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'confusion'
        }
        self.positive_emotions = {
            'joy', 'excitement', 'admiration', 'amusement', 'approval',
            'gratitude', 'love', 'optimism', 'pride', 'relief', 'caring', 'curiosity', 'desire', 'surprise', 'realization'
        }

    def generate(self, text: str, emotions: List[Dict[str, Any]], used_fallback: bool, all_scores: List[Dict[str, Any]]) -> str:
        """
        Generates a generous, balanced response based on the emotion distribution.
        """
        if not emotions:
            return "I'm here and I'm listening. Take your time, and tell me what's on your mind."

        top_emotion = emotions[0]['label']
        top_score = emotions[0]['score']

        # 1. Check for Mixed States (Negative top, but hidden positive signals)
        # This is where we find the "positive side" the user asked for!
        hidden_positives = [e for e in all_scores if e['label'] in self.positive_emotions and e['score'] > 0.15]
        
        if top_emotion in self.negative_emotions and hidden_positives:
            return self._generate_mixed_reframe(top_emotion, hidden_positives[0]['label'], text)

        # 2. Low Confidence / Fallback
        if used_fallback or top_score < 0.35:
            return self._generate_uncertain_response(text)

        # 3. Direct Response based on top emotion
        if top_emotion in self.negative_emotions:
            return self._generate_negative_reframe(top_emotion, text)
        else:
            return self._generate_positive_response(top_emotion, text)

    # =========================================================================
    # NEGATIVE REFRAMING: Validate + Normalize + Gentle Pivot
    # =========================================================================
    def _generate_negative_reframe(self, emotion: str, text: str) -> str:
        templates = {
            'anger': "It is completely understandable to feel angry when things feel unfair or out of your control. Your frustration is valid. While we can't always change the situation immediately, we can look at what boundaries or next steps might help you regain a sense of power. What do you think would be a fair resolution?",
            'annoyance': "Those persistent, grinding frustrations are exhausting, and it makes total sense that you're feeling worn down by this. You've been patient for long enough. Let's take a breath—what is one small thing we can adjust right now to make this a little easier?",
            'disappointment': "Disappointment carries a heavy, quiet sting, especially when you had genuine hope for a different outcome. I'm truly sorry it didn't go as planned. Sometimes, acknowledging that hurt is the first step. What is one thing you learned from this experience that you can take forward?",
            'fear': "It is completely okay to feel scared. Fear is just your mind's way of trying to protect you, even when it feels overwhelming. You are safe here. Let's break this down together: what is the very first, smallest step we can take to make this feel more manageable?",
            'nervousness': "That knotted, anxious feeling is so real, especially when something matters deeply to you. The fact that you care this much is actually a strength, not a weakness. Let's focus on what you can control right now. What is one thing that usually helps you feel grounded?",
            'sadness': "I hear how heavy this feels, and I want to validate that sadness. You don't have to rush past it or force yourself to 'be positive' right now. Just know that you don't have to carry it alone. I'm here with you. Would it help to talk about what's weighing on you most?",
            'grief': "Grief is one of the most profound and disorienting experiences, and there is absolutely no 'right' way to navigate it. Please be gentle with yourself right now. I am holding space for you, and I am listening without any judgment.",
            'embarrassment': "Oh, those moments where we want the ground to swallow us are universally human, even if they feel deeply personal in the moment. The fact that you're reflecting on it shows great self-awareness. Try to offer yourself the same grace you would give a friend in this situation.",
            'remorse': "It sounds like you are carrying a heavy weight of regret, and that shows how deeply you care about doing the right thing. Be careful not to let self-criticism overshadow your capacity to grow. What is one small, constructive action you can take today to make amends or learn from this?",
            'confusion': "It is completely normal to feel lost when things aren't making sense. That mental fog can be exhausting. Let's untangle this together, one piece at a time. What is the very first thing that feels unclear to you?"
        }
        return templates.get(emotion, f"I hear that you're feeling {emotion}, and that is completely valid. Let's explore this together. What do you need most right now?")

    # =========================================================================
    # MIXED STATE: Acknowledge the negative, but highlight the hidden positive
    # =========================================================================
    def _generate_mixed_reframe(self, negative_emotion: str, positive_emotion: str, text: str) -> str:
        return (
            f"It sounds like you're holding a really complex mix of feelings right now. "
            f"It is completely understandable to feel {negative_emotion} given the situation, "
            f"but I also notice a genuine sense of {positive_emotion} in what you're sharing. "
            f"Big, meaningful moments rarely come with just one emotion. "
            f"Both of these feelings are valid. Which of these two feels like it needs your attention the most right now?"
        )

    # =========================================================================
    # POSITIVE REINFORCEMENT: Amplify the good
    # =========================================================================
    def _generate_positive_response(self, emotion: str, text: str) -> str:
        templates = {
            'joy': "That is genuinely wonderful to hear! Your joy is contagious. It sounds like something really beautiful is happening in your life. Tell me more about what's making you feel this way!",
            'excitement': "That excitement is so palpable! New beginnings and milestones carry such an electric mix of anticipation and possibility. Tell me everything—what's happening?",
            'gratitude': "Holding onto that kind of gratitude is a beautiful thing. It sounds like someone or something made a real, positive difference for you. What are you feeling most thankful for right now?",
            'pride': "You absolutely should feel proud. What you've accomplished clearly took real effort, courage, and dedication. That feeling is hard-earned and completely deserved. How are you going to celebrate this?",
            'relief': "Oh, that sense of relief when a heavy weight finally lifts—there's nothing quite like it. It sounds like you've been carrying something stressful for a while. How does it feel to finally have this behind you?",
            'admiration': "It's wonderful when someone or something genuinely moves you like that. That kind of admiration says a lot about your own values and what you appreciate in the world. What stood out to you the most?",
            'optimism': "That sense of hope is a powerful thing. It sounds like you're genuinely looking forward to what's ahead, and that mindset will serve you well. What are you most excited to see unfold?"
        }
        return templates.get(emotion, f"That's wonderful to hear! It sounds like you're experiencing some real {emotion}. Tell me more about what's making you feel this way!")

    # =========================================================================
    # UNCERTAIN / LOW CONFIDENCE
    # =========================================================================
    def _generate_uncertain_response(self, text: str) -> str:
        return "I want to make sure I'm truly understanding how you're feeling, because what you're going through clearly matters. It sounds like there might be a lot going on beneath the surface. Can you tell me a bit more about what's been happening, and how it's been affecting you?"