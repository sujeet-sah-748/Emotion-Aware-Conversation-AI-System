"""
EmotionChat Backend — FastAPI server for multi-label emotion classification
and psychologically-grounded response composition.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from response_generator import compose_response


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ADAPTER_DIR = "./final_adapter"   # Path to your LoRA adapter folder
MAX_LENGTH = 128
SIGMOID_THRESHOLD = 0.5

# The 28 GoEmotions labels in their canonical order.
# Index 14 = "fear", index 13 = "excitement", etc.
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
# FASTAPI APP SETUP
# =============================================================================
app = FastAPI(
    title="EmotionChat API",
    description=(
        "Backend API for multi-label emotion classification using a LoRA-fine-tuned "
        "DistilRoBERTa model, with a psychologically-grounded response composer."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODEL LOADING (Runs once at startup)
# =============================================================================
print("🔄 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=28,
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded successfully on {device}!")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class PredictRequest(BaseModel):
    text: str
    threshold: float = SIGMOID_THRESHOLD
    top_k: int = None


class PredictResponse(BaseModel):
    text: str
    emotions: List[Dict[str, Any]]
    device: str


class ChatRequest(BaseModel):
    text: str
    threshold: float = 0.2   # Lower threshold to catch subtle co-occurring emotions
    top_k: int = 5


class ChatResponse(BaseModel):
    text: str
    emotions: List[Dict[str, Any]]
    bot_response: str
    is_crisis_override: bool
    validated_emotions: List[str]
    bridged_emotion: Optional[str]
    device: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def predict_emotions(text: str, threshold: float, top_k: int = None):
    """
    Run the LoRA-fine-tuned classifier on a single text input.
    Returns a list of {"label": str, "score": float} dicts, sorted by score.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    ).to(device)

    logits = model(**enc).logits.squeeze(0).cpu().numpy()
    probs = sigmoid(logits)

    results = sorted(
        [{"label": id2label[i], "score": float(p)} for i, p in enumerate(probs)],
        key=lambda x: -x["score"]
    )

    if top_k:
        return results[:top_k]

    return [r for r in results if r["score"] >= threshold] or [results[0]]


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/")
def read_root():
    return {
        "message": "EmotionChat API is running!",
        "endpoints": {
            "/predict": "POST — returns raw emotion scores",
            "/chat": "POST — returns emotion scores + composed bot response",
            "/health": "GET — health check",
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictResponse)
def analyze_emotion(request: PredictRequest):
    """
    Returns raw emotion classification scores for the given text.
    Used by external clients and the frontend's emotion badge UI.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        emotions = predict_emotions(request.text, request.threshold, request.top_k)
        return PredictResponse(
            text=request.text,
            emotions=emotions,
            device=str(device)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_with_emotion(request: ChatRequest):
    """
    Returns emotion scores AND a psychologically-grounded bot response
    in a single round-trip, minimizing latency.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # 1. Get raw emotion predictions from the LoRA model
        emotions = predict_emotions(request.text, request.threshold, request.top_k)

        # 2. Compose the supportive response using the response generator
        composed = compose_response(request.text, emotions)

        return ChatResponse(
            text=request.text,
            emotions=emotions,
            bot_response=composed.text,
            is_crisis_override=composed.is_crisis_override,
            validated_emotions=composed.validated_emotions,
            bridged_emotion=composed.bridged_emotion,
            device=str(device)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))