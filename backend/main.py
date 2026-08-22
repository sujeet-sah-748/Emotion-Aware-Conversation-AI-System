import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
# FASTAPI APP SETUP
# =============================================================================
app = FastAPI(
    title="Emotion-Aware Chatbot API",
    description="Backend API for multi-label emotion classification using LoRA.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
) -> tuple[list, bool, list]:
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

# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/")
def read_root():
    return {"message": "Emotion-Aware Chatbot Backend is running!"}

@app.post("/predict", response_model=EmotionResponse)
def analyze_emotion(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        emotions, used_fallback, all_scores = predict_emotions(
            request.text, request.threshold, request.top_k
        )
        return EmotionResponse(
            text=request.text,
            emotions=emotions,
            used_fallback=used_fallback,
            all_scores=all_scores if request.include_all_scores else None,
            device=str(device),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
