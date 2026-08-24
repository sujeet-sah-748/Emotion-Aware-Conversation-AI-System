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

@app.post("/chat", response_model=ChatResponse)
def chat_with_emotion(request: ChatRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        # 1. Get raw emotion predictions from your fine-tuned LoRA model
        emotions = predict_emotions(request.text, request.threshold, request.top_k)
        
        # 2. Compose the supportive response using your new logic
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
