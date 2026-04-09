@echo off
echo Starting Emotion Chatbot Backend...
cd server
set PYTHONPATH=%CD%
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
