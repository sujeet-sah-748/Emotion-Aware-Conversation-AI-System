# Quick Fix: PyTorch Not Found Error

## The Error
```
NameError: name 'torch' is not defined
```

## What Happened
PyTorch (torch) is not installed in your Python environment, but it's required for the emotion detection model.

## Quick Fix (Choose One)

### Option 1: Install PyTorch CPU Version (Recommended for Quick Start)
```bash
cd server
pip install torch torchvision torchaudio
```

### Option 2: Install PyTorch with GPU Support (If you have NVIDIA GPU)
```bash
cd server
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 3: Reinstall All Dependencies
```bash
cd server
pip install -r requirements.txt
```

## Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

You should see output like:
```
PyTorch version: 2.1.2
```

## Check All Dependencies
I've created a diagnostic script for you:
```bash
cd server
python check_dependencies.py
```

This will show you which packages are installed and which are missing.

## Then Restart Your Server
```bash
cd server
python -m uvicorn app.main:app --reload
```

## Why This Happened
The `transformers` library (used for emotion detection) requires PyTorch as a backend. Even though it's listed in `requirements.txt`, it might not have been installed properly or you might be using a different Python environment.

## Still Having Issues?

### Check your Python environment:
```bash
# Which Python are you using?
python --version

# Where is it located?
where python  # On Windows
which python  # On Linux/Mac

# What packages are installed?
pip list | grep torch
```

### Make sure you're in the right virtual environment:
```bash
# If using venv
cd server
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Then install
pip install torch
```

