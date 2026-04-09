# Quick Frontend Test Guide

## Issue Found & Fixed ✅

**Problem:** Missing `client/.env` file
**Solution:** Created `.env` with API URL configuration

---

## Test Steps (Do This Now!)

### 1. Restart Frontend Server

```bash
# Stop current server (Ctrl+C)
cd client
npm run dev
```

### 2. Open Browser

Navigate to: `http://localhost:5173/chat`

### 3. Send a Test Message

Type: "Hello, I feel stressed about exams"

### 4. What Should Happen

✅ **User message appears immediately** (blue bubble on right)
✅ **Loading indicator shows** (3 bouncing dots)
✅ **AI response appears** (gray bubble on left) after 2-5 seconds
✅ **Response is contextual** (mentions stress/exams)

---

## Verify It's Working

### Check 1: Browser DevTools

1. Press F12
2. Go to "Network" tab
3. Send a message
4. Look for: `POST /api/v1/chat/message`
5. Status should be: `200 OK`

### Check 2: Backend Logs

In your backend terminal, you should see:
```
[DEBUG] Starting chat processing for user: guest
[DEBUG] Detecting emotion...
[DEBUG] Emotion detected: stress
[DEBUG] Generating response...
[DEBUG] Response generated: I hear that exams...
[DEBUG] Message saved successfully
```

### Check 3: Response Content

The AI should respond with something like:
```
"I hear that exams are weighing on you. That tension makes sense. 
Want me to suggest a quick breathing exercise or help break down 
your study plan?"
```

NOT generic mock responses like:
```
"I hear you, and I appreciate you sharing that with me..."
```

---

## If It Still Doesn't Work

### Problem: CORS Error

**Error in console:**
```
Access to fetch... has been blocked by CORS policy
```

**Fix:** Check `server/app/core/config.py`:
```python
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
```

### Problem: Connection Refused

**Error:**
```
Failed to fetch
net::ERR_CONNECTION_REFUSED
```

**Fix:** Make sure backend is running:
```bash
cd server
python -m uvicorn app.main:app --reload
```

### Problem: 500 Internal Server Error

**Check backend logs** for the actual error, then:
```bash
# Check error log
cat server/chat_error.log
```

### Problem: Response Not Appearing

**Debug steps:**
1. Check browser console for JavaScript errors
2. Check Network tab - is response received?
3. Check response structure matches expected format
4. Try refreshing the page

---

## Expected vs Actual

### ✅ Expected Behavior:
1. Type message → appears immediately
2. Loading indicator → shows for 2-5 seconds
3. AI response → appears with contextual content
4. Backend logs → show processing steps
5. Network tab → shows successful API call

### ❌ If You See This (Mock Data):
1. Response appears after exactly 1.5 seconds
2. Generic responses not related to your message
3. No backend logs
4. No network requests in DevTools

---

## Quick Diagnostic

Run this in your browser console (F12 → Console):

```javascript
// Test API connection
fetch('http://localhost:8000/api/v1/chat/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ content: 'test' })
})
.then(r => r.json())
.then(d => console.log('✅ API Working:', d))
.catch(e => console.error('❌ API Error:', e))
```

**Expected output:**
```javascript
✅ API Working: {
  id: "...",
  conversation_id: "...",
  role: "assistant",
  content: "I'm here to listen...",
  emotion_data: {...},
  timestamp: "..."
}
```

---

## Files Changed

✅ Created: `client/.env`
```
VITE_API_URL=http://localhost:8000/api/v1
```

---

## Summary

1. ✅ Backend is working correctly
2. ✅ Frontend code is correct (Chat.tsx)
3. ✅ API integration is correct
4. ✅ Created missing .env file
5. 🔄 **Restart frontend to apply changes**

After restarting, the chat should work perfectly!

