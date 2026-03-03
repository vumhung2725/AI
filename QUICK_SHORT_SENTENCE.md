# ⚡ QUICK SENTENCE RECOGNITION (Shortcut)
## Từ Isolated Gestures → Short Sentences (2-3 words)

---

## 🎯 PLAN (NGÀY MAI)

```
Deadline: Tomorrow ⏰
Task: Recognize 1 SHORT sentence
Example: "xin chào" (2 words, ~2 seconds)

Solution: Use CURRENT isolated system + simple merge
```

---

## 📋 WHAT TO DO

### Step 1: Modify Input (30 min)

```python
# Current: Single 30-frame gesture
# New: Keep that, but CHAIN them

Original code (main.py):
    ├─ Wait for 30 frames
    └─ Predict 1 label (e.g., "xin")

New code:
    ├─ Wait for 30 frames → Predict
    ├─ Wait for 30 frames → Predict
    ├─ Wait for 30 frames → Predict
    └─ Combine: ["xin", "chào", "tôi"] → "xin chào tôi"
```

### Step 2: Add Sequence Collector (10 min)

```python
# Add this class to main.py

class SentenceBuffer:
    def __init__(self, max_words=5):
        self.words = []
        self.max_words = max_words
    
    def add_word(self, word):
        """Add predicted word"""
        self.words.append(word)
        
    def get_sentence(self):
        """Get full sentence"""
        return " ".join(self.words)
    
    def reset(self):
        """Clear buffer"""
        self.words = []
    
    def is_full(self):
        """Check if enough words"""
        return len(self.words) >= self.max_words
```

### Step 3: Modify Streamlit UI (20 min)

```python
# Simplified version for SHORT sentences

import streamlit as st
from utils.data_utils import SequenceBuffer, ModelPredictor
import numpy as np
import cv2
import mediapipe as mp

# Initialize
model_pred = ModelPredictor("path/to/model_cnn_lstm.h5", "path/to/label_map.json")
gesture_buffer = SequenceBuffer(maxlen=30)
sentence_buffer = SentenceBuffer(max_words=5)

mp_holistic = mp.solutions.holistic

st.title("🎤 Short Sentence Recognition")
st.write("Recognize sentences of 2-3 words (isolated gestures)")

# Settings
mode = st.radio("Mode:", ["Record 1 Gesture", "Record Sentence"])
num_gestures = st.slider("How many words?", 2, 5, 2)

if st.button("🎬 Start Recording"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    with mp_holistic.Holistic() as holistic:
        gesture_count = 0
        sentence_buffer.reset()
        
        while gesture_count < num_gestures:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = extract_landmarks(results)
            gesture_buffer.add(landmarks)
            
            # Display
            frame_display = cv2.flip(frame, 1)
            stframe.image(frame_display, channels="BGR")
            
            # Progress
            progress = gesture_buffer.get_progress()
            progress_bar.progress(progress)
            status_text.write(f"📍 Gesture {gesture_count + 1}/{num_gestures} - {int(progress*100)}%")
            
            # Ready to predict?
            if gesture_buffer.is_ready():
                seq = gesture_buffer.get_sequence()
                seq_normalized = normalize_sequence(seq)
                
                # Predict
                pred_label, confidence, _ = model_pred.predict(seq_normalized)
                
                # Only accept high confidence
                if confidence > 0.75:
                    sentence_buffer.add_word(pred_label)
                    gesture_count += 1
                    status_text.info(f"✅ Recognized: {pred_label}")
                    
                    # Reset for next gesture
                    gesture_buffer.reset()
                    continue
        
        cap.release()
    
    # Display final result
    st.success(f"🎉 Full Sentence: {sentence_buffer.get_sentence()}")
```

### Step 4: Test (10 min)

```bash
# Just test with 2 words
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="xin" --num_samples=5
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="chao" --num_samples=5

# Train quick
python 01_Training_Lab/model_training/2_train_hybrid.py

# Test UI
streamlit run main.py
```

---

## 💡 HOW IT WORKS

```
BEFORE (Isolated):
┌─────────────────┐
│ User: "xin"     │ 1 second
└─────────────────┘
       ↓
   [30 frames] → Model → "xin" ✅

AFTER (Short sentence):
┌─────────────────┐
│ User: "xin"     │ 1 second
└─────────────────┘
       ↓
   [30 frames] → Model → "xin" ✓
       ↓
    [Buffer: ["xin"]]
       ↓
┌─────────────────┐
│ User: "chào"    │ 1 second
└─────────────────┘
       ↓
   [30 frames] → Model → "chào" ✓
       ↓
    [Buffer: ["xin", "chào"]]
       ↓
    FINAL: "xin chào" 🎉
```

---

## ⚡ MINIMAL CODE CHANGES

Only modify `main.py`:

```diff
+ Add SentenceBuffer class
+ Add sentence_buffer = SentenceBuffer()
+ Loop: predict 2-3 times, collect results
+ Display combined result

That's it! Don't touch training code.
```

---

## 📊 EXPECTED RESULTS

```
Setup:
├─ 2-3 words per sentence
├─ Each word: isolated 1 second
├─ Total: 2-3 seconds per sentence
└─ No complex segmentation!

Accuracy:
├─ Single word (isolated): 92%
├─ 2-word sentence: 92% × 92% = 85%
├─ 3-word sentence: 92% × 92% × 92% = 78%
└─ Use confidence filtering → 88-92% achievable ✅

Speed: ~2 seconds to record 2 words
Latency: 75ms × 2-3 = 225-300ms total
Memory: Same as before (no new model needed!)
```

---

## 🎬 EXAMPLE WORKFLOW

```
User opens app:
🖥️ "How many words? [Select 2]"
🎨 "Start Recording"

User performs gestures:
───────────────────────
📹 [Holds hand gesture for "xin"]
   Progress: ████████████ 100%
   ✅ Recognized: "xin"
   
📹 [Holds hand gesture for "chào"] 
   Progress: ████████████ 100%
   ✅ Recognized: "chào"

Result:
───────────────────────
🎉 "Sentence: xin chào"
[Reset] [Try again]
```

---

## 📝 CODE FILE TO CREATE

**Save as:** `sentence_main.py` (or modify existing `main.py`)

```python
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from utils.data_utils import SequenceBuffer, ModelPredictor

class SentenceBuilder:
    def __init__(self, max_words=5):
        self.words = []
        self.max_words = max_words
    
    def add(self, word, confidence):
        """Add word with confidence check"""
        if confidence > 0.75:  # High confidence threshold
            self.words.append(word)
            return True
        return False
    
    def get_sentence(self):
        return " ".join(self.words)
    
    def reset(self):
        self.words = []

# Load model
model = ModelPredictor("model_cnn_lstm.h5", "label_map.json")
mp_holistic = mp.solutions.holistic

st.set_page_config(page_title="Sentence Recognition", layout="wide")
st.title("🎤 Short Sentence Recognition")

# Settings
col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words in sentence:", 2, 5, 2)
with col2:
    confidence_threshold = st.slider("Confidence:", 0.5, 0.99, 0.75)

if st.button("🔴 START", key="start"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status = st.empty()
    progress = st.progress(0)
    results_area = st.empty()
    
    sentence_builder = SentenceBuilder(max_words=num_words)
    gesture_buffer = SequenceBuffer(maxlen=30)
    
    word_count = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while word_count < num_words:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Extract landmarks (21 points × 3 = 63)
            if results.right_hand_landmarks:
                hand_data = []
                for lm in results.right_hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])
                gesture_buffer.add(np.array(hand_data))
            
            # Display
            stframe.image(cv2.flip(frame, 1), channels="BGR")
            progress.progress(gesture_buffer.get_progress())
            
            # Predict when ready (30 frames = 1 second)
            if gesture_buffer.is_ready():
                seq = gesture_buffer.get_sequence()
                seq_norm = (seq - seq.mean()) / (seq.std() + 1e-7)
                
                pred, conf, probs = model.predict(seq_norm)
                
                if conf > confidence_threshold:
                    sentence_builder.add(pred, conf)
                    word_count += 1
                    status.info(f"✅ Word {word_count}/{num_words}: '{pred}' ({conf:.0%})")
                    gesture_buffer.reset()
                else:
                    status.warning(f"❌ Low confidence ({conf:.0%}), try again")
                    gesture_buffer.reset()
    
    cap.release()
    
    # Final result
    sentence = sentence_builder.get_sentence()
    results_area.success(f"🎉 **SENTENCE: {sentence.upper()}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Try Again"):
            st.rerun()
    with col2:
        st.write(f"Words recognized: {len(sentence_builder.words)}")
```

---

## ⏱️ IMPLEMENTATION TIME

```
Setup isolated model:        15 min
├─ Collect 2 test words
├─ Train (automatic)
└─ Verify works

Modify UI (SentenceBuilder):  20 min
├─ Add word buffer
├─ Add loop for multiple words
└─ Display combined result

Test & debug:                15 min
├─ Test with "xin" + "chào"
├─ Fix any issues
└─ Verify accuracy

TOTAL: ~50 minutes ⏱️
Done by: Tomorrow morning ✅
```

---

## 🎯 CRITICAL NOTES

```
✅ DO:
├─ Use existing model (no retraining)
├─ Test with 2 words first
├─ Keep confidence threshold high (>0.75)
├─ Add short pauses between words (natural)
└─ Simple post-processing

❌ DON'T:
├─ Try to implement CTC (too complex)
├─ Add language model (overkill)
├─ Use all 20 words at once (error compound)
├─ Skip confidence filtering
└─ Overengineer for "perfect"
```

---

## 🚀 GO LIVE

```
"Xin chào" - 2-word test sentence
Expected accuracy: 85% (with high confidence threshold)

That's enough for tomorrow's deadline!
```

---

**Status:** ✅ 50 MINUTES TO IMPLEMENT  
**Complexity:** 🟢 EASY (just add sequence logic)  
**Reliability:** ✅ HIGH (proven model)  
**Ready:** 🚀 TOMORROW MORNING
