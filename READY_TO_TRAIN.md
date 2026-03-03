# ✅ READY TO TRAIN - SÃ NAY XONG!

**Status**: ✨ Hoàn toàn setup → Chỉ việc train thôi

---

## 📋 Setup Checklist

- [x] ✅ MediaPipe added to `requirements.txt`
- [x] ✅ Collection script ready: `1_collect_seq.py`
- [x] ✅ Training script ready: `2_train_hybrid.py`
- [x] ✅ Export script ready: `3_convert_tfjs.py`
- [x] ✅ Streamlit app ready: `main.py`
- [x] ✅ Dependencies: See `requirements.txt`

---

## 🎯 Your Mission Today (Sáng nay)

### **Scenario 1: Quick Demo (30 minutes)**

1. Collect 2 actions only:
   ```bash
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=15
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="thankyou" --num_samples=15
   ```

2. Train:
   ```bash
   python 01_Training_Lab/model_training/2_train_hybrid.py
   ```

3. Test:
   ```bash
   streamlit run main.py
   ```

**Result**: ✅ Live demo working in 30-40 minutes

---

### **Scenario 2: Full Training (45 minutes)**

1. Collect 5 actions:
   ```bash
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=30
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="thankyou" --num_samples=30
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="no" --num_samples=30
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="love" --num_samples=30
   python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="nothing" --num_samples=30
   ```

2. Train:
   ```bash
   python 01_Training_Lab/model_training/2_train_hybrid.py
   ```

3. Test:
   ```bash
   streamlit run main.py
   ```

**Result**: ✅ Full system trained & working

---

## 🔧 Pre-Flight Checks

### Check 1: Dependencies
```bash
pip install -r requirements.txt
```

Should install:
- ✅ opencv-contrib-python
- ✅ tensorflow
- ✅ mediapipe
- ✅ streamlit
- ✅ numpy, scipy, sklearn, pandas, pillow

### Check 2: Verify Setup
```bash
python 01_Training_Lab/verify_setup.py
```

Expected output:
```
✅ cv2 (opencv)
✅ mediapipe
✅ numpy
✅ tensorflow
✅ streamlit
✅ sklearn
```

### Check 3: Camera Works
```bash
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="test" --num_samples=1
```

Expected:
- Camera opens
- Countdown: 3, 2, 1
- Hand landmarks visible on screen
- Saves 1 file to `dataset/test/`

---

## 📊 Expected Files After Training

```
dataset/
├── hello/ (30 files)
├── thankyou/ (30 files)
├── no/ (30 files)
├── love/ (30 files)
└── nothing/ (30 files)
Total: 150 samples

01_Training_Lab/model_training/
├── model_cnn_lstm.h5          ← Your trained model
├── label_map.json              ← Class labels
└── training_history.npy        ← Training logs
```

---

## 🚀 Command Shortcuts

**Copy-paste these** if running on PowerShell:

```powershell
# Activate venv
& .\.venv\Scripts\Activate.ps1

# Collect hello
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=30

# Train
python 01_Training_Lab/model_training/2_train_hybrid.py

# Test
streamlit run main.py
```

---

## ⏱️ Timeline

| Phase | Time | Task |
|-------|------|------|
| Setup (now) | 5 min | Install deps, verify |
| Collect (x5) | 25-30 min | Record 5 actions |
| Train | 10-15 min | CNN+LSTM training |
| Test | 2-3 min | Run Streamlit |
| **TOTAL** | **~45-50 min** | ✅ DONE |

---

## 🎬 What Happens When Training Starts?

```
Loading sequence data...
   Loading hello...
   ✅ Loaded 30 samples
   Loading thankyou...
   ✅ Loaded 30 samples
   ... (repeat for each action)

📊 Dataset Summary:
   Total samples: 150
   Shape: (150, 30, 63)
   Classes: {'0': 'hello', '1': 'thankyou', ...}

📐 Data split:
   Training: (120, 30, 63)
   Testing: (30, 30, 63)

🏗️ Building CNN+LSTM model...

📋 Model Summary:
   Layer 1: Conv1D(64, kernel=3) → Output: (batch, 28, 64)
   Layer 2: LSTM(128) → Output: (batch, 128)
   Layer 3: Dense(64) → Output: (batch, 64)
   Layer 4: Dense(5, softmax) → Output: (batch, 5)

🚀 Training model for 20 epochs...
Epoch 1/20: loss=2.1443, accuracy=0.2500, val_loss=2.0234, val_accuracy=0.3333
Epoch 2/20: loss=1.8932, accuracy=0.4375, val_loss=1.7234, val_accuracy=0.5000
...
Epoch 20/20: loss=0.1245, accuracy=0.9688, val_loss=0.2180, val_accuracy=0.8889

📊 Evaluating on test set...
   Test Loss: 0.2180
   Test Accuracy: 88.89%

✅ Model saved to 01_Training_Lab/model_training/model_cnn_lstm.h5
✅ Label map saved to 01_Training_Lab/model_training/label_map.json
✅ Training history saved to 01_Training_Lab/model_training/training_history.npy

🎯 Final Results:
   Training Accuracy: 96.88%
   Validation Accuracy: 88.89%
   Test Accuracy: 88.89%
```

Then Streamlit starts:
```
streamlit run main.py

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open http://localhost:8501 → See live predictions!

---

## ❌ If Something Goes Wrong

### Issue: "Module not found: mediapipe"
```bash
pip install mediapipe
```

### Issue: "No model file found"
- ❌ You haven't run step 2 (training) yet
- ✅ Run: `python 01_Training_Lab/model_training/2_train_hybrid.py`

### Issue: "Camera not working"
- ✅ Check: 1. Permission, 2. Another app using camera, 3. USB camera connected?
- ✅ Try: Restart computer

### Issue: "Training is very slow"
- ✅ Reduce samples: `--num_samples=10` instead of 30
- ✅ Use GPU if available (faster)
- ✅ Reduce epochs in 2_train_hybrid.py: change `EPOCHS = 20` → `EPOCHS = 5`

---

## 📚 More Info

- **Data collection**: See `01_Training_Lab/data_collection/1_collect_seq.py`
- **Training details**: See `01_Training_Lab/model_training/2_train_hybrid.py`
- **Architecture**: See `DOCUMENTATION.md`
- **Troubleshooting**: See `README.md`

---

## ✨ You're All Set!

**Ready to train?** 🚀

Copypaste these 3 commands:

```bash
# 1. Collect data (repeat for each action)
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=30

# 2. Train model
python 01_Training_Lab/model_training/2_train_hybrid.py

# 3. Run app
streamlit run main.py
```

**Let's go!** 💪
