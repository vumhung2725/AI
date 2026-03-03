# 🚀 CNN+LSTM Upgrade - Quick Start Guide

## 📋 New Directory Structure

```
01_Training_Lab/
├── data_collection/
│   └── 1_collect_seq.py           # Collect sequence data (30 frames per sample)
├── model_training/
│   ├── 2_train_hybrid.py          # Train CNN+LSTM model
│   ├── 3_convert_tfjs.py          # Export model to TFJS format
│   ├── model_cnn_lstm.h5          # Trained model (created after step 2)
│   └── label_map.json             # Class labels (created after step 2)
├── utils/
│   └── data_utils.py              # SequenceBuffer, ModelPredictor classes
└── dataset/                        # Your data goes here
    ├── action_hello/
    │   ├── action_hello_sample_0.npy
    │   ├── action_hello_sample_1.npy
    │   └── ...
    ├── action_no/
    └── ...

02_Web_Client/
├── js/                            # (For future: Vanilla JS web app)
└── assets/
    └── models/
        ├── tfjs_model/            # Exported TFJS model
        └── label_map.json         # Class labels

03_Docs_QC/
└── reports/                       # Documentation & test results
```

---

## 🎯 Step-by-Step Instructions

### **Step 1: Collect Sequence Data** (10-30 mins per action)

Run this command for each action you want to train:

```bash
python 01_Training_Lab/data_collection/1_collect_seq.py \
    --action_name="hello" \
    --num_samples=30 \
    --sequence_length=30 \
    --confidence=0.5
```

**Parameters:**
- `--action_name`: Name of the sign action (e.g., "hello", "thank_you", "yes", "no")
- `--num_samples`: Number of training samples (default: 30) - **Increase to 50-100 for better accuracy**
- `--sequence_length`: Length of each sequence in frames (default: 30)
- `--confidence`: MediaPipe detection confidence (0.3-1.0)

**What it does:**
- Opens your webcam
- After 3 seconds, starts recording
- Collects 30 consecutive frames of your hand position
- Saves as `.npy` file in `dataset/[action_name]/`

**Tips:**
- Perform the action smoothly and clearly
- Vary speed and angle for better model generalization
- Make sure hand is visible and well-lit

### **Example: Collect Multiple Actions**

```bash
# Hello gesture
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=50

# Thank you gesture
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="thank_you" --num_samples=50

# Yes gesture
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="yes" --num_samples=50

# No gesture
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="no" --num_samples=50
```

---

### **Step 2: Train CNN+LSTM Model** (5-15 mins depending on data)

After collecting data for at least 2-3 actions:

```bash
python 01_Training_Lab/model_training/2_train_hybrid.py
```

**What it does:**
- Loads all `.npy` files from `dataset/` folder
- Builds CNN+LSTM model:
  - **Conv1D Layer**: Extracts features from sequence
  - **LSTM Layer**: Learns temporal patterns
  - **Dense Layers**: Classification
- Splits data into training (80%) and testing (20%)
- Trains for 20 epochs
- Saves model to `01_Training_Lab/model_training/model_cnn_lstm.h5`
- Creates `label_map.json` with action names

**Expected output:**
```
📊 Dataset Summary:
   Total samples: 150
   Shape: (150, 30, 63)
   Classes: {'0': 'hello', '1': 'thank_you', '2': 'yes', '3': 'no'}

🚀 Training model for 20 epochs...
Epoch 1/20: loss=2.1443, accuracy=0.2500, val_loss=2.0234, val_accuracy=0.3333
Epoch 2/20: loss=1.8932, accuracy=0.4375, val_loss=1.7234, val_accuracy=0.5000
...
Epoch 20/20: loss=0.1245, accuracy=0.9688, val_loss=0.2180, val_accuracy=0.8889

📊 Evaluating on test set...
   Test Loss: 0.2180
   Test Accuracy: 88.89%
```

---

### **Step 3 (Optional): Export to TFJS Format** (1 min)

If you want to use the model in a **web browser** (JavaScript):

```bash
python 01_Training_Lab/model_training/3_convert_tfjs.py
```

**What it does:**
- Converts Keras `.h5` model → TFJS format
- Creates `model.json` + `.bin` files
- Copies to `02_Web_Client/assets/models/`

**Output:**
```
✅ Model converted and saved!
📁 Files created:
   - 02_Web_Client/assets/models/tfjs_model/model.json
   - 02_Web_Client/assets/models/tfjs_model/group1-shard1of1.bin
   - 02_Web_Client/assets/models/label_map.json
```

---

### **Step 4: Run Real-Time App** ✅

Use the new Streamlit app with CNN+LSTM model:

```bash
streamlit run main.py
```

**Features:**
- ✅ Real-time sequence collection (30 frames)
- ✅ Live CNN+LSTM prediction
- ✅ Confidence score display
- ✅ Buffer progress bar
- ✅ Hand pose visualization

---

## 📊 Model Architecture

```
Input: (Batch=1, Sequence_Length=30, Features=63)
        ↓
Conv1D Layer (64 filters, 3 kernel size)
Dropout (0.3)
        ↓
LSTM Layer (128 units)
Dropout (0.3)
        ↓
Dense Layer (64 units, ReLU)
Dropout (0.3)
        ↓
Dense Layer (N_classes, Softmax)
        ↓
Output: Probabilities for each action
```

---

## 🎯 Key Differences: SVM vs CNN+LSTM

| Aspect | SVM (Old) | CNN+LSTM (New) |
|--------|----------|----------------|
| Input | Single frame (1x63) | Sequence (30x63) |
| Time awareness | ❌ No | ✅ Yes |
| Movement detection | ❌ Static pose only | ✅ Dynamic action |
| Training time | ~1 sec | ~5-15 mins |
| Accuracy | ~70% | ~85-95% |

---

## 🔧 Troubleshooting

### **Error: "Model not found"**
```bash
# Make sure you ran Step 2 first
python 01_Training_Lab/model_training/2_train_hybrid.py
```

### **WebCam not opening**
```bash
# Check camera index (usually 0 for built-in)
python01_Training_Lab/data_collection/1_collect_seq.py --action_name="test"
```

### **Low accuracy (<70%)**
- Collect more data (200+ samples per action)
- Vary angles and distances
- Ensure good lighting
- Perform gestures more distinctly

### **Out of memory error**
- Reduce `--num_samples` for each collection
- Reduce model complexity in `2_train_hybrid.py`

---

## 📚 Next Steps (Future)

- [ ] Implement Vanilla JS web app (without Streamlit)
- [ ] Deploy TFJS model to production
- [ ] Create data augmentation pipeline
- [ ] Fine-tune model hyperparameters
- [ ] Multi-hand recognition
- [ ] Gesture combinations

---

## 💡 Tips for Best Results

1. **Data Quality > Quantity**: 50 good samples > 500 bad samples
2. **Consistency**: Use same lighting, distance, and background
3. **Variation**: Collect from different angles and speeds
4. **Testing**: Test on unseen data before deployment
5. **Iterative**: Start simple (2-3 actions), then expand

---

**Happy training! 🚀**
