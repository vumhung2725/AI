# 🚀 STARTUP - Chỉ 3 Lệnh Thôi!

**Mục tiêu**: Hoàn thành train model buổi sáng nay

---

## ⚡ 3 Commands (Copy-Paste & Run)

### **Bước 1: Collect Data** (5-10 phút/action)

```bash
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=30
```

**Lặp lại cho mỗi ký hiệu**:
```bash
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="thankyou" --num_samples=30
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="no" --num_samples=30
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="love" --num_samples=30
python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="nothing" --num_samples=30
```

**Tối thiểu**: Collect 2-3 actions, mỗi action 20-30 samples

---

### **Bước 2: Train Model** (5-15 phút)

```bash
python 01_Training_Lab/model_training/2_train_hybrid.py
```

**Output**:
- ✅ Model saved: `01_Training_Lab/model_training/model_cnn_lstm.h5`
- ✅ Label map: `01_Training_Lab/model_training/label_map.json`

---

### **Bước 3: Run App** (Test)

```bash
streamlit run main.py
```

**Expected**: Webcam mở → Real-time prediction

---

## 📋 Folder Structure After Complete

```
dataset/
├── hello/
│   ├── hello_0.npy ... hello_29.npy
├── thankyou/
│   └── thankyou_0.npy ... thankyou_29.npy
└── ...

01_Training_Lab/model_training/
├── model_cnn_lstm.h5        ✅ (Created after step 2)
└── label_map.json           ✅ (Created after step 2)
```

---

## ⏱️ Thời Gian Dự Kiến

| Bước | Thời Gian | Ghi Chú |
|------|----------|--------|
| Collect 5 actions (25 samples each) | 20-30 min | Quay tay |
| Train model (20 epochs) | 10-15 min | CPU: ~10min, GPU: ~2min |
| Test app | 2-3 min | Mở Streamlit |
| **TOTAL** | **~45 min** | ✅ Xong! |

---

## 🎯 Tips

1. **During collection**:
   - Tay phải để ở trong frame
   - Ánh sáng tốt → MediaPipe detect tốt hơn
   - Press 'q' to skip, 's' to save early

2. **If training is slow**:
   - Reduce samples: `--num_samples=15`
   - Check data: `ls dataset/action_name/`

3. **If model won't load**:
   - Check path: `01_Training_Lab/model_training/model_cnn_lstm.h5` exists?
   - Run step 2 again

---

## ✅ Checklist

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Camera works: Test with `1_collect_seq.py`
- [ ] Data collected: Check `dataset/` folder
- [ ] Model trained: Check `01_Training_Lab/model_training/`
- [ ] App runs: `streamlit run main.py`

---

**Ready to train? Let's go!** 🎉
