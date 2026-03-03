# Hướng dẫn Train Model Sign Language Recognition

## 📋 Yêu cầu trước khi train

- Tối thiểu: **2 actions**, mỗi action **10+ samples**
- Tốt nhất: **3-5 actions**, mỗi action **20-30 samples**

---

## Bước 1: Thu thập dữ liệu

### Thu thập action "NO" (20 samples):
```bash
cd c:/Users/GIGA/Sign-Language-Classification
.venv\Scripts\python.exe 01_Training_Lab/data_collection/1_collect_seq.py --action_name no --num_samples 20
```

**Cách thu thập:**
1. Đợi 3 giây đếm ngược
2. Thực hiện cử chỉ "NO" (lắc đầu) liên tục
3. Nhấn **'s'** để lưu mỗi khi hoàn thành 1 sample
4. Lặp lại 20 lần

### Thu thập action "THANKYOU" (20 samples):
```bash
.venv\Scripts\python.exe 01_Training_Lab/data_collection/1_collect_seq.py --action_name thankyou --num_samples 20
```

### Thu thập action "ILOVEYOU" (20 samples):
```bash
.venv\Scripts\python.exe 01_Training_Lab/data_collection/1_collect_seq.py --action_name iloveyou --num_samples 20
```

---

## Bước 2: Kiểm tra dữ liệu

Sau khi thu thập, kiểm tra thư mục `dataset/`:
```
dataset/
├── hello/          (đã có 1 sample)
├── no/            (20 samples)
├── thankyou/       (20 samples)
└── iloveyou/      (20 samples)
```

---

## Bước 3: Train Model

```bash
.venv\Scripts\python.exe 01_Training_Lab/model_training/2_train_hybrid.py
```

### Các thông số training:
- **Epochs:** 20
- **Batch Size:** 16
- **Validation Split:** 20%

### Kết quả mong đợi:
```
📊 Dataset Summary:
   Total samples: 61
   Classes: {0: 'hello', 1: 'no', 2: 'thankyou', 3: 'iloveyou'}

🎯 Final Results:
   Training Accuracy: ~95%
   Validation Accuracy: ~90%
   Test Accuracy: ~85%
```

---

## Bước 4: Chạy ứng dụng Streamlit

```bash
.venv\Scripts\streamlit.exe run main.py
```

---

## Xử lý lỗi thường gặp

### Lỗi: "No module named 'cv2'"
→ Sử dụng `.venv\Scripts\python.exe` thay vì `python`

### Lỗi: "n_samples=1, test_size=0.2"
→ Cần thu thập thêm dữ liệu (tối thiểu 10 samples)

### Lỗi: Import error trong main.py
→ Đã được fix: Sử dụng sys.path.insert

---

## Cấu trúc Model CNN+LSTM

```
Input: (30 frames, 63 features)
├── Conv1D(64 filters, kernel=3) → ReLU → Dropout(0.3)
├── LSTM(128 units) → Dropout(0.3)
├── Dense(64) → ReLU → Dropout(0.3)
└── Dense(num_classes) → Softmax
```

---

## Mẹo để có model tốt

1. ✅ Thu thập đa dạng ánh sáng
2. ✅ Giữ khoảng cách camera ổn định
3. ✅ Thực hiện cử chỉ rõ ràng, chậm rãi
4. ✅ Nhiều người thu thập → model tổng quát hơn
5. ❌ Tránh background phức tạp

