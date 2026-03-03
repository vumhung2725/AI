# BAO CAO MO HINH - HE THONG NHAN DAN NGON NGU KY HIEU

---

## 1. TONG QUAN VE CAC MO HINH

### 1.1 Cac mo hinh su dung trong du an

| STT | Mo hinh | Chuc nang | File |
|-----|---------|-----------|------|
| 1 | MediaPipe Hand Landmarker | Trich xuat 21 diem landmark tu anh | Tu dong (MediaPipe) |
| 2 | CNN-LSTM | Mo hinh phan loai hanh dong | 2_train_hybrid.py |
| 3 | ModelPredictor | Du doan thoi gian thuc | data_utils.py |
| 4 | SequenceBuffer | Luu tru buffer 30 frames | data_utils.py |

---

## 2. MO HINH 1: MEDIAPIPE HAND LANDMARKER

### 2.1 Gioi thieu
MediaPipe Hand Landmarker la mo hinh duoc phat trien boi Google, su dung de trich xuat 21 diem landmark tren ban tay tu hinh anh hoac video.

### 2.2 Cach hoat dong

```
[Hinh anh] -> [MediaPipe Hand Landmarker] -> [21 diem landmark]
                                             
Dau vao: Hinh anh RGB (1920x1080 hoac thap hon)
Dau ra: 21 diem landmark, moi diem co 3 toa do (x, y, z)
```

### 2.3 Cau truc 21 diem landmark tren ban tay

```
           8           12          16          20
            |           |           |           |
            |           |           |           |
    4-------|----5------|----9------|----13-----|----17
            |           |           |           |
            |           |           |           |
            3-----------7-----------11----------15-----------19
            |           |           |           |
            |           |           |           |
    2-------|----6------|----10----|----14-----|----18
            |           |           |           |
            |           |           |           |
            1-----------0-----------5-----------9-----------13
                          |
                          |
                         0 (Wrist - Cot tay)
```

**Giai thich cac diem:**
- **0**: Wrist (Cot tay) - Diem neo
- **1-4**: Thumb (Ngon cai) - Cac diem ngon cai
- **5-8**: Index Finger (Ngon tro) - Cac diem ngon tro
- **9-12**: Middle Finger (Ngon giua) - Cac diem ngon giua
- **13-16**: Ring Finger (Ngon dan) - Cac diem ngon dan
- **17-20**: Pinky (Ngon ut) - Cac diem ngon ut

### 2.4 Thong so ky thuat

| Thong so | Gia tri |
|----------|---------|
| So landmark | 21 |
| Toa do/landmark | 3 (x, y, z) |
| Tong dac trung | 63 |
| Do chinh xac | 95.7% |
| Model size | ~7.8MB |

---

## 3. MO HINH 2: CNN-LSTM (MONG LAI)

### 3.1 Gioi thieu
Day la mo hinh chinh cua du an, la su ket hop giua:
- **CNN (Convolutional Neural Network)**: Trich xuat dac trung khong gian
- **LSTM (Long Short-Term Memory)**: Hoc quan he thoi gian

### 3.2 Cau truc mo hinh

```
Dau vao: (batch_size, 30, 63)
         │
         ▼
┌────────────────────────────────────────────┐
│  Conv1D Layer                               │
│  - Filters: 64                              │
│  - Kernel size: 3                          │
│  - Activation: ReLU                        │
│  Output: (batch_size, 28, 64)              │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Dropout (0.3)                             │
│  - Giam overfitting                         │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  LSTM Layer                                 │
│  - Units: 128                              │
│  - return_sequences: False                 │
│  Output: (batch_size, 128)                 │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Dropout (0.3)                             │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Dense Layer                                │
│  - Units: 64                               │
│  - Activation: ReLU                        │
│  Output: (batch_size, 64)                  │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Dropout (0.3)                             │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Output Layer                               │
│  - Units: so_lop (num_classes)             │
│  - Activation: Softmax                      │
│  Output: (batch_size, so_lop)              │
└────────────────────────────────────────────┘
```

### 3.3 Chung toi dung gi de xay dung mo hinh?

#### Layer 1: Conv1D
```python
layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
             input_shape=(30, 63))
```
**Muc dich:** Trich xuat cac dac trung cuc bo tu chuoi input 30 frames

**Cach hoat dong:**
- Quet qua 3 frames lien tiep
- Trich xuat 64 features khac nhau
- Giu lai thong tin vi tri

#### Layer 2: LSTM
```python
layers.LSTM(units=128, return_sequences=False)
```
**Muc dich:** Hoc cac mau thoi gian trong chuoi hanh dong

**Tai sao LSTM?'
- Cac hanh dong ngon ngu ky hieu dien ra trong mot khoang thoi gian
- Can giu lai thong tin tu cac frames truoc
- LSTM giai quyet van de vanishing gradient cua RNN thong thuong

#### Layer 3-4: Dense + Softmax
```python
layers.Dense(64, activation='relu')
layers.Dense(num_classes, activation='softmax')
```
**Muc dich:** Phan loai hanh dong dua tren dac trung da hoc

### 3.4 Cong thuc toan hoc

#### Conv1D:
$$y_i = \sigma(W \cdot x_{i:i+k} + b)$$

Trong do:
- $x_{i:i+k}$: Input window 3 frames
- $W$: Weight matrix (64 x 3 x 63)
- $\sigma$: Ham ReLU

#### LSTM:

**Forget Gate:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Cell State Update:**
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Output Gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State:**
$$h_t = o_t \cdot \tanh(C_t)$$

#### Softmax:
$$p(y_i | x) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### 3.5 Thong so training

| Thong so | Gia tri | Ly do |
|----------|---------|-------|
| Sequence Length | 30 frames | Du de bao gom mot hanh dong hoan chinh |
| Batch Size | 16 | Can bang giua toc do va bo nho |
| Epochs | 20 | Du de hoi tu ma khong qua giai |
| Learning Rate | 0.001 | Mac dinh cho Adam optimizer |
| Dropout | 0.3 | Giam overfitting |
| Validation Split | 20% | Du du lieu validation |

---

## 4. MO HINH 3: SEQUENCEBUFFER

### 4.1 Gioi thieu
SequenceBuffer la mot cau truc du lieu de luu tru sliding window 30 frames lien tiep.

### 4.2 Cau truc
```python
class SequenceBuffer:
    def __init__(self, sequence_length=30, num_features=63):
        self.sequence_length = 30
        self.num_features = 63
        self.buffer = deque(maxlen=30)
```

### 4.3 Cac phuong thuc

| Phuong thuc | Chuc nang |
|-------------|-----------|
| add_frame() | Them 1 frame vao buffer |
| is_ready() | Kiem tra buffer da du 30 frames chua |
| get_sequence() | Lay ra chuoi 30 frames |
| reset() | Xoa buffer |
| get_progress() | Lay phan tram buffer da du |

### 4.4 Sieu do hoat dong

```
Frame 1: [f1, 0, 0, ..., 0]        Progress: 3.3%
Frame 2: [f1, f2, 0, ..., 0]        Progress: 6.7%
...
Frame 30: [f1, f2, f3, ..., f30]    Progress: 100% -> is_ready() = True
Frame 31: [f2, f3, f4, ..., f31]    Slide window
```

---

## 5. MO HINH 4: MODELPREDICTOR

### 5.1 Gioi thieu
ModelPredictor la lop wrapper cho mo hinh Keras, cung cap ham predict() de du doan hanh dong tu chuoi input.

### 5.2 Cau truc
```python
class ModelPredictor:
    def __init__(self, model_path, label_map_path):
        self.model = tf.keras.models.load_model(model_path)
        self.label_map = {...}  # {0: 'hello', 1: 'no', ...}
    
    def predict(self, sequence):
        # Normalize
        X_normalized = (sequence - mean) / (std + 1e-8)
        
        # Predict
        probabilities = self.model.predict(X_batch)[0]
        
        # Get result
        predicted_class = np.argmax(probabilities)
        action_name = self.label_map[predicted_class]
        
        return action_name, confidence, probabilities
```

### 5.3 Quy trinh du doan

```
[30 frames x 63] -> [Normalize (z-score)] -> [Add batch dimension]
                                                    │
                                                    ▼
[Predicted Class] <- [ArgMax] <- [Softmax Output] <-
```

### 5.4 Qua trinh normalize

```python
X_mean = sequence.mean(axis=(0, 1), keepdims=True)
X_std = sequence.std(axis=(0, 1), keepdims=True)
X_normalized = (sequence - X_mean) / (X_std + 1e-8)
```

**Tai sao can normalize?**
- Giup mo hinh hoi tu nhanh hon
- Giam anh huong cua gia tri lon nho
- Tuy nhien: Moi sample deu duoc normalize rieng (khong su mean/std cua training data)

---

## 6. LUONG DU LIEU TRONG HE THONG

### 6.1 Tu thu thap den du doan

```
1. CAMERA CAPTURE
   Webcam -> OpenCV Frame (BGR)

2. MEDIAPIPE PROCESS
   Frame -> RGB -> HandLandmarker.detect() -> 21 landmarks (63 features)

3. BUFFER ACCUMULATION
   63 features -> SequenceBuffer.add_frame()
   
   Lap lai 30 lan cho den khi buffer day

4. MODEL PREDICTION
   30 frames -> Normalize -> CNN-LSTM -> Softmax -> Predicted Class

5. UI DISPLAY
   Predicted Class -> Streamlit Display
```

### 6.2 Thoi gian xu ly

| Buoc | Thoi gian (ms) |
|------|----------------|
| Camera capture | ~5ms |
| MediaPipe detect | ~10ms |
| Buffer + predict | ~5ms |
| **Tong** | **~20ms/frame** |
| **FPS** | **~50 FPS** |

---

## 7. CAC THU VIEN SU DUNG

| Thu vien | Phien ban | Chuc nang |
|----------|-----------|-----------|
| TensorFlow | 2.20 | Framework deep learning |
| Keras | 3.x | API xay dung mo hinh |
| MediaPipe | 0.10.32 | Trich xuat landmark |
| OpenCV | - | Xu ly anh va video |
| Streamlit | - | Giao dien web |
| NumPy | - | Xu ly mang |
| Scikit-learn | - | Tien xu ly du lieu |

---

## 8. KET LUAN

### 8.1 Tom tat mo hinh

1. **MediaPipe Hand Landmarker**: Trich xuat 21 diem landmark (63 features) tu moi frame
2. **CNN-LSTM**: Hoc dac trung khong gian (CNN) va thoi gian (LSTM) de phan loai hanh dong
3. **SequenceBuffer**: Luu tru buffer 30 frames de tao chuoi input
4. **ModelPredictor**: Wrapper cho viec du doan thoi gian thuc

### 8.2 Diem manh

- Trich xuat landmark chinh xac cao (95.7%)
- Mo hinh CNN-LSTM hieu qua cho du lieu sequence
- Co the chay real-time (~50 FPS)
- Giao dien Streamlit de su dung

### 8.2 Diem can cai thien

- Can them nhieu du lieu training
- Co the thu nghiem them LSTM layers
- Them regularization de giam overfitting

---

**File bao cao:** MODEL_REPORT.md
**Ngay cap nhat:** 2024

