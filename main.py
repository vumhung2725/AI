"""
CNN+LSTM Sign Language Recognition - Streamlit App
Real-time sequence-based action recognition
"""

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import sys
import mediapipe as mp
import time

# Add training_lab parent to path to access 01_Training_Lab as a module
# Note: Directory starts with number, so we add parent and use path manipulation 
training_lab_path = str(Path(__file__).parent / "01_Training_Lab")
if training_lab_path not in sys.path:
    sys.path.insert(0, training_lab_path)

# Now import from the utils subfolder inside 01_Training_Lab 
from utils.data_utils import SequenceBuffer, ModelPredictor

# MediaPipe 0.10.x uses new API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand detection using HSV color
def detect_hand_contours(frame):
    """Detect hand contours using HSV color range"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([5, 40, 40])
    upper_skin = np.array([15, 255, 255])
    lower_skin2 = np.array([170, 40, 40])
    upper_skin2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def extract_hand_points(contours, frame_shape):
    """Extract 21-point hand landmarks from contours"""
    h, w = frame_shape[:2]
    landmarks = []
    
    if contours:
        # Get largest contour (assuming it's the hand)
        cnt = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(cnt) > 500:  # Minimum area threshold
            # Fit ellipse and get key points
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            # Extract points from contour
            for i, point in enumerate(approx[:21]):  # Take first 21 points
                x, y = point[0]
                landmarks.append([x/w, y/h, 0.5])  # Normalize and add z-coordinate
            
            # Pad with zeros if less than 21 points
            while len(landmarks) < 21:
                landmarks.append([0.0, 0.0, 0.0])
        else:
            landmarks = [[0.0, 0.0, 0.0]] * 21
    else:
        landmarks = [[0.0, 0.0, 0.0]] * 21
    
    return landmarks

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# ===== SENTENCE BUILDER CLASS =====
class SentenceBuilder:
    """Build sentences from isolated gestures"""
    def __init__(self, max_words=5):
        self.words = []
        self.max_words = max_words
    
    def add(self, word, confidence):
        """Add word if confidence high enough"""
        if confidence > 0.70:
            self.words.append(word)
            return True
        return False
    
    def get_sentence(self):
        """Get full sentence"""
        return " ".join(self.words)
    
    def get_word_count(self):
        return len(self.words)
    
    def reset(self):
        """Clear for next sentence"""
        self.words = []
    
    def is_full(self):
        """Check if enough words"""
        return len(self.words) >= self.max_words
# ===== END SENTENCE BUILDER =====

# Page config
st.set_page_config(
    page_title="Sign Language Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .prediction-box {
            font-size: 48px;
            font-weight: bold;
            color: #00ff00;
            text-align: center;
            padding: 20px;
            border: 3px solid #00ff00;
            border-radius: 10px;
            background-color: #000000;
        }
        .confidence-bar {
            font-size: 20px;
            color: #ffffff;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_hand_landmarks(results):
    """Extract right hand landmarks (63 features: 21 points * 3)"""
    landmarks = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks = [0.0] * 63
    return np.array(landmarks)


def extract_hand_landmarks_new(results):
    """Extract hand landmarks from new MediaPipe API (63 features: 21 points * 3)"""
    landmarks = []
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])
            break  # Only first hand
    else:
        landmarks = [0.0] * 63
    return np.array(landmarks)


def main():
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # ===== MODE SELECTOR =====
    mode = st.sidebar.radio("Mode:", ["Single Gesture", "Short Sentence (2-3 words)"])
    if mode == "Short Sentence (2-3 words)":
        num_words = st.sidebar.slider("Words to record:", 2, 5, 2)
    else:
        num_words = 1
    # ===== END MODE SELECTOR =====
    
    confidence_threshold = st.sidebar.slider("MediaPipe Confidence", 0.3, 1.0, 0.7)
    model_path = "01_Training_Lab/model_training/model_cnn_lstm.h5"
    label_map_path = "01_Training_Lab/model_training/label_map.json"
    
    # Check if model exists
    if not Path(model_path).exists():
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please run: `python 01_Training_Lab/model_training/2_train_hybrid.py`")
        return
    
    # Load model
    st.sidebar.write("Loading model...")
    try:
        predictor = ModelPredictor(model_path, label_map_path)
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🖐️ Sign Language Recognition")
        # ===== CONDITIONAL TITLE =====
        if mode == "Single Gesture":
            st.write("**CNN+LSTM Model** - Real-time Action Recognition from Sequence of Frames")
        else:
            st.write(f"**Sentence Mode** - Record {num_words} gestures to form a sentence (each gesture is ~1 second)")
        # ===== END CONDITIONAL TITLE =====
        video_placeholder = st.empty()
    
    with col2:
        st.write("### 📊 Prediction")
        prediction_box = st.empty()
        confidence_display = st.empty()
        buffer_progress = st.empty()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize sequence buffer
    sequence_buffer = SequenceBuffer(sequence_length=30, num_features=63)
    
    # ===== SENTENCE BUILDER INIT =====
    sentence_builder = SentenceBuilder(max_words=num_words if mode == "Short Sentence (2-3 words)" else 5)
    sentence_words_container = st.empty()
    # ===== END SENTENCE BUILDER INIT =====
    
    # Prediction smoothing (avoid flickering)
    last_prediction = None
    prediction_confidence_threshold = 0.7
    
    frame_count = 0
    frame_timestamp_ms = 0
    
    # Initialize MediaPipe Hand Landmarker (new API)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            frame_count += 1
            frame_timestamp_ms += 33  # ~30 FPS
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (new API)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            # Extract landmarks and add to buffer
            landmarks = extract_hand_landmarks_new(results)
            sequence_buffer.add_frame(landmarks)
            
            # Draw hand landmarks (simplified)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    for lm in hand_landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Prediction
            prediction_text = "Collecting data..."
            confidence_text = ""
            
            if sequence_buffer.is_ready():
                sequence = sequence_buffer.get_sequence()
                action_name, confidence, probabilities = predictor.predict(sequence)
                
                # ===== PREDICTION LOGIC =====
                if mode == "Short Sentence (2-3 words)":
                    # Sentence mode: collect multiple words
                    if confidence > prediction_confidence_threshold:
                        if sentence_builder.add(action_name, confidence):
                            prediction_text = f"✅ Added: {action_name}"
                            confidence_text = f"Confidence: {confidence:.2%}"
                            
                            # Reset for next gesture
                            sequence_buffer.reset()
                            
                            # Check if sentence is complete
                            if sentence_builder.is_full():
                                with col1:
                                    st.success(f"🎉 **SENTENCE: {sentence_builder.get_sentence().upper()}**")
                                # Reset for next sentence
                                sentence_builder.reset()
                        else:
                            prediction_text = "Low confidence, try again"
                            confidence_text = ""
                    else:
                        prediction_text = "Uncertain"
                        confidence_text = f"Confidence too low: {confidence:.2%}"
                
                else:
                    # Single gesture mode (original behavior)
                    if confidence > prediction_confidence_threshold:
                        last_prediction = action_name
                        prediction_text = action_name
                        confidence_text = f"Confidence: {confidence:.2%}"
                    else:
                        prediction_text = "Uncertain"
                        confidence_text = f"Confidence too low: {confidence:.2%}"
                # ===== END PREDICTION LOGIC =====
            
            # Draw prediction on frame
            cv2.putText(frame, f"Prediction: {prediction_text}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Draw buffer progress
            progress = sequence_buffer.get_progress()
            bar_length = 30
            bar = "█" * int(progress * bar_length) + "░" * (bar_length - int(progress * bar_length))
            cv2.putText(frame, f"Buffer: [{bar}] {int(progress*100)}%",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Convert back to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update display
            with col1:
                video_placeholder.image(frame_rgb, use_column_width=True)
                # ===== DISPLAY WORD COUNT =====
                if mode == "Short Sentence (2-3 words)":
                    sentence_words_container.info(
                        f"📝 Words collected: {sentence_builder.get_word_count()}/{num_words}\n"
                        f"Current: {sentence_builder.get_sentence() if sentence_builder.words else 'None'}"
                    )
                # ===== END DISPLAY WORD COUNT =====
            
            with col2:
                prediction_box.markdown(f'<div class="prediction-box">{prediction_text}</div>', 
                                       unsafe_allow_html=True)
                confidence_display.markdown(f'<div class="confidence-bar">{confidence_text}</div>',
                                          unsafe_allow_html=True)
                buffer_progress.progress(progress)
    
    cap.release()


if __name__ == "__main__":
    main()
