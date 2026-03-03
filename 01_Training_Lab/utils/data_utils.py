"""
Data utilities for sequence-based pose estimation
"""

import numpy as np
import json
from pathlib import Path
from collections import deque


class SequenceBuffer:
    """Sliding window buffer for sequence collection"""
    
    def __init__(self, sequence_length=30, num_features=63):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.buffer = deque(maxlen=sequence_length)
    
    def add_frame(self, landmarks):
        """Add a frame (landmarks) to buffer"""
        if len(landmarks) == self.num_features:
            self.buffer.append(landmarks)
        else:
            # Pad with zeros if landmarks missing
            padded = np.zeros(self.num_features)
            padded[:len(landmarks)] = landmarks[:self.num_features]
            self.buffer.append(padded)
    
    def is_ready(self):
        """Check if buffer has enough frames"""
        return len(self.buffer) == self.sequence_length
    
    def get_sequence(self):
        """Get current sequence as numpy array"""
        if self.is_ready():
            return np.array(list(self.buffer))
        else:
            return None
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def get_progress(self):
        """Get buffer fill percentage"""
        return len(self.buffer) / self.sequence_length


class ModelPredictor:
    """Wrapper for TensorFlow model prediction"""
    
    def __init__(self, model_path, label_map_path):
        import tensorflow as tf
        
        self.model = tf.keras.models.load_model(model_path)
        
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            # Convert string keys to int
            self.label_map = {int(k): v for k, v in label_data.items()}
    
    def predict(self, sequence):
        """
        Predict action from sequence
        Args:
            sequence: numpy array of shape (30, 63)
        
        Returns:
            action_name: str
            confidence: float (0-1)
        """
        # Normalize
        X_mean = sequence.mean(axis=(0, 1), keepdims=True)
        X_std = sequence.std(axis=(0, 1), keepdims=True)
        X_normalized = (sequence - X_mean) / (X_std + 1e-8)
        
        # Add batch dimension
        X_batch = np.expand_dims(X_normalized, axis=0)
        
        # Predict
        probabilities = self.model.predict(X_batch, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        action_name = self.label_map.get(int(predicted_class), "Unknown")
        
        return action_name, float(confidence), probabilities


def load_label_map(label_map_path):
    """Load label map from JSON"""
    with open(label_map_path, 'r') as f:
        label_data = json.load(f)
        return {int(k): v for k, v in label_data.items()}


def normalize_sequence(sequence):
    """Normalize sequence using z-score"""
    mean = sequence.mean(axis=(0, 1), keepdims=True)
    std = sequence.std(axis=(0, 1), keepdims=True)
    return (sequence - mean) / (std + 1e-8)
