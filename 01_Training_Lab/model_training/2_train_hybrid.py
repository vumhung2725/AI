"""
CNN + LSTM Hybrid Model for Sign Language Recognition
Input: Sequence of 30 frames, each with 63 coordinates (21 hand landmarks * 3)
Output: Action classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
from pathlib import Path


def load_sequence_data(dataset_path="dataset", sequence_length=30):
    """
    Load all .npy files from dataset folder
    Expected structure:
    dataset/
    ├── action_hello/
    │   ├── action_hello_sample_0.npy
    │   ├── action_hello_sample_1.npy
    │   └── ...
    ├── action_no/
    └── ...
    """
    X = []
    y = []
    label_map = {}
    label_idx = 0
    
    dataset_path = Path(dataset_path)
    
    # Iterate through action folders
    for action_folder in sorted(dataset_path.iterdir()):
        if not action_folder.is_dir():
            continue
        
        action_name = action_folder.name
        label_map[label_idx] = action_name
        
        print(f"Loading {action_name}...")
        
        # Load all .npy files in this action folder
        npy_files = list(action_folder.glob("*.npy"))
        for npy_file in npy_files:
            try:
                sequence_data = np.load(npy_file)
                
                # Validate shape
                if sequence_data.shape == (sequence_length, 63):
                    X.append(sequence_data)
                    y.append(label_idx)
                else:
                    print(f"⚠️  Skipped {npy_file.name} - Wrong shape {sequence_data.shape}")
            except Exception as e:
                print(f"❌ Error loading {npy_file.name}: {e}")
        
        print(f"   ✅ Loaded {len(npy_files)} samples")
        label_idx += 1
    
    X = np.array(X)  # Shape: (num_samples, 30, 63)
    y = np.array(y)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {label_map}")
    
    return X, y, label_map


def build_cnn_lstm_model(sequence_length=30, num_features=63, num_classes=5):
    """
    Build hybrid CNN+LSTM model
    
    Architecture:
    - Conv1D: Extract features from sequence
    - LSTM: Learn temporal dependencies
    - Dense: Classification
    """
    model = Sequential([
        # Input: (batch_size, 30, 63)
        
        # Conv1D layer: Extract local features from sequence
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                     input_shape=(sequence_length, num_features)),
        layers.Dropout(0.3),
        
        # LSTM: Learn temporal patterns
        layers.LSTM(units=128, return_sequences=False),
        layers.Dropout(0.3),
        
        # Dense layers for classification
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def main():
    # Create output directory
    model_dir = Path("01_Training_Lab/model_training")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 16
    EPOCHS = 20
    VALIDATION_SPLIT = 0.2
    
    # Load data
    print("🔄 Loading sequence data...")
    X, y, label_map = load_sequence_data(dataset_path="dataset", sequence_length=SEQUENCE_LENGTH)
    
    # Check if data is loaded
    if len(X) == 0:
        print("❌ No data found! Please run 1_collect_seq.py first.")
        return
    
    # Convert labels to one-hot encoding
    num_classes = len(label_map)
    y_onehot = keras.utils.to_categorical(y, num_classes)
    
    # Normalize data (z-score normalization)
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    print(f"\n📐 Data split:")
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Build model
    print(f"\n🏗️  Building CNN+LSTM model...")
    model = build_cnn_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        num_features=63,
        num_classes=num_classes
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    # Train model
    print(f"\n🚀 Training model for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model
    model_path = model_dir / "model_cnn_lstm.h5"
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Save label map
    label_map_path = model_dir / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"✅ Label map saved to {label_map_path}")
    
    # Save training history
    history_path = model_dir / "training_history.npy"
    np.save(history_path, history.history)
    print(f"✅ Training history saved to {history_path}")
    
    print(f"\n🎯 Final Results:")
    print(f"   Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"   Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
