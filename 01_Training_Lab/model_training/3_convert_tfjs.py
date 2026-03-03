"""
Convert Keras/TensorFlow model to TensorFlow.js format
Output: model.json + .bin files for use in browser
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import json


def convert_to_tfjs():
    """Convert trained Keras model to TFJS format"""
    
    model_dir = Path("01_Training_Lab/model_training")
    output_dir = Path("02_Web_Client/assets/models")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    h5_model_path = model_dir / "model_cnn_lstm.h5"
    label_map_path = model_dir / "label_map.json"
    
    # Check if model exists
    if not h5_model_path.exists():
        print(f"❌ Model not found at {h5_model_path}")
        print("Please run 2_train_hybrid.py first!")
        return
    
    print(f"🔄 Loading Keras model from {h5_model_path}...")
    model = tf.keras.models.load_model(h5_model_path)
    
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Convert to TFJS
    print(f"\n🔄 Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(
        model,
        f"{output_dir}/tfjs_model"
    )
    
    print(f"✅ Model converted and saved to {output_dir}/tfjs_model/")
    
    # Copy label map to assets
    if label_map_path.exists():
        import shutil
        label_dest = output_dir / "label_map.json"
        shutil.copy(label_map_path, label_dest)
        print(f"✅ Label map copied to {label_dest}")
    
    # Create a simple prediction test
    print(f"\n🧪 Testing model prediction...")
    test_input = model.predict(np.zeros((1, 30, 63)))  # Dummy input
    print(f"   Output shape: {test_input.shape}")
    print(f"   Output probabilities: {test_input[0]}")
    
    print(f"\n✅ Conversion complete!")
    print(f"\n📝 To use in browser:")
    print(f"   Load model: await tf.loadLayersModel('assets/models/tfjs_model/model.json')")
    print(f"   Input shape: (batch, 30, 63)")
    print(f"\n📁 Files created:")
    print(f"   - {output_dir}/tfjs_model/model.json")
    print(f"   - {output_dir}/tfjs_model/group1-shard*.bin")
    print(f"   - {output_dir}/label_map.json")


if __name__ == "__main__":
    import numpy as np
    convert_to_tfjs()
