"""
Convert Public Datasets (CSV / Image) → Landmarks (.npy format)
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def convert_csv_to_npy(csv_file, output_dir="dataset", sequence_length=30):
    """
    Convert CSV landmark data to .npy files
    
    CSV format expected:
    label, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21
    (21 landmarks × 3 = 63 features)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Reading CSV from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Detect label column
    label_col = df.columns[0]  # Usually first column
    
    # Get unique labels
    unique_labels = df[label_col].unique()
    
    print(f"Found {len(unique_labels)} actions: {unique_labels}")
    
    sample_count = {}
    
    for label in unique_labels:
        # Create action folder
        action_dir = output_dir / str(label)
        action_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all rows for this label
        label_data = df[df[label_col] == label]
        
        print(f"\n📍 Processing action: {label} ({len(label_data)} rows)")
        
        # Each row = 1 sequence (assume CSV already has 30 frames stacked)
        for idx, (_, row) in enumerate(label_data.iterrows()):
            # Extract landmarks (skip label column)
            landmarks = row.drop(label_col).values.astype(np.float32)
            
            # Reshape to (30, 63) if length is 1890 (30*63)
            if len(landmarks) == 1890:
                landmarks = landmarks.reshape(30, 63)
            elif len(landmarks) == 63:
                # Single frame - pad to (30, 63)
                landmarks = np.repeat(landmarks[np.newaxis, :], 30, axis=0)
            
            # Save
            save_path = action_dir / f"{label}_sample_{idx}.npy"
            np.save(save_path, landmarks)
            
            if idx % 10 == 0:
                print(f"   ✅ Saved sample {idx}")
        
        sample_count[label] = len(label_data)
        print(f"   ✅ Total: {len(label_data)} samples")
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Summary:")
    for label, count in sample_count.items():
        print(f"   {label}: {count} samples")


def convert_image_folder_to_npy(image_dir, output_dir="dataset", sequence_length=30):
    """
    Convert image folders (with MediaPipe extraction)
    
    Assumes: image_dir/action_name/*.jpg
    """
    import cv2
    import mediapipe as mp
    
    mp_holistic = mp.solutions.holistic
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get action folders
    action_folders = [d for d in image_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(action_folders)} action folders")
    
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as holistic:
        
        for action_folder in action_folders:
            action_name = action_folder.name
            output_action_dir = output_dir / action_name
            output_action_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📍 Processing action: {action_name}")
            
            # Get all images
            image_files = sorted(action_folder.glob("*.jpg")) + sorted(action_folder.glob("*.png"))
            
            # Group into sequences of 30
            for seq_idx in range(0, len(image_files), sequence_length):
                seq_images = image_files[seq_idx:seq_idx + sequence_length]
                
                if len(seq_images) < sequence_length:
                    # Pad with last image
                    while len(seq_images) < sequence_length:
                        seq_images.append(seq_images[-1])
                
                landmarks_seq = []
                
                for img_path in seq_images:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Extract landmarks
                    results = holistic.process(img_rgb)
                    
                    # Right hand landmarks
                    if results.right_hand_landmarks:
                        landmarks = []
                        for lm in results.right_hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks_seq.append(landmarks)
                    else:
                        landmarks_seq.append([0.0] * 63)
                
                # Save sequence
                if len(landmarks_seq) == sequence_length:
                    save_path = output_action_dir / f"{action_name}_sample_{seq_idx//sequence_length}.npy"
                    np.save(save_path, np.array(landmarks_seq, dtype=np.float32))
                    print(f"   ✅ Saved sample {seq_idx//sequence_length}")
            
            print(f"   ✅ Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser("Convert Public Datasets to Landmark Format")
    parser.add_argument("--type", choices=["csv", "images"], required=True,
                       help="Dataset type: csv or images")
    parser.add_argument("--input", required=True,
                       help="Input file (CSV) or directory (images)")
    parser.add_argument("--output", default="dataset",
                       help="Output directory (default: dataset/)")
    parser.add_argument("--seq_length", type=int, default=30,
                       help="Sequence length (default: 30)")
    
    args = parser.parse_args()
    
    if args.type == "csv":
        convert_csv_to_npy(args.input, args.output, args.seq_length)
    elif args.type == "images":
        convert_image_folder_to_npy(args.input, args.output, args.seq_length)


if __name__ == "__main__":
    main()
