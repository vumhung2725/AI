"""
Extract Landmarks from Video Files (WLASL, MS-ASL, or custom videos)
Convert video → landmarks .npy files
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_landmarks_from_video(video_path, sequence_length=30, confidence=0.5):
    """
    Extract hand landmarks from a video file
    
    Returns:
        sequences: list of numpy arrays with shape (seq_len, 63)
    """
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    
    sequences = []
    current_sequence = []
    frame_count = 0
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                # End of video
                if len(current_sequence) == sequence_length:
                    sequences.append(np.array(current_sequence, dtype=np.float32))
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            results = holistic.process(rgb_frame)
            
            # Extract right hand landmarks
            if results.right_hand_landmarks:
                landmarks = []
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                current_sequence.append(landmarks)
            else:
                # If hand not detected, add zeros
                current_sequence.append([0.0] * 63)
            
            frame_count += 1
            
            # When sequence is full, save it
            if len(current_sequence) == sequence_length:
                sequences.append(np.array(current_sequence, dtype=np.float32))
                # Sliding window: remove first frame, add new frame
                current_sequence = current_sequence[1:]
    
    cap.release()
    return sequences


def process_video_dataset(video_dir, output_dir="dataset", sequence_length=30, confidence=0.5):
    """
    Process directory of videos organized by action
    
    Expected structure:
    video_dir/
    ├── hello/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── thank_you/
    │   └── ...
    └── ...
    """
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get action folders
    action_folders = [d for d in video_dir.iterdir() if d.is_dir()]
    
    print(f"🎬 Found {len(action_folders)} action folders\n")
    
    total_samples = {}
    
    for action_folder in sorted(action_folders):
        action_name = action_folder.name
        output_action_dir = output_dir / action_name
        output_action_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_files = list(action_folder.glob("*.mp4")) + \
                     list(action_folder.glob("*.avi")) + \
                     list(action_folder.glob("*.mov"))
        
        print(f"📍 Processing action: {action_name}")
        print(f"   Found {len(video_files)} videos")
        
        sample_idx = 0
        
        for video_file in tqdm(video_files, desc=f"   Processing videos"):
            # Extract sequences from video
            sequences = extract_landmarks_from_video(
                video_file,
                sequence_length=sequence_length,
                confidence=confidence
            )
            
            # Save each sequence
            for seq in sequences:
                save_path = output_action_dir / f"{action_name}_sample_{sample_idx}.npy"
                np.save(save_path, seq)
                sample_idx += 1
        
        total_samples[action_name] = sample_idx
        print(f"   ✅ Extracted {sample_idx} sequences\n")
    
    # Print summary
    print("\n" + "="*50)
    print("✅ DATASET CONVERSION COMPLETE")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"\nSummary:")
    for action, count in total_samples.items():
        print(f"   {action}: {count} sequences")
    print(f"\nTotal: {sum(total_samples.values())} sequences across {len(total_samples)} actions")


def main():
    parser = argparse.ArgumentParser("Extract Landmarks from Videos")
    parser.add_argument("--video_dir", required=True,
                       help="Directory containing video folders organized by action")
    parser.add_argument("--output", default="dataset",
                       help="Output directory (default: dataset/)")
    parser.add_argument("--seq_length", type=int, default=30,
                       help="Sequence length (default: 30)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="MediaPipe confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    process_video_dataset(
        args.video_dir,
        args.output,
        args.seq_length,
        args.confidence
    )


if __name__ == "__main__":
    main()
