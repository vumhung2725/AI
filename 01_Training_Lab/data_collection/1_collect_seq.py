"""
Sequence-based Pose Data Collection (MediaPipe Holistic)
Collect 30 consecutive frames for each action
Output: .npy files with shape (30, 63) for each sample
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# MediaPipe 0.10.x uses new API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create hand landmarker
def create_hand_landmarker():
    """Create MediaPipe hand landmarker"""
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    return vision.HandLandmarker.create_from_options(options)


def extract_right_hand_landmarks(results):
    """
    Extract right hand landmarks from MediaPipe results
    Returns: numpy array of shape (63,) = 21 landmarks × 3 coordinates
    """
    landmarks = []
    
    # Handle new MediaPipe API
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])
            break  # Only first hand
    else:
        # If no hand detected, return zeros
        landmarks = [0.0] * 63
    
    return np.array(landmarks, dtype=np.float32)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Sequence Pose Data Collection")
    parser.add_argument("--action_name", help="Name of the action (e.g., 'hello')", type=str, default="hello")
    parser.add_argument("--num_samples", help="Number of samples to collect", type=int, default=10)
    parser.add_argument("--sequence_length", help="Frames per sample", type=int, default=30)
    parser.add_argument("--confidence", help="MediaPipe confidence", type=float, default=0.7)
    
    args = parser.parse_args()
    
    # Create dataset folder
    dataset_path = Path("dataset") / args.action_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎬 COLLECTING DATA FOR: {args.action_name.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {args.num_samples}")
    print(f"Frames/sample: {args.sequence_length}")
    print(f"Total frames: {args.num_samples} × {args.sequence_length} = {args.num_samples * args.sequence_length}")
    print(f"Output: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Initialize MediaPipe Hand Landmarker (new API - VIDEO mode)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    sample_count = 0
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        
        while sample_count < args.num_samples:
            # Collect one sequence
            sequence_data = []
            frame_count = 0
            
            print(f"\n📹 SAMPLE {sample_count + 1}/{args.num_samples}")
            print(f"⏳ Get ready... Starting in 3 seconds")
            print(f"   (Press 'q' during countdown to skip, 's' to save early)")
            
            # Countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera error!")
                    return
                
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                # Show countdown
                cv2.putText(frame, f"Starting in {i}...", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Data Collection", frame)
                
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    sample_count = args.num_samples  # Exit
                    break
            
            if sample_count == args.num_samples:
                break
            
            print("🔴 RECORDING... Perform the action now!")
            
            # Collect frames
            while frame_count < args.sequence_length:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera error!")
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe (VIDEO mode requires timestamp)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms = int(time.time() * 1000)
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # Extract landmarks
                landmarks = extract_right_hand_landmarks(results)
                sequence_data.append(landmarks)
                
                # Draw landmarks (simplified for new API)
                if results.hand_landmarks:
                    # Draw dots for each landmark
                    for hand_landmarks in results.hand_landmarks:
                        for lm in hand_landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw progress bar
                progress = (frame_count + 1) / args.sequence_length
                bar_length = int(progress * 25)
                bar = "█" * bar_length + "░" * (25 - bar_length)
                
                cv2.putText(frame, f"[{bar}] {frame_count + 1}/{args.sequence_length}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Action: {args.action_name}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "Press 's' to save early, 'q' to skip",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow("Data Collection", frame)
                
                # Check keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("❌ Skipped this sample")
                    sequence_data = []
                    break
                elif key == ord('s'):
                    print("⏹️  Saving early...")
                    break
                
                frame_count += 1
            
            # Save if we have frames
            if len(sequence_data) == args.sequence_length:
                sequence_array = np.array(sequence_data)
                save_path = dataset_path / f"{args.action_name}_{sample_count:03d}.npy"
                np.save(save_path, sequence_array)
                print(f"✅ SAVED: {save_path.name}")
                print(f"   Shape: {sequence_array.shape}")
                sample_count += 1
            elif len(sequence_data) > 0:
                print(f"⚠️  Incomplete sample ({len(sequence_data)}/{args.sequence_length} frames)")
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE! Collected {sample_count}/{args.num_samples} samples")
    print(f"📁 Saved to: {dataset_path}")
    print(f"{'='*60}\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
