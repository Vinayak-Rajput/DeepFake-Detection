# Save this as verify_videos.py
import os
import cv2
from tqdm import tqdm # Progress bar library (install with: pip install tqdm)
import time
import argparse # For command-line arguments (optional)

def verify_videos(base_video_dir, real_subdir, fake_subdir, frames_to_check=5, output_log="corrupt_videos.log"):
    """
    Iterates through video files in specified directories, checks if they can be opened
    and if initial frames can be read using OpenCV. Logs potentially corrupt files.

    Args:
        base_video_dir (str): Path to the main directory containing video subdirs.
        real_subdir (str): Subdirectory name for real videos.
        fake_subdir (str): Subdirectory name for fake videos.
        frames_to_check (int): How many initial frames to attempt reading.
        output_log (str): File path to save the list of corrupt files.
    """
    print("--- Video Verification Script ---")
    
    # --- Construct Full Paths ---
    real_video_dir = os.path.join(base_video_dir, real_subdir)
    fake_video_dir = os.path.join(base_video_dir, fake_subdir)

    # --- Get List of All Video Files ---
    print("Scanning video directories...")
    real_files = []
    fake_files = []
    all_video_files = []
    
    if os.path.exists(real_video_dir) and os.path.isdir(real_video_dir):
        real_files = [os.path.join(real_video_dir, f) for f in os.listdir(real_video_dir) if f.lower().endswith('.mp4')]
        all_video_files.extend(real_files)
        print(f"Found {len(real_files)} videos in {real_subdir}")
    else:
        print(f"ERROR: Real video directory not found or not a directory: {real_video_dir}")

    if os.path.exists(fake_video_dir) and os.path.isdir(fake_video_dir):
        fake_files = [os.path.join(fake_video_dir, f) for f in os.listdir(fake_video_dir) if f.lower().endswith('.mp4')]
        all_video_files.extend(fake_files)
        print(f"Found {len(fake_files)} videos in {fake_subdir}")
    else:
        print(f"ERROR: Fake video directory not found or not a directory: {fake_video_dir}")

    if not all_video_files:
        print("\nNo MP4 video files found in the specified directories. Exiting.")
        return

    print(f"\nFound {len(all_video_files)} total video files to verify.")

    # --- Verification Loop ---
    corrupt_files_list = []
    start_time = time.time()

    print(f"\nVerifying videos (attempting to read first {frames_to_check} frames)...")
    
    # Use tqdm for a progress bar over the video files
    for video_path in tqdm(all_video_files, unit="video", desc="Checking Videos"):
        if not os.path.exists(video_path):
            tqdm.write(f"WARNING: File listed but not found: {video_path}") # Use tqdm.write inside loop
            corrupt_files_list.append(video_path + " # File Not Found")
            continue

        cap = None  # Ensure cap is defined for the finally block
        error_reason = ""
        is_corrupt = False
        
        try:
            cap = cv2.VideoCapture(video_path)

            # 1. Check if video opened successfully
            if not cap.isOpened():
                is_corrupt = True
                error_reason = "Failed to open"
            else:
                # 2. Try reading specified number of frames
                frames_read_count = 0
                for i in range(frames_to_check):
                    ret, frame = cap.read()
                    if not ret:
                        # If reading fails early, it might be corrupt or just very short.
                        # We'll flag it as potentially corrupt for review.
                        is_corrupt = True
                        error_reason = f"Failed to read frame {i+1}"
                        break # Stop trying to read frames for this video
                    frames_read_count += 1
                
                # Optional: Add a check for zero total frames if needed
                # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # if total_frames <= 0:
                #    is_corrupt = True
                #    error_reason = "Reported zero total frames"

        except Exception as e:
            # Catch unexpected errors during OpenCV processing
            is_corrupt = True
            error_reason = f"Exception: {str(e)}"
            tqdm.write(f"ERROR processing {os.path.basename(video_path)}: {e}") # Log exception

        finally:
            if cap is not None and cap.isOpened():
                cap.release() # Ensure video file is released

        if is_corrupt:
            corrupt_files_list.append(f"{video_path} # Reason: {error_reason}")
            # Optional: print immediately using tqdm.write to avoid breaking progress bar
            # tqdm.write(f"Potential Issue: {video_path} ({error_reason})")

    end_time = time.time()
    
    # --- Results ---
    print("\n--- Verification Complete ---")
    print(f"Checked {len(all_video_files)} videos in {end_time - start_time:.2f} seconds.")
    
    if corrupt_files_list:
        print(f"\nFound {len(corrupt_files_list)} potentially corrupt or problematic files:")
        # Save the list to the log file
        try:
            with open(output_log, 'w') as f:
                for file_path in corrupt_files_list:
                    print(file_path) # Also print to console
                    f.write(file_path + "\n")
            print(f"\nList of problematic files saved to: {output_log}")
        except Exception as e:
            print(f"\nERROR: Could not write log file '{output_log}': {e}")
            print("Problematic files:")
            for file_path in corrupt_files_list:
                 print(file_path)
    else:
        print("\n✅ All video files seem readable by OpenCV.")

if __name__ == "__main__":
    # --- Use Argparse for flexibility (Optional) ---
    parser = argparse.ArgumentParser(description="Verify video files using OpenCV.")
    parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/Celeb-DF/videos",
                        help="Base directory containing real and fake video subdirectories.")
    parser.add_argument("--real_subdir", type=str, default="Celeb-real",
                        help="Subdirectory name for real videos.")
    parser.add_argument("--fake_subdir", type=str, default="Celeb-synthesis",
                        help="Subdirectory name for fake videos.")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of initial frames to check per video.")
    parser.add_argument("--log_file", type=str, default="corrupt_videos.log",
                        help="Output file to log potentially corrupt video paths.")
    
    args = parser.parse_args()

    # Run the verification function
    verify_videos(args.base_dir, args.real_subdir, args.fake_subdir, args.frames, args.log_file)