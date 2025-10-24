# Save this script as validate_on_youtube.py
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm # Progress bar
import time

# --- Configuration ---
# --- IMPORTANT: Set this path correctly! ---
YOUTUBE_REAL_DIR = "./Celeb-DF/videos/Celeb-real" # <--- CHANGE THIS

MODEL_PATH = "cnn_lstm_fake_detector_fully_trained.keras" # Path to your trained model
SEQUENCE_LENGTH = 10 # Must match the training configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3

# --- Load Model (using load_weights approach for robustness) ---
# --- (Copied relevant parts from predictor.py, simplified) ---
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout

def build_cnn_lstm_model():
    """Defines the architecture for the CNN+LSTM model."""
    base_model_cnn = Xception(weights=None, include_top=False,
                              input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), pooling='avg')
    base_model_cnn.trainable = False
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)),
        TimeDistributed(base_model_cnn),
        LSTM(128, return_sequences=False),
        Dropout(0.5), Dense(64, activation='relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="cnn_lstm_model")
    return model

print("Loading trained CNN+LSTM model...")
model = None
error_loading = False
try:
    if not os.path.exists(MODEL_PATH):
         print(f"[ERROR] Model file not found at: {MODEL_PATH}")
         error_loading = True
    else:
        model = build_cnn_lstm_model()
        # Build model with dummy input before loading weights
        dummy_input_lstm = np.zeros((1, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
        _ = model(dummy_input_lstm, training=False)
        model.load_weights(MODEL_PATH)
        print(f"[INFO] Successfully loaded model weights from: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    error_loading = True

if error_loading or model is None:
    print("Exiting due to model loading failure.")
    exit()

# --- Prediction Functions (Copied from predictor.py) ---
def preprocess_frame(frame):
    """Resizes and normalizes a single frame."""
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = img_to_array(frame) / 255.0 # Normalize
    return frame

def predict_video_sequence(video_path):
    """ Predicts a video using the loaded CNN+LSTM model."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Error opening video file: {os.path.basename(video_path)}")
        return None, None # Return None if video can't be opened

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print(f"  [WARN] Video file seems empty or corrupt: {os.path.basename(video_path)}")
        cap.release()
        return None, None

    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = frames[-1]*255.0 if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), dtype=np.uint8)
            processed_frame = preprocess_frame(frame)
            frames.append(processed_frame)
    except Exception as e:
         print(f"  [WARN] Error processing frames for {os.path.basename(video_path)}: {e}")
         cap.release()
         return None, None # Return None if frame processing fails
    finally:
        cap.release()

    # Padding/Truncating
    if len(frames) < SEQUENCE_LENGTH:
        diff = SEQUENCE_LENGTH - len(frames)
        padding_frame = frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
        frames.extend([padding_frame] * diff)
    elif len(frames) > SEQUENCE_LENGTH:
        frames = frames[:SEQUENCE_LENGTH]

    sequence_batch = np.expand_dims(np.array(frames), axis=0)

    # Make prediction
    prediction = model.predict(sequence_batch, verbose=0)[0][0] # verbose=0 suppresses progress bar per prediction

    confidence_real = float(prediction)
    label = "Real" if confidence_real > 0.5 else "Fake"

    return label, confidence_real

# --- Main Validation Pipeline ---
print(f"\nStarting validation on folder: {YOUTUBE_REAL_DIR}")

if not os.path.exists(YOUTUBE_REAL_DIR) or not os.path.isdir(YOUTUBE_REAL_DIR):
    print(f"[ERROR] Validation directory not found: {YOUTUBE_REAL_DIR}")
    exit()

# Get list of video files
youtube_videos = [
    os.path.join(YOUTUBE_REAL_DIR, f)
    for f in os.listdir(YOUTUBE_REAL_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov')) # Add other formats if needed
]

if not youtube_videos:
    print(f"[ERROR] No video files found in {YOUTUBE_REAL_DIR}")
    exit()

print(f"Found {len(youtube_videos)} videos to test.")

correct_predictions = 0
total_predictions = 0
failed_videos = 0
start_time = time.time()

# Iterate through videos with a progress bar
for video_path in tqdm(youtube_videos, unit="video", desc="Validating"):
    pred_label, pred_conf = predict_video_sequence(video_path)

    if pred_label is None: # Handle cases where prediction failed
        failed_videos += 1
        continue

    total_predictions += 1
    ground_truth_label = "Real" # We assume all videos in this folder are real

    if pred_label == ground_truth_label:
        correct_predictions += 1
    # else:
        # Optional: Log misclassified videos
        # tqdm.write(f"  MISCLASSIFIED: {os.path.basename(video_path)} predicted as {pred_label} (Conf: {pred_conf:.4f})")


end_time = time.time()

# --- Report Results ---
print("\n--- Validation Complete ---")
print(f"Processed {total_predictions} videos (skipped {failed_videos} due to errors) in {end_time - start_time:.2f} seconds.")

if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nAccuracy on '{os.path.basename(YOUTUBE_REAL_DIR)}' dataset: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    # You can calculate False Negative Rate here: (total_predictions - correct_predictions) / total_predictions
    fnr = ((total_predictions - correct_predictions) / total_predictions) * 100
    print(f"False Negative Rate (Real videos predicted as Fake): {fnr:.2f}%")

else:
    print("No videos were successfully processed.")

print("Script finished.")