# Save this script as validate_on_fakes.py
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm # Progress bar
import time
import argparse # For command-line arguments

# --- Configuration ---
# --- IMPORTANT: Set this default path or provide via command line ---
DEFAULT_FAKE_DIR = "./Celeb-DF/videos/Celeb-synthesis" # <--- CHANGE THIS DEFAULT if needed

MODEL_PATH = "cnn_lstm_fake_detector_fully_trained.keras" # Path to your trained model
SEQUENCE_LENGTH = 10 # Must match the training configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3

# --- Load Model (using load_weights approach for robustness) ---
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
        return None, None

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
         return None, None
    finally:
        cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        diff = SEQUENCE_LENGTH - len(frames)
        padding_frame = frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
        frames.extend([padding_frame] * diff)
    elif len(frames) > SEQUENCE_LENGTH:
        frames = frames[:SEQUENCE_LENGTH]

    sequence_batch = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(sequence_batch, verbose=0)[0][0]

    confidence_real = float(prediction)
    # --- Label logic: assumes {'fake': 0, 'real': 1} ---
    label = "Real" if confidence_real > 0.5 else "Fake"

    return label, confidence_real

# --- Main Validation Pipeline ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on a directory of FAKE videos.")
    parser.add_argument("--fake_dir", type=str, default=DEFAULT_FAKE_DIR,
                        help="Path to the directory containing fake video files.")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to the trained .keras model file.")
    args = parser.parse_args()

    # Update model path if provided via argument
    MODEL_PATH = args.model
    # Ensure the model is loaded (code above handles this, re-check needed if path changed)
    if model is None:
         print(f"Model failed to load from {MODEL_PATH}, exiting.")
         exit()


    fake_video_dir = args.fake_dir
    print(f"\n--- Starting Validation on FAKE Videos ---")
    print(f"Target folder: {fake_video_dir}")

    if not os.path.exists(fake_video_dir) or not os.path.isdir(fake_video_dir):
        print(f"[ERROR] Validation directory not found: {fake_video_dir}")
        exit()

    # Get list of video files
    fake_videos = [
        os.path.join(fake_video_dir, f)
        for f in os.listdir(fake_video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]

    if not fake_videos:
        print(f"[ERROR] No video files found in {fake_video_dir}")
        exit()

    print(f"Found {len(fake_videos)} videos to test.")

    correct_fake_predictions = 0 # Videos correctly predicted as "Fake"
    total_predictions = 0
    failed_videos = 0
    start_time = time.time()

    for video_path in tqdm(fake_videos, unit="video", desc="Validating Fakes"):
        pred_label, pred_conf = predict_video_sequence(video_path)

        if pred_label is None:
            failed_videos += 1
            continue

        total_predictions += 1
        ground_truth_label = "Fake" # <<< Ground truth is FAKE for this script

        if pred_label == ground_truth_label:
            correct_fake_predictions += 1
        # else:
            # Optional: Log misclassified fakes (predicted as Real)
            # tqdm.write(f"  MISCLASSIFIED (False Positive): {os.path.basename(video_path)} predicted as {pred_label} (Conf_Real: {pred_conf:.4f})")

    end_time = time.time()

    # --- Report Results ---
    print("\n--- Validation Complete ---")
    print(f"Processed {total_predictions} videos (skipped {failed_videos} due to errors) in {end_time - start_time:.2f} seconds.")

    if total_predictions > 0:
        # Accuracy = True Negative Rate (TNR)
        tnr = (correct_fake_predictions / total_predictions) * 100
        print(f"\nTrue Negative Rate (Correctly identified Fakes): {tnr:.2f}% ({correct_fake_predictions}/{total_predictions})")

        # False Positive Rate (FPR) - Fakes predicted as Real
        fpr = ((total_predictions - correct_fake_predictions) / total_predictions) * 100
        print(f"False Positive Rate (Fakes predicted as Real): {fpr:.2f}%")

    else:
        print("No videos were successfully processed.")

    print("Script finished.")