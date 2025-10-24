# utils/predictor.py (Grad-CAM Removed)
import tensorflow as tf
import numpy as np
import cv2
import os
import traceback
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, TimeDistributed, LSTM, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.regularizers import l2
# --- Grad-CAM Imports REMOVED ---

print("[DEBUG] utils/predictor.py: Starting execution...")

# --- Configuration ---
SEQUENCE_LENGTH = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3

# --- Define Model Architectures ---
# ... (Keep build_cnn_lstm_model() and build_cnn_only_model() functions as before) ...
def build_cnn_lstm_model():
    base_model_cnn = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), pooling='avg')
    base_model_cnn.trainable = False
    model = Sequential([ Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), TimeDistributed(base_model_cnn), LSTM(128, return_sequences=False), Dropout(0.5), Dense(64, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid') ], name="cnn_lstm_model")
    print("[DEBUG] CNN+LSTM architecture defined.")
    return model

def build_cnn_only_model():
    base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
    base_model.trainable = False
    model = Sequential([ Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), base_model, GlobalAveragePooling2D(), Dense(128, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.6), Dense(1, activation='sigmoid') ], name="cnn_only_model")
    print("[DEBUG] CNN-Only architecture defined.")
    return model

# --- Load Weights ---
# ... (Keep the weight loading logic exactly as before) ...
model_lstm = None
model_cnn = None
error_loading = False
print("[DEBUG] Attempting to build models and load weights...")
try:
    model_lstm = build_cnn_lstm_model()
    model_cnn = build_cnn_only_model()
    # --- Load CNN+LSTM Weights ---
    model_lstm_weights_path = "cnn_lstm_fake_detector_fully_trained.keras"
    if not os.path.exists(model_lstm_weights_path): model_lstm_weights_path = os.path.join(os.path.dirname(__file__), "..", model_lstm_weights_path)
    if os.path.exists(model_lstm_weights_path):
        dummy_input_lstm = np.zeros((1, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)); _ = model_lstm(dummy_input_lstm, training=False)
        model_lstm.load_weights(model_lstm_weights_path); print(f"[INFO] Successfully loaded LSTM model weights from: {model_lstm_weights_path}")
    else: print(f"[ERROR] LSTM Model weights file not found at: {model_lstm_weights_path}"); error_loading = True; model_lstm = None
    # --- Load CNN-Only Weights ---
    model_cnn_weights_path = "xception_regularized_detector.keras"
    if not os.path.exists(model_cnn_weights_path): model_cnn_weights_path = os.path.join(os.path.dirname(__file__), "..", model_cnn_weights_path)
    if os.path.exists(model_cnn_weights_path):
        dummy_input_cnn = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)); _ = model_cnn(dummy_input_cnn, training=False)
        model_cnn.load_weights(model_cnn_weights_path); print(f"[INFO] Successfully loaded CNN model weights from: {model_cnn_weights_path}")
    else: print(f"[ERROR] CNN Model weights file not found at: {model_cnn_weights_path}"); error_loading = True; model_cnn = None
except Exception as e: print(f"\n\n[FATAL ERROR] Exception during model initialization: {e}"); traceback.print_exc(); error_loading = True; model_lstm = None; model_cnn = None
if error_loading: print("\n--- WARNING: One or more models failed to load weights. ---")
else: print("[DEBUG] Model building and weight loading successful.")

# --- Grad-CAM Function REMOVED ---

# --- Prediction Functions ---
def preprocess_frame(frame):
    """Resizes and normalizes a single frame."""
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = img_to_array(frame) / 255.0 # Normalize
    return frame

def predict_single_image(image_path):
    """Predicts image, returns (label, conf)."""
    if model_cnn is None:
        raise ValueError("CNN-only model weights not loaded.")
    try:
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model_cnn.predict(img_batch, verbose=0)[0][0]
        confidence_real = float(prediction)
        label = "Real" if confidence_real > 0.5 else "Fake"
        return label, confidence_real # Return only 2 values
    except Exception as e:
        print(f"[ERROR] Error during single image prediction: {e}")
        raise ValueError(f"Error predicting image: {e}")


def predict_video_sequence(video_path):
    """ Predicts video, returns (label, conf)."""
    if model_lstm is None:
        raise ValueError("CNN+LSTM model weights not loaded.")
    # ... (Keep the video processing logic exactly as before) ...
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Error opening video file: {video_path}")
    frames, total_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1: raise ValueError(f"Video file seems empty or corrupt: {video_path}")
    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
            if not ret: frame = frames[-1]*255.0 if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), dtype=np.uint8)
            processed_frame = preprocess_frame(frame); frames.append(processed_frame)
    finally: cap.release()
    if len(frames) < SEQUENCE_LENGTH:
        diff = SEQUENCE_LENGTH - len(frames); padding_frame = frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
        frames.extend([padding_frame] * diff)
    elif len(frames) > SEQUENCE_LENGTH: frames = frames[:SEQUENCE_LENGTH]
    sequence_batch = np.expand_dims(np.array(frames), axis=0)
    prediction = model_lstm.predict(sequence_batch, verbose=0)[0][0]
    confidence_real = float(prediction); label = "Real" if confidence_real > 0.5 else "Fake"
    return label, confidence_real # Return only 2 values

print("[DEBUG] utils/predictor.py: Finished execution.")