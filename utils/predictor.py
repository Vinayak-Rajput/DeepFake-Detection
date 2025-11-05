# utils/predictor.py (Updated for conditional XAI)
import tensorflow as tf
import numpy as np
import cv2
import os
import traceback
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# Import necessary layers and models
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, TimeDistributed, LSTM, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization # Make sure BatchNormalization is imported if used in models
)
from tensorflow.keras.regularizers import l2
# --- LIME Imports ---
# Important: Handle potential import errors if lime isn't installed
try:
    import lime
    from lime import lime_image
    # skimage.segmentation is used internally by lime, mark_boundaries is removed
    import matplotlib.pyplot as plt # Needed for colormap and saving
    lime_available = True
    print("[DEBUG] LIME library loaded.")
except ImportError:
    print("[WARN] LIME library not found. Explainability disabled. Install with 'pip install lime scikit-image'.")
    lime_available = False
    # Define dummy classes/functions if lime is not available to prevent NameErrors later
    class LimeImageExplainer: pass # Dummy class
    # Dummy mark_boundaries replacement (returns original image) - NOT USED IN OVERLAY
    # def mark_boundaries(a,b): return a
    import sys
    # --- Corrected MockPlt ---
    class MockPlt:
        def imsave(self, *args, **kwargs):
            pass # Method body needs to be indented
        # Add dummy get_cmap if needed, though cv2 handles colormap now
        def get_cmap(self, *args, **kwargs):
             # Return a dummy function that mimics colormap behavior (e.g., returns input scaled)
             def dummy_cmap(arr):
                 # Simple grayscale-like mapping for the dummy
                 scaled = np.clip(arr, 0, 1)
                 return np.stack([scaled]*3 + [np.ones_like(arr)], axis=-1) # RGBA
             return dummy_cmap

    # --- END CORRECTION ---
    plt = MockPlt()
# --- End LIME Imports ---

print("[DEBUG] utils/predictor.py: Starting execution...")

# --- Configuration ---
SEQUENCE_LENGTH = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3

# --- Define Model Architectures ---
def build_cnn_lstm_model():
    """Defines the architecture for the CNN+LSTM model."""
    base_model_cnn = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), pooling='avg')
    base_model_cnn.trainable = False
    model = Sequential([ Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), TimeDistributed(base_model_cnn), LSTM(128, return_sequences=False), Dropout(0.5), Dense(64, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid') ], name="cnn_lstm_model")
    print("[DEBUG] CNN+LSTM architecture defined.")
    return model

def build_cnn_only_model():
    """Defines the architecture for the regularized Xception CNN-only model."""
    base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), name="xception_base")
    base_model.trainable = False
    model = Sequential([ Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), base_model, GlobalAveragePooling2D(), Dense(128, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.6), Dense(1, activation='sigmoid') ], name="cnn_only_model")
    print("[DEBUG] CNN-Only architecture defined.")
    return model

# --- Load Weights ---
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


# --- LIME Setup ---
explainer = None
if lime_available:
    try:
        explainer = lime_image.LimeImageExplainer()
        print("[DEBUG] LIME explainer initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LIME explainer: {e}")
        lime_available = False # Disable LIME if init fails


def cnn_predict_proba(images):
    """ Wrapper function for the CNN model's prediction needed by LIME. """
    if model_cnn is None: return np.zeros((len(images), 2)) # Shape (num_images, 2 classes)
    try:
        # Assuming LIME provides images in float [0,1] range matching model input
        preds = model_cnn.predict(images, verbose=0) # Prob_Real
        prob_fake = 1.0 - preds
        prob_real = preds
        return np.hstack((prob_fake, prob_real)) # Return shape (n_images, 2) -> [P(Fake), P(Real)]
    except Exception as e:
        print(f"[ERROR] Exception in cnn_predict_proba: {e}")
        return np.zeros((len(images), 2)) # Return default on error

# --- LIME Generation Function (OVERLAY VERSION) ---
def generate_lime_explanation(image_path, predicted_label_index):
    """
    Generates a LIME explanation overlay heatmap.
    predicted_label_index: 0 for Fake, 1 for Real (based on model output)
    """
    if not lime_available or explainer is None:
        print("[WARN] LIME not available or not initialized. Skipping explanation.")
        return None
    if model_cnn is None:
        print("[WARN] Cannot generate LIME: CNN Model not loaded.")
        return None

    try:
        # Load image as numpy array (float 0-255 range for LIME)
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array_0_255 = img_to_array(img) # Default is float32

        print("[INFO] Generating LIME explanation overlay (this may take a moment)...")
        explanation = explainer.explain_instance(
            image=img_array_0_255.astype('double'), # LIME often prefers double
            classifier_fn=cnn_predict_proba,
            top_labels=1,             # Explain the top prediction class
            hide_color=0,             # Use gray to hide pixels
            num_samples=1000,         # Number of perturbations
            num_features=100          # Consider all features initially for heatmap
        )

        if not explanation.top_labels: # Handle cases where LIME fails to explain
             print("[WARN] LIME did not produce top labels.")
             return None
        label_to_explain = explanation.top_labels[0]
        local_exp = explanation.local_exp.get(label_to_explain)
        if local_exp is None:
            print(f"[WARN] LIME did not produce explanation weights for label {label_to_explain}.")
            return None

        mask = np.zeros(explanation.segments.shape, dtype=float)
        weight_sum = 0.0
        for feature_index, weight in local_exp:
            mask[explanation.segments == feature_index] = max(0, weight) # Show only positive influence for Viridis
            weight_sum += abs(weight) # Keep track for normalization check
        
        if weight_sum > 1e-6: # Avoid division by zero
             max_weight = np.max(mask)
             if max_weight > 1e-6: mask_norm = mask / max_weight
             else: mask_norm = np.zeros_like(mask) # Handle case where max is zero/negative
        else: mask_norm = np.zeros_like(mask)
        
        heatmap = plt.get_cmap('viridis')(mask_norm)[..., :3] # Get RGB, discard alpha
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        original_img_bgr_uint8 = cv2.cvtColor(img_array_0_255.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        if heatmap_uint8.shape[:2] != original_img_bgr_uint8.shape[:2]:
            heatmap_uint8 = cv2.resize(heatmap_uint8, (original_img_bgr_uint8.shape[1], original_img_bgr_uint8.shape[0]))
        
        overlay = cv2.addWeighted(original_img_bgr_uint8, 0.6, heatmap_uint8, 0.4, 0)
        
        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)
        explanation_filename = f"limeoverlay_{name}{ext}" # New prefix
        explanation_save_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(explanation_save_dir, exist_ok=True)
        explanation_path = os.path.join(explanation_save_dir, explanation_filename)

        if not os.access(os.path.dirname(explanation_path), os.W_OK):
             print(f"[ERROR] Cannot write LIME overlay to {explanation_path}.")
             return None
        
        cv2.imwrite(explanation_path, overlay)
        print(f"[INFO] LIME explanation overlay saved to: {explanation_path}")
        return explanation_filename # Return filename

    except Exception as e:
        print(f"[ERROR] Failed to generate LIME overlay: {e}")
        traceback.print_exc()
        return None


# --- Prediction Functions ---
def preprocess_frame(frame):
    """Resizes and normalizes a single frame."""
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = img_to_array(frame) / 255.0 # Normalize
    return frame

# --- UPDATED: predict_single_image now accepts generate_xai flag ---
def predict_single_image(image_path, generate_xai=False):
    """Predicts image, generates LIME, returns (label, conf, explanation_filename)."""
    if model_cnn is None: raise ValueError("CNN-only model weights not loaded.")
    explanation_file = None; label = "Error"; confidence_real = 0.0
    try:
        # Prediction
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model_cnn.predict(img_batch, verbose=0)[0][0]
        confidence_real = float(prediction)
        label_index = 1 if confidence_real > 0.5 else 0
        label = "Real" if label_index == 1 else "Fake"

        # --- UPDATED: Conditional LIME Generation ---
        if generate_xai and lime_available:
            print("[INFO] User requested XAI. Generating LIME...")
            explanation_file = generate_lime_explanation(image_path, label_index)
        elif generate_xai:
            print("[WARN] User requested XAI, but LIME library is not available.")
        else:
            print("[INFO] User did not request XAI. Skipping.")
        
        return label, confidence_real, explanation_file

    except Exception as e:
        print(f"[ERROR] Error during single image prediction or LIME: {e}")
        return label, confidence_real, None


def predict_video_sequence(video_path):
    """ Predicts video, returns (label, conf, None)."""
    if model_lstm is None: raise ValueError("CNN+LSTM model weights not loaded.")
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
    return label, confidence_real, None # Return None for explanation file

print("[DEBUG] utils/predictor.py: Finished execution.")