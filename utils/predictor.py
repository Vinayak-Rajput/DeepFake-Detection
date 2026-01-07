# utils/predictor.py (Interactive LIME Data Support)
import tensorflow as tf
import numpy as np
import cv2
import os
import traceback
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# --- LIME Imports ---
try:
    import lime
    from lime import lime_image
    import matplotlib.pyplot as plt
    lime_available = True
    print("[DEBUG] LIME library loaded.")
except ImportError:
    print("[WARN] LIME library not found. Explainability disabled.")
    lime_available = False
    class LimeImageExplainer: pass
    class MockPlt:
        def imsave(self, *args, **kwargs): pass
        def get_cmap(self, *args, **kwargs): return lambda arr: np.stack([arr]*3 + [np.ones_like(arr)], axis=-1)
    plt = MockPlt()

print("[DEBUG] utils/predictor.py: Starting execution...")

# --- Configuration ---
SEQUENCE_LENGTH = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3

# --- Model Definitions (Keep exactly as before) ---
def build_cnn_lstm_model():
    base_model_cnn = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), pooling='avg')
    base_model_cnn.trainable = False
    model = Sequential([ Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), TimeDistributed(base_model_cnn), LSTM(128, return_sequences=False), Dropout(0.5), Dense(64, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid') ], name="cnn_lstm_model")
    return model

def build_cnn_only_model():
    base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), name="xception_base")
    base_model.trainable = False
    model = Sequential([ Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)), base_model, GlobalAveragePooling2D(), Dense(128, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.6), Dense(1, activation='sigmoid') ], name="cnn_only_model")
    return model

# --- Load Weights (Keep exactly as before) ---
model_lstm = None
model_cnn = None
error_loading = False
try:
    model_lstm = build_cnn_lstm_model()
    model_cnn = build_cnn_only_model()
    model_lstm_weights_path = "cnn_lstm_fake_detector_fully_trained.keras"
    if not os.path.exists(model_lstm_weights_path): model_lstm_weights_path = os.path.join(os.path.dirname(__file__), "..", model_lstm_weights_path)
    if os.path.exists(model_lstm_weights_path):
        dummy_input_lstm = np.zeros((1, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)); _ = model_lstm(dummy_input_lstm, training=False)
        model_lstm.load_weights(model_lstm_weights_path); print(f"[INFO] Loaded LSTM weights.")
    else: model_lstm = None
    
    model_cnn_weights_path = "xception_regularized_detector.keras"
    if not os.path.exists(model_cnn_weights_path): model_cnn_weights_path = os.path.join(os.path.dirname(__file__), "..", model_cnn_weights_path)
    if os.path.exists(model_cnn_weights_path):
        dummy_input_cnn = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)); _ = model_cnn(dummy_input_cnn, training=False)
        model_cnn.load_weights(model_cnn_weights_path); print(f"[INFO] Loaded CNN weights.")
    else: model_cnn = None
except Exception as e: print(f"[FATAL ERROR] Model initialization: {e}"); traceback.print_exc(); model_lstm = None; model_cnn = None


# --- LIME Setup ---
explainer = None
if lime_available:
    try:
        explainer = lime_image.LimeImageExplainer()
    except Exception as e:
        print(f"[ERROR] LIME init failed: {e}"); lime_available = False

def cnn_predict_proba(images):
    if model_cnn is None: return np.zeros((len(images), 2))
    try:
        preds = model_cnn.predict(images, verbose=0)
        prob_fake = 1.0 - preds
        prob_real = preds
        return np.hstack((prob_fake, prob_real))
    except Exception as e:
        print(f"[ERROR] Predict proba error: {e}"); return np.zeros((len(images), 2))

# --- Updated LIME Generation Function ---
def generate_lime_explanation(image_path, predicted_label_index):
    """Generates LIME overlay and returns filename AND raw interactive data."""
    if not lime_available or explainer is None: return None, None
    if model_cnn is None: return None, None

    try:
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array_0_255 = img_to_array(img)

        print("[INFO] Generating LIME explanation...")
        explanation = explainer.explain_instance(
            image=img_array_0_255.astype('double'),
            classifier_fn=cnn_predict_proba,
            top_labels=1, hide_color=0, num_samples=1000, num_features=100
        )

        if not explanation.top_labels: return None, None
        label_to_explain = explanation.top_labels[0]
        local_exp = explanation.local_exp.get(label_to_explain)
        if local_exp is None: return None, None

        # --- 1. Generate Static Overlay (Keep this for history/simple view) ---
        mask = np.zeros(explanation.segments.shape, dtype=float)
        weight_sum = 0.0
        for feature_index, weight in local_exp:
            mask[explanation.segments == feature_index] = max(0, weight)
            weight_sum += abs(weight)
        
        if weight_sum > 1e-6:
             max_weight = np.max(mask)
             if max_weight > 1e-6: mask_norm = mask / max_weight
             else: mask_norm = np.zeros_like(mask)
        else: mask_norm = np.zeros_like(mask)
        
        heatmap = plt.get_cmap('viridis')(mask_norm)[..., :3]
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        original_img_bgr_uint8 = cv2.cvtColor(img_array_0_255.astype(np.uint8), cv2.COLOR_RGB2BGR)
        if heatmap_uint8.shape[:2] != original_img_bgr_uint8.shape[:2]:
            heatmap_uint8 = cv2.resize(heatmap_uint8, (original_img_bgr_uint8.shape[1], original_img_bgr_uint8.shape[0]))
        
        overlay = cv2.addWeighted(original_img_bgr_uint8, 0.6, heatmap_uint8, 0.4, 0)
        
        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)
        explanation_filename = f"limeoverlay_{name}{ext}"
        explanation_save_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(explanation_save_dir, exist_ok=True)
        explanation_path = os.path.join(explanation_save_dir, explanation_filename)
        
        if os.access(os.path.dirname(explanation_path), os.W_OK):
             cv2.imwrite(explanation_path, overlay)
        else:
             explanation_filename = None # Fallback if write fails

        # --- 2. Prepare Interactive Data ---
        # Convert numpy segments array to list for JSON serialization
        segments_list = explanation.segments.tolist()
        # Create dictionary of weights: {segment_id: weight}
        weights_dict = {int(seg_id): float(weight) for seg_id, weight in local_exp}
        
        lime_data = {
            "segments": segments_list,
            "weights": weights_dict,
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT
        }

        return explanation_filename, lime_data

    except Exception as e:
        print(f"[ERROR] Failed to generate LIME: {e}"); traceback.print_exc()
        return None, None

# --- Updated Prediction Functions ---
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)); frame = img_to_array(frame) / 255.0
    return frame

def predict_single_image(image_path, generate_xai=False):
    if model_cnn is None: raise ValueError("CNN model not loaded.")
    explanation_file = None; lime_data = None; label = "Error"; confidence_real = 0.0
    try:
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model_cnn.predict(img_batch, verbose=0)[0][0]
        confidence_real = float(prediction)
        label_index = 1 if confidence_real > 0.5 else 0
        label = "Real" if label_index == 1 else "Fake"

        if generate_xai and lime_available:
            print("[INFO] Generating LIME...")
            # Now receives two values
            explanation_file, lime_data = generate_lime_explanation(image_path, label_index)
        
        return label, confidence_real, explanation_file, lime_data # Return 4 values

    except Exception as e:
        print(f"[ERROR] Error during image prediction: {e}")
        return label, confidence_real, None, None

def predict_video_sequence(video_path):
    # ... (Same video logic) ...
    if model_lstm is None: raise ValueError("LSTM model not loaded.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Error opening video")
    frames, total_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1: raise ValueError("Empty video")
    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
            if not ret: frame = frames[-1]*255.0 if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), dtype=np.uint8)
            processed_frame = preprocess_frame(frame); frames.append(processed_frame)
    finally: cap.release()
    if len(frames) < SEQUENCE_LENGTH:
         frames.extend([frames[-1]] * (SEQUENCE_LENGTH - len(frames)))
    sequence_batch = np.expand_dims(np.array(frames[:SEQUENCE_LENGTH]), axis=0)
    prediction = model_lstm.predict(sequence_batch, verbose=0)[0][0]
    confidence_real = float(prediction); label = "Real" if confidence_real > 0.5 else "Fake"
    return label, confidence_real, None, None # Return 4 values (last 2 None)