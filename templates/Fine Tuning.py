import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np # Import numpy
import cv2 # Import cv2

# --- Import the VideoSequenceGenerator class ---
# (Pasting the class here for a self-contained script)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array

class VideoSequenceGenerator(Sequence):
    """Generates batches of frame sequences from video files for CNN+LSTM training."""
    def __init__(self, video_files, labels, batch_size, sequence_length,
                 img_height, img_width, shuffle=True, n_channels=3):
        self.video_files = video_files
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_files))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.video_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_files = [self.video_files[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]
        X, y = self.__data_generation(batch_video_files, batch_labels)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.video_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_video_files, batch_labels):
        """Generates data containing batch_size sequences."""
        X = np.empty((self.batch_size, self.sequence_length, self.img_height, self.img_width, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, video_path in enumerate(batch_video_files):
            sequence = self._load_video_sequence(video_path)
            X[i,] = sequence
            y[i] = batch_labels[i]

        return X, np.array(y)

    def _load_video_sequence(self, video_path):
        """Loads, preprocesses, and samples frames for one video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)

        try:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frame = frames[-1] if frames else np.zeros((self.img_height, self.img_width, self.n_channels))
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                frame = img_to_array(frame) / 255.0 # Normalize
                frames.append(frame)
        finally:
            cap.release()
        
        if len(frames) < self.sequence_length:
            diff = self.sequence_length - len(frames)
            padding_frame = frames[-1] if frames else np.zeros((self.img_height, self.img_width, self.n_channels))
            frames.extend([padding_frame] * diff)
        elif len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]

        return np.array(frames)
# --- End of VideoSequenceGenerator ---


# --- Configuration ---
BASE_VIDEO_DIR = "Celeb-DF/videos" # Or "/content/drive/MyDrive/Celeb-DF/videos" on Colab
REAL_VIDEO_SUBDIR = "Celeb-real"
FAKE_VIDEO_SUBDIR = "Celeb-synthesis"
CORRUPT_FILE = "Celeb-DF/videos/Celeb-real/id27_0005.mp4" # Path to remove (relative to project root)

SEQUENCE_LENGTH = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8 # Keep batch size low for fine-tuning
EPOCHS = 10 # Number of *additional* epochs for fine-tuning
MODEL_TO_LOAD = "cnn_lstm_fake_detector_fully_trained.keras"
NEW_MODEL_NAME = "cnn_lstm_model_FINETUNED.keras"

# --- 1. Data Preparation (Same as before) ---
print("Preparing video file lists...")
real_video_dir = os.path.join(BASE_VIDEO_DIR, REAL_VIDEO_SUBDIR)
fake_video_dir = os.path.join(BASE_VIDEO_DIR, FAKE_VIDEO_SUBDIR)
real_files = [os.path.join(real_video_dir, f) for f in os.listdir(real_video_dir) if f.lower().endswith('.mp4')]
fake_files = [os.path.join(fake_video_dir, f) for f in os.listdir(fake_video_dir) if f.lower().endswith('.mp4')]
video_files = real_files + fake_files
labels = [1] * len(real_files) + [0] * len(fake_files)

# Remove the one known corrupt file
# Note: Adjust path separator for consistency if needed
corrupt_path_normalized = os.path.normpath(CORRUPT_FILE)
try:
    index_to_remove = -1
    # Find the file path regardless of OS separator
    for i, f in enumerate(video_files):
        if os.path.normpath(f) == corrupt_path_normalized:
            index_to_remove = i
            break
            
    if index_to_remove != -1:
        video_files.pop(index_to_remove)
        labels.pop(index_to_remove)
        print(f"Successfully removed corrupt file: {CORRUPT_FILE}")
    else:
        print(f"WARNING: Corrupt file not found in list: {CORRUPT_FILE}")
except ValueError:
    print(f"WARNING: Corrupt file not found in list: {CORRUPT_FILE}")

# Split into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    video_files, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Total videos: {len(video_files)}")
print(f"Training videos: {len(train_files)}, Validation videos: {len(val_files)}")

# --- 2. Create Data Generators ---
print("Creating data generators...")
train_generator = VideoSequenceGenerator(
    video_files=train_files, labels=train_labels, batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH, img_height=IMG_HEIGHT, img_width=IMG_WIDTH
)
val_generator = VideoSequenceGenerator(
    video_files=val_files, labels=val_labels, batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, shuffle=False
)

# --- 3. Load the Existing Model ---
print(f"Loading existing model from {MODEL_TO_LOAD}...")
if not os.path.exists(MODEL_TO_LOAD):
    print(f"ERROR: Model file not found: {MODEL_TO_LOAD}")
    print("Please run the main training script first.")
    exit()

# We need to define the model architecture to load weights properly
# (Using the same method as predictor.py)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.applications import Xception

def build_cnn_lstm_model():
    base_model_cnn = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
    base_model_cnn.trainable = False # Will be changed later
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)),
        TimeDistributed(base_model_cnn, name="time_distributed_xception"), # Give the base a name
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="cnn_lstm_model")
    return model

model = build_cnn_lstm_model()
# Build the model by passing dummy data
dummy_input = np.zeros((1, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3))
_ = model(dummy_input, training=False) 
# Load the weights from our last training run
model.load_weights(MODEL_TO_LOAD)
print("Model and weights loaded successfully.")


# --- 4. Fine-Tuning Setup ---
print("Setting model for fine-tuning...")
# Find the Xception base model within our TimeDistributed layer
base_cnn = model.get_layer('time_distributed_xception').layer
base_cnn.trainable = True # Unfreeze the base

# How many layers to freeze from the bottom? Xception has ~132 layers
# Let's freeze the first 100 layers (the early feature extractors)
# and only fine-tune the top 32 (the more complex feature extractors)
FINE_TUNE_AT = 100 
for layer in base_cnn.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"Unfrozen Xception base model. First {FINE_TUNE_AT} layers remain frozen.")

# --- 5. Re-Compile with a Very Low Learning Rate ---
optimizer = Adam(learning_rate=1e-5) # Use 0.00001
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary() # Print the summary to see trainable/non-trainable params

# --- 6. Train (Fine-Tune) the Model ---
print("Starting model fine-tuning...")

# Add EarlyStopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, # Stop if val_loss doesn't improve for 5 epochs
    restore_best_weights=True # Keep the best model
)
# Add ModelCheckpoint to save the best model found during fine-tuning
checkpoint = callbacks.ModelCheckpoint(
    NEW_MODEL_NAME, # Save path
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint] # Use both callbacks
)

print(f"Model fine-tuning complete. Best model saved to {NEW_MODEL_NAME}")

# --- 7. Plot training history ---
print("Generating fine-tuning history plot...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Fine-Tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Fine-Tuning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plot_save_path = 'training_history_fine_tune.png'
plt.savefig(plot_save_path)
print(f"Training history plot saved as {plot_save_path}")
print("Script finished.")