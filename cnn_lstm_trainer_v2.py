# Save this as cnn_lstm_trainer.py
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, TimeDistributed, LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence # Needed for custom data generator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2 # OpenCV for video processing
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust BASE_VIDEO_DIR if your Celeb-DF folder is elsewhere
BASE_VIDEO_DIR = "/content/drive/MyDrive/Celeb-DF/videos"
REAL_VIDEO_SUBDIR = "Celeb-real"
FAKE_VIDEO_SUBDIR = "Celeb-synthesis"

SEQUENCE_LENGTH = 10    # Number of frames per sequence (time steps)
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8          # Reduced batch size for LSTM memory usage
EPOCHS = 15             # Number of epochs to train
N_CHANNELS = 3

# --- 1. VideoSequenceGenerator Class Definition ---
class VideoSequenceGenerator(Sequence):
    """
    Generates batches of frame sequences from video files for CNN+LSTM training.
    """
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

        # Pad if video is shorter than sequence_length
        if len(frames) < self.sequence_length:
            diff = self.sequence_length - len(frames)
            padding_frame = frames[-1] if frames else np.zeros((self.img_height, self.img_width, self.n_channels))
            frames.extend([padding_frame] * diff)
        # Truncate if somehow longer (shouldn't happen with linspace)
        elif len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]

        return np.array(frames)

# --- 2. Data Preparation ---
print("Preparing video file lists...")
real_video_dir = os.path.join(BASE_VIDEO_DIR, REAL_VIDEO_SUBDIR)
fake_video_dir = os.path.join(BASE_VIDEO_DIR, FAKE_VIDEO_SUBDIR)

real_files = [os.path.join(real_video_dir, f) for f in os.listdir(real_video_dir) if f.lower().endswith('.mp4')]
fake_files = [os.path.join(fake_video_dir, f) for f in os.listdir(fake_video_dir) if f.lower().endswith('.mp4')]

# LIMIT DATA FOR QUICK TESTING (Optional - Comment out for full training)
#print("WARNING: Limiting data for quick testing! Comment out this section for full training.")
#real_files = real_files[:50]  Use only 50 real videos
#fake_files = fake_files[:50]  Use only 50 fake videos
# --- End of Limiting Data ---

video_files = real_files + fake_files
labels = [1] * len(real_files) + [0] * len(fake_files) # 1 for real, 0 for fake

# --- ADDED THIS BLOCK TO REMOVE CORRUPT FILES ---
corrupt_file_path = "/content/drive/MyDrive/Celeb-DF/videos/Celeb-real/id27_0005.mp4" # Path from the log
print(f"Attempting to remove problematic file: {corrupt_file_path}")

try:
    index_to_remove = video_files.index(corrupt_file_path)
    video_files.pop(index_to_remove)
    labels.pop(index_to_remove)
    print(f"Successfully removed {corrupt_file_path} from the dataset lists.")
except ValueError:
    print(f"WARNING: Problematic file {corrupt_file_path} not found in the initial list. Skipping removal.")
# --- END OF BLOCK TO REMOVE ---

# Check if video files exist (simple check for the first few)
if not video_files:
    raise ValueError(f"No video files found in {BASE_VIDEO_DIR}. Check paths.")
for vf in video_files[:5]:
    if not os.path.exists(vf):
        print(f"WARNING: Video file not found: {vf}")


# Split into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    video_files, labels, test_size=0.2, random_state=42, stratify=labels # Stratify helps keep class balance
)
print(f"Total videos: {len(video_files)}")
print(f"Training videos: {len(train_files)}")
print(f"Validation videos: {len(val_files)}")


# --- 3. Create Data Generators ---
print("Creating data generators...")
train_generator = VideoSequenceGenerator(
    video_files=train_files,
    labels=train_labels,
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    n_channels=N_CHANNELS,
    shuffle=True
)

val_generator = VideoSequenceGenerator(
    video_files=val_files,
    labels=val_labels,
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    n_channels=N_CHANNELS,
    shuffle=False # No need to shuffle validation data
)

# --- 4. Build the CNN + LSTM Model ---
print("Building the CNN+LSTM model...")
# Load the pre-trained Xception base, including Global Average Pooling
base_model_cnn = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS),
    pooling='avg' # Add pooling directly here
)
base_model_cnn.trainable = False # Freeze the base model

# Build the sequential model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)),
    TimeDistributed(base_model_cnn), # Apply CNN to each frame
    LSTM(128, return_sequences=False), # Process sequence features
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Final classification
])

# --- 5. Compile the Model ---
optimizer = Adam(learning_rate=0.0001) # Low learning rate for fine-tuning heads
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 6. Train the Model ---
print("Starting CNN+LSTM model training...")

# Add EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, # Stop if val_loss doesn't improve for 5 epochs
    restore_best_weights=True # Keep the best model weights found
)

# Fit using the generators
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# --- 7. Save the Model ---
model_save_path = "cnn_lstm_fake_detector_fully_trained_v2.keras"
model.save(model_save_path)
print(f"Model training complete. Saved as {model_save_path}")

# --- 8. Plot training history ---
print("Generating training history plot...")
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
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plot_save_path = 'training_history_cnn_lstm_fully_trained_v2.png'
plt.savefig(plot_save_path)
print(f"Training history plot saved as {plot_save_path}")
# plt.show() # Uncomment to display plot if running locally

print("Script finished.")