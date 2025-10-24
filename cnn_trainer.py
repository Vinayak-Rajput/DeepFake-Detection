# Save this as cnn_trainer.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- Configuration ---
DATA_DIR = "processed_frames"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# --- 1. Prepare Data Generators ---
# Use augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for the validation set
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # Must be same as above
)

# --- 2. Load Data from Directories ---
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training' # Set as training data
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation' # Set as validation data
)

# Check class indices (e.g., {'fake': 0, 'real': 1})
print(f"Class Indices: {train_generator.class_indices}")
# We need to remember this for our app!
# If 'real' is 1, a high output (near 1) means "Real".
# If 'real' is 0, a high output (near 1) means "Fake".
# Let's assume {'fake': 0, 'real': 1} for this build.

# --- 3. Build the Model ---
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Classifier Head
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Add dropout to prevent overfitting
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Train the Model ---
print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10, # Use 10-15 epochs for a good start
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Save the Model ---
model.save("cnn_fake_detector.h5")
print("Model training complete. Saved as cnn_fake_detector.h5")