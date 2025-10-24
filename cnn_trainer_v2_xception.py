# Save this as cnn_trainer_v2_xception.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception # Import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATA_DIR = "processed_frames" # Or "/content/data/processed_frames" if using Colab
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Keep batch size reasonable for GPU memory

# --- 1. Prepare Data Generators (Same as before, with augmentation) ---
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale pixel values
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# --- 2. Load Data from Directories ---
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Check class indices (Important!)
print(f"Class Indices: {train_generator.class_indices}")
# We're still assuming {'fake': 0, 'real': 1}

# --- 3. Build the Model using Xception ---

# Load the pre-trained Xception base, excluding the final classification layer
base_model = Xception(
    weights='imagenet', # Load weights pre-trained on ImageNet
    include_top=False, # DON'T include the final Dense layer
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze the layers of the base model (we don't want to retrain Xception yet)
base_model.trainable = False

# Create the new model on top
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Define the input shape explicitly
    base_model,
    GlobalAveragePooling2D(), # Reduces spatial dimensions
    Dense(256, activation='relu'), # Add a dense layer for classification
    Dropout(0.5), # Add dropout for regularization
    Dense(1, activation='sigmoid') # Final output layer for binary classification
])

# --- 4. Compile the Model ---
# Use a lower learning rate for fine-tuning
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model (only the new layers) ---
print("Starting model fine-tuning (training new head)...")
EPOCHS = 10 # Train the head for a few epochs first

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 6. (Optional) Fine-tune: Unfreeze some base layers ---
# After the head is trained, you can optionally unfreeze the top layers
# of the base model and train with a very low learning rate.
# print("Starting full model fine-tuning (unfreezing some layers)...")
# base_model.trainable = True
# # Freeze layers up to a certain point (e.g., keep the first 100 layers frozen)
# for layer in base_model.layers[:100]:
#    layer.trainable = False

# optimizer = Adam(learning_rate=1e-5) # Use a very low learning rate
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# history_fine = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS + 5, # Train for a few more epochs
#     initial_epoch=history.epoch[-1], # Continue from where we left off
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE
# )

# --- 7. Save the Model ---
# Save in the recommended .keras format
model.save("xception_fake_detector.keras")
print("Model training complete. Saved as xception_fake_detector.keras")

# --- 8. (Optional) Plot training history ---
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Add fine-tuning history if you ran that step
# if 'history_fine' in locals():
#     acc += history_fine.history['accuracy']
#     val_acc += history_fine.history['val_accuracy']
#     loss += history_fine.history['loss']
#     val_loss += history_fine.history['val_loss']

epochs_range = range(len(acc)) # Adjust if fine-tuning

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history_xception.png') # Save the plot
plt.show()