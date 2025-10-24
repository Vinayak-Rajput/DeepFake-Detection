# Save this as cnn_trainer_v3_regularized.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input,
    RandomBrightness, RandomContrast # Added for augmentation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 # Added for L2 regularization
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Ensure this points to the correct place (/content/data/processed_frames if on Colab)
DATA_DIR = "processed_frames"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Keep batch size reasonable, maybe even reduce to 16 if memory was tight

# --- 1. Prepare Data Generators (More Aggressive Augmentation) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30, # Increased range
    width_shift_range=0.25, # Increased range
    height_shift_range=0.25, # Increased range
    shear_range=0.25, # Increased range
    zoom_range=0.25, # Increased range
    horizontal_flip=True,
    brightness_range=[0.7, 1.3], # Added brightness augmentation
    fill_mode='nearest'
    # Consider adding RandomContrast layer in the model instead if needed
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
    subset='validation',
    shuffle=False # Typically don't shuffle validation data
)

print(f"Class Indices: {train_generator.class_indices}")

# --- 3. Build the Model (Reduced Complexity, Stronger Regularization) ---
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False # Keep base frozen for initial training

model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # Add augmentation layers directly into the model if preferred over ImageDataGenerator for some types
    # RandomBrightness(factor=0.2),
    # RandomContrast(factor=0.2),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, # Reduced units from 256
          activation='relu',
          kernel_regularizer=l2(0.001)), # Added L2 regularization
    Dropout(0.6), # Increased dropout rate from 0.5
    Dense(1, activation='sigmoid')
])

# --- 4. Compile the Model ---
optimizer = Adam(learning_rate=0.0001) # Keep learning rate low
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model Head ---
print("Starting model training (regularized head)...")
EPOCHS = 15 # Train a bit longer to see if regularization helps

# Add EarlyStopping to prevent training for too long if validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=5,         # Stop after 5 epochs with no improvement
    restore_best_weights=True # Restore weights from the best epoch
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
    callbacks=[early_stopping] # Add the early stopping callback
)

# --- 6. (Optional) Fine-tuning Block (Keep commented for now) ---
# Consider unfreezing ONLY after confirming the head training isn't overfitting severely
# print("Starting full model fine-tuning (unfreezing some layers)...")
# base_model.trainable = True
# FINE_TUNE_AT = 100 # Unfreeze from layer 100 onwards
# for layer in base_model.layers[:FINE_TUNE_AT]:
#    layer.trainable = False
#
# optimizer_fine = Adam(learning_rate=1e-5) # Very low learning rate
# model.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])
#
# history_fine = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
#     epochs=EPOCHS + 10,
#     initial_epoch=history.epoch[-1] + 1,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
#     callbacks=[early_stopping] # Can reuse early stopping
# )

# --- 7. Save the Model ---
model.save("xception_regularized_detector.keras")
print("Model training complete. Saved as xception_regularized_detector.keras")

# --- 8. Plot training history ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Add fine-tuning history if it ran
# if 'history_fine' in locals():
#     acc += history_fine.history['accuracy'][1:] # Skip first epoch overlap
#     val_acc += history_fine.history['val_accuracy'][1:]
#     loss += history_fine.history['loss'][1:]
#     val_loss += history_fine.history['val_loss'][1:]

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

plt.tight_layout() # Adjust layout
plt.savefig('training_history_xception_regularized.png')
print("Training history plot saved as training_history_xception_regularized.png")
# plt.show() # Uncomment if running locally and want to see the plot immediately