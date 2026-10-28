# Custom Convolutional Neural Network (CNN) for Plant Disease Detection
# End-to-end multi-layer feature extraction and classification

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset configuration
DATASET_DIR = "PlantLeave Dataset/plantvillage dataset/color"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# Load training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Save class names for UI and prediction
class_names = train_ds.class_names
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

num_classes = len(class_names)
print(f"Total Classes: {num_classes}")

# Build Custom CNN Architecture
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Rescaling(1.0 / 255)(inputs)

# Convolutional and Pooling Blocks
x = layers.Conv2D(32, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)

# Last convolutional layer with explicit name for Grad-CAM
x = layers.Conv2D(128, (3, 3), activation="relu", name="last_conv")(x)
x = layers.MaxPooling2D(2, 2)(x)

# Classification Dense Head
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
print("Starting Custom CNN Training...")
model.fit(
    train_ds.take(150),
    validation_data=val_ds.take(30),
    epochs=EPOCHS
)

# Save trained model
model.save("custom_cnn_model.keras")
print("Custom CNN model saved successfully as custom_cnn_model.keras")
