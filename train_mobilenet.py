# Transfer Learning with MobileNetV2 for Plant Disease Detection
# Pre-trained ImageNet Feature Extraction and Fine-Tuning

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

num_classes = len(train_ds.class_names)
print(f"Total Classes: {num_classes}")

# Load Pre-trained MobileNetV2 (Freeze weights for Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Connect MobileNetV2 features to classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
print("Starting MobileNetV2 Transfer Learning Training...")
model.fit(
    train_ds.take(150),
    validation_data=val_ds.take(30),
    epochs=EPOCHS
)

# Save trained model
model.save("mobilenet_model.keras")
print("MobileNetV2 model saved successfully as mobilenet_model.keras")
