#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:42:46 2025

@author: nidhi
"""

import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


base_directory = pathlib.Path("/Users/nidhi/Downloads/cats_vs_dogs_small")  

#Create dataset from directory
def create_datasets(train_size, base_dir, image_size=(180, 180), batch_size=32):
    """
    Creates training, validation, and test datasets from image directories.
    Returns:
    Datasets for training, validation, and testing
    """
    base_dir = pathlib.Path(base_dir)  

    # Load full datasets from directory
    full_train = keras.utils.image_dataset_from_directory(
        base_dir / "train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    full_val = keras.utils.image_dataset_from_directory(
        base_dir / "validation",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    full_test = keras.utils.image_dataset_from_directory(
        base_dir / "test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Calculate split sizes
    train_batches = train_size // batch_size
    val_batches = int(0.2 * train_batches)  # Use 20% of train for validation
    
    # Subset datasets to required sizes
    train_dataset = full_train.take(train_batches)
    val_dataset = full_val.take(val_batches)
    test_dataset = full_test  

    return train_dataset, val_dataset, test_dataset


# Data Augmentation

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])


# Train Model from Scratch

def train_from_scratch(train_size=1000, epochs=30):
    """
    Trains a model from scratch.
    Returns:
    Test accuracy of the trained model
    """
    train_ds, val_ds, test_ds = create_datasets(train_size, base_directory)

    # Define CNN Model
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)  #  dropout for regularization
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=f"scratch_model_{train_size}.keras",
                                        save_best_only=True,
                                        monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    # Evaluate Model
    test_model = keras.models.load_model(f"scratch_model_{train_size}.keras")
    test_loss, test_acc = test_model.evaluate(test_ds)
    print(f"Test Accuracy (Scratch, {train_size} samples): {test_acc:.3f}")

    # Plot Training History
    plot_history(history)
    
    return test_acc


# Train Model with Pretrained VGG16

def train_pretrained(train_size=1000, epochs=20):
    """
    Train Model with Pretrained VGG16
    Returns:
    Test accuracy of the trained model
    """
    train_ds, val_ds, test_ds = create_datasets(train_size, base_directory)

    # Load VGG16 Pretrained Model
    conv_base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))
    conv_base.trainable = False  # Freeze layers

    # Define Model on Top of VGG16
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)  # dropout for regularization
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=f"pretrained_model_{train_size}.keras",
                                        save_best_only=True,
                                        monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    # Evaluate Model
    test_model = keras.models.load_model(f"pretrained_model_{train_size}.keras")
    test_loss, test_acc = test_model.evaluate(test_ds)
    print(f"Test Accuracy (Pretrained, {train_size} samples): {test_acc:.3f}")

    # Plot Training History
    plot_history(history)

    return test_acc


def plot_history(history):
    """
    Plots training and validation accuracy and loss.
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()

#Running with different training sample sizes
print("\n*** Training from Scratch (1000 Samples) ***")
acc_scratch_1000 = train_from_scratch(train_size=1000)

print("\n*** Training from Scratch (2000 Samples) ***")
acc_scratch_2000 = train_from_scratch(train_size=2000)

print("\n*** Training from Scratch (2500 Samples) ***")
acc_scratch_2500 = train_from_scratch(train_size=2500)

print("\n*** Training with Pretrained VGG16 (1000 Samples) ***")
acc_pretrained_1000 = train_pretrained(train_size=1000)

print("\n*** Training with Pretrained VGG16 (1500 Samples) ***")
acc_pretrained_1500 = train_pretrained(train_size=1500)


# Accuracy 
print("\n*** Accuracy ***")
print(f"From Scratch (1000 Samples): {acc_scratch_1000:.3f}")
print(f"From Scratch (2000 Samples): {acc_scratch_2000:.3f}")
print(f"From Scratch (2500 Samples): {acc_scratch_2500:.3f}")
print(f"Pretrained VGG16 (1000 Samples): {acc_pretrained_1000:.3f}")
print(f"Pretrained VGG16 (1500 Samples): {acc_pretrained_1500:.3f}")
