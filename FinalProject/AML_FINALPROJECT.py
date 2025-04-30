#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:27:57 2025

@author: shivam
"""

#Loading libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight

# Set paths to the dataset and images
root_path = '/Users/shivam/Downloads/archive'
img_dir_a = os.path.join(root_path, 'HAM10000_images_part_1')
img_dir_b = os.path.join(root_path, 'HAM10000_images_part_2')
meta_file = os.path.join(root_path, 'HAM10000_metadata.csv')

# Organized subset paths
clean_dir = '/Users/shivam/Downloads/HAM10000_subset'
output_img_path = os.path.join(clean_dir, 'images')

# Create directories for 'benign' and 'malignant' classes
os.makedirs(os.path.join(output_img_path, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_img_path, 'malignant'), exist_ok=True)

# Load metadata
meta_df = pd.read_csv(meta_file)
meta_df['image_id'] = meta_df['image_id'] + '.jpg'

# Map original diagnosis labels into binary 'benign' and 'malignant' classes

malignant_classes = ['mel', 'bcc', 'akiec']
meta_df['class'] = meta_df['dx'].apply(lambda dx: 'malignant' if dx in malignant_classes else 'benign')

# Copy images into their respective class folders
for _, rec in meta_df.iterrows():
    filename = rec['image_id']
    category = rec['class']
    source_path = os.path.join(img_dir_a, filename) if os.path.exists(os.path.join(img_dir_a, filename)) else os.path.join(img_dir_b, filename)
    if os.path.exists(source_path):
        destination = os.path.join(output_img_path, category, filename)
        shutil.copy(source_path, destination)

print("Image dataset prepared.")

# Define image properties and batch size
IMG_H, IMG_W = 224, 224
BS = 32

# Create ImageDataGenerator for augmentation and validation split
image_flow = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = image_flow.flow_from_directory(
    output_img_path,
    target_size=(IMG_H, IMG_W),
    batch_size=BS,
    class_mode='binary',
    subset='training'
)

valid_gen = image_flow.flow_from_directory(
    output_img_path,
    target_size=(IMG_H, IMG_W),
    batch_size=BS,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Compute class weights to address class imbalance
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
weights = dict(enumerate(weights))
print("Balanced class weights:", weights)

# Define callbacks for training
train_hooks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
]

# ---------------------- Model 1: Basic CNN ----------------------

# Define a simple CNN architecture
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_H, IMG_W, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


#Compile the CNN model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nFitting Basic CNN...\n")

#Train the CNN model
cnn_hist = cnn_model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=train_hooks,
    class_weight=weights
)

# ---------------------- Model 2: MobileNetV2 ----------------------

# Load the MobileNetV2 base model (pre-trained on ImageNet)

mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, 3))
mobilenet_base.trainable = False

# Add custom classifier layers on top of MobileNetV2
mobilenet_net = Sequential([
    mobilenet_base,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile MobileNetV2 model
mobilenet_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nFitting MobileNetV2...\n")

#Train the model
mobilenet_hist = mobilenet_net.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=train_hooks,
    class_weight=weights
)

# Fine-tune the MobileNetV2 model
mobilenet_base.trainable = True
mobilenet_net.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
mobilenet_net.fit(
    train_gen,
    epochs=5,
    validation_data=valid_gen,
    callbacks=train_hooks,
    class_weight=weights
)

# ---------------------- Model 3: ResNet50 ----------------------

# Load the ResNet50 base model (pre-trained on ImageNet)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, 3))
resnet_base.trainable = False

# Add custom classifier layers on top of ResNet50
resnet_net = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the ResNet50 model with Adam optimizer and binary crossentropy loss function
resnet_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nFitting ResNet50...\n")
resnet_hist = resnet_net.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=train_hooks,
    class_weight=weights
)

resnet_base.trainable = True
resnet_net.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the ResNet50 model
resnet_net.fit(
    train_gen,
    epochs=5,
    validation_data=valid_gen,
    callbacks=train_hooks,
    class_weight=weights
)

## ---------------------- Plotting Accuracy ----------------------

# Store the results of all models (Basic CNN, MobileNetV2, ResNet50) for comparison

results = {
    'Basic CNN': (cnn_model, cnn_hist),
    'MobileNetV2': (mobilenet_net, mobilenet_hist),
    'ResNet50': (resnet_net, resnet_hist)
}

# Plot validation accuracy of all models across epochs


plt.figure(figsize=(12, 6))
for label, (mdl, hist) in results.items():
    plt.plot(hist.history['val_accuracy'], label=f'{label} Val Acc')
plt.title('Validation Accuracy Across Models')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ---------------------- Evaluation ----------------------

# Evaluate the models and print classification reports and confusion matrices

for label, (mdl, hist) in results.items():
    print(f"\n{label} - Performance:")
    predictions = mdl.predict(valid_gen)
    bin_preds = (predictions > 0.5).astype(int).reshape(-1)
    actuals = valid_gen.classes
    print(classification_report(actuals, bin_preds, target_names=['Benign', 'Malignant']))
    print(confusion_matrix(actuals, bin_preds))

# ---------------------- Confusion Matrix Plot ----------------------

# Function to visualize confusion matrix using a heatmap
def visualize_conf_matrix(y_real, y_pred, model_label):
    mat = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title(f'{model_label} - Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

for label, (mdl, hist) in results.items():
    preds = mdl.predict(valid_gen)
    preds_binary = (preds > 0.5).astype(int).reshape(-1)
    visualize_conf_matrix(valid_gen.classes, preds_binary, label)

# ---------------------- ROC Curve Plot ----------------------

# Function to plot the ROC curve for model performance evaluation

def draw_roc(y_real, probs, label_name):
    fpr, tpr, _ = roc_curve(y_real, probs)
    score = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{label_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

for label, (mdl, hist) in results.items():
    proba = mdl.predict(valid_gen).ravel()
    draw_roc(valid_gen.classes, proba, label)
