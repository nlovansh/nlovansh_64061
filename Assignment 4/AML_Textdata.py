#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:28:53 2025

@author: nidhi
"""

#  Download and extract IMDB dataset
import os
if not os.path.exists("aclImdb"):
    print("**** Downloading and extracting IMDB dataset ****")
    os.system("curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
    os.system("tar -xf aclImdb_v1.tar.gz")
    os.system("rm -r aclImdb/train/unsup") # Remove unsupervised data
else:
    print("IMDB dataset already downloaded.")

# Create validation set if not already done
import pathlib, shutil, random
from tensorflow import keras

batch_size = 32
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

if not val_dir.exists():
    print("**** Creating validation set ****")
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category, exist_ok=True)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            src = train_dir / category / fname
            dst = val_dir / category / fname
            if src.exists():
                shutil.move(src, dst)
    print("Validation split completed.")
else:
    print("Validation set already exists. Skipping split.")

# Load training, validation, and test datasets
train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)
text_only_train_ds = train_ds.map(lambda x, y: x) # For adapting the vectorizer

# Defining parameters
import numpy as np
from tensorflow.keras import layers

max_tokens = 10000
max_length = 150  # Cut off each review after 150 words
embedding_dim = 100  # Dimension for word embeddings
training_sizes = [100, 500, 1000, 5000, 25000] # Different training set sizes to evaluate

# Text vectorization
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length
)
text_vectorization.adapt(text_only_train_ds)

#Function to apply vectorization
def vectorize(ds): return ds.map(lambda x, y: (text_vectorization(x), y))

# Vectorize datasets
int_train_ds_full = vectorize(train_ds)
int_val_ds_full = vectorize(val_ds)
int_test_ds = vectorize(test_ds)

# Take exactly 10,000 samples for validation
val_subset = int_val_ds_full.unbatch().take(10000).batch(batch_size)

# Load GloVe embeddings
print("**** Loading GloVe embeddings ****")
embedding_index = {}
glove_path = "glove.6B.100d.txt"

if not os.path.exists(glove_path):
    os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
    os.system("unzip -q glove.6B.zip")

with open(glove_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        embedding_index[word] = np.fromstring(coefs, "f", sep=" ")

vocab = text_vectorization.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))
embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

# Build model function
def build_model(use_glove=False):
    if use_glove:
        embedding_layer = layers.Embedding(
            input_dim=max_tokens,
            output_dim=embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
            mask_zero=True
        )
    else:
        embedding_layer = layers.Embedding(
            input_dim=max_tokens,
            output_dim=embedding_dim,
            mask_zero=True
        )

    inputs = keras.Input(shape=(None,), dtype="int64")
    x = embedding_layer(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Evaluation function
import tensorflow as tf

def evaluate_models(training_sizes, batch_size=32):
    results = []

    for size in training_sizes:
        print(f"\n Training size: {size}")

        train_subset = int_train_ds_full.unbatch().take(size).batch(batch_size)

        try:
            # Simple Embedding
            model_simple = build_model(use_glove=False)
            model_simple.fit(train_subset, validation_data=val_subset, epochs=10, verbose=0)
            acc_simple = model_simple.evaluate(int_test_ds, verbose=0)[1]

            # GloVe Embedding
            model_glove = build_model(use_glove=True)
            model_glove.fit(train_subset, validation_data=val_subset, epochs=10, verbose=0)
            acc_glove = model_glove.evaluate(int_test_ds, verbose=0)[1]

            print(f" Accuracy - Simple: {acc_simple:.3f}, GloVe: {acc_glove:.3f}")
            better = "GloVe" if acc_glove > acc_simple else "Simple"
            print(f" {better} embedding performed better for size {size}")
            results.append((size, acc_simple, acc_glove))

        except Exception as e:
            print(f" Error at size {size}: {str(e)}")
            continue

    # Final summary
    print("\n Performance Summary:")
    for size, acc_simple, acc_glove in results:
        better = "GloVe" if acc_glove > acc_simple else "Simple"
        print(f"Size {size}: Simple = {acc_simple:.3f}, GloVe = {acc_glove:.3f}  Better: {better}")

    return results

#  Run experiment
results = evaluate_models(training_sizes)



import matplotlib.pyplot as plt
training_sizes, acc_simple_list, acc_glove_list = zip(*results)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, acc_simple_list, marker='o', label='Simple Embedding')
plt.plot(training_sizes, acc_glove_list, marker='o', label='GloVe Embedding')
plt.title('Performance Comparison: Simple vs. GloVe Embedding')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
