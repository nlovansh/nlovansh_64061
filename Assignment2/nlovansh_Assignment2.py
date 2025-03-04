# -*- coding: utf-8 -*-
"""
Spyder Editor

Assignment 2 - Hyperparameter Tuning for IMDB
"""
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam


# Load the IMDB dataset
max_features = 10000  # Vocabulary size
maxlen = 500  # Limit review length
batch_size = 32

# Load the training and test data from IMDB

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Split the training data into training and validation sets
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# Define the model with an embedding layer
model_1_layer = models.Sequential([
    layers.Embedding(max_features, 32, input_length=maxlen),  # Added Embedding layer
    layers.Flatten(),
    layers.Dense(16, activation="relu"),  # Hidden layer
    layers.Dense(1, activation="sigmoid")  # Output layer
])

# Compile the model with Adam optimizer and binary crossentropy loss
model_1_layer.compile(optimizer="adam",  # Changed optimizer to Adam
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

# Train the model
history_1_layer = model_1_layer.fit(partial_x_train,
                                     partial_y_train,
                                     epochs=30,  # Increased epochs
                                     batch_size=64,  # Reduced batch size
                                     validation_data=(x_val, y_val),
                                     verbose=1)

# Evaluate the model
results_1_layer = model_1_layer.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (1 layer with embedding): {results_1_layer[1]}")

# Plot training and validation loss
history_dict = history_1_layer.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss (1 Layer with Embedding)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy (1 Layer with Embedding)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#2. Varying the Number of Hidden Units


# Experiment with different numbers of hidden units

# 32 units in the hidden layers

model_32_units = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(500,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model_32_units.compile(optimizer="rmsprop",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

history_32_units = model_32_units.fit(partial_x_train,
                                     partial_y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_val, y_val),
                                     verbose=0)
# Evaluate the model with 32 units

results_32_units = model_32_units.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (32 units): {results_32_units[1]}")


# 64 units in the hidden layers
model_64_units = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(500,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile and train the model


model_64_units.compile(optimizer="rmsprop",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

history_64_units = model_64_units.fit(partial_x_train,
                                     partial_y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_val, y_val),
                                     verbose=0)

# Evaluate the model with 64 units

results_64_units = model_64_units.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (64 units): {results_64_units[1]}")


#3. Using mse Loss Function

model_mse = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(500,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid") #Keep Sigmoid even with MSE
])

# Compile and train the model with MSE loss

model_mse.compile(optimizer="rmsprop",
                  loss="mse",  # Mean Squared Error
                  metrics=["accuracy"])

history_mse = model_mse.fit(partial_x_train,
                                 partial_y_train,
                                 epochs=20,
                                 batch_size=512,
                                 validation_data=(x_val, y_val),
                                 verbose=0)

# Evaluate the model with MSE loss

results_mse = model_mse.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (MSE loss): {results_mse[1]}")

#4. Using tanh Activation

model_tanh = keras.Sequential([
    layers.Dense(16, activation="tanh", input_shape=(500,)),
    layers.Dense(16, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])

# Compile and train the model with tanh activation

model_tanh.compile(optimizer="rmsprop",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

history_tanh = model_tanh.fit(partial_x_train,
                                  partial_y_train,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(x_val, y_val),
                                  verbose=0)

# Evaluate the model with tanh activation

results_tanh = model_tanh.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (tanh activation): {results_tanh[1]}")

#5. Improving Performance with Regularization and Dropout



model_regularized = keras.Sequential([
    layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(500,)), #L2 Regularization
    layers.Dropout(0.5), #Dropout
    layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile and train the model with regularization and dropout

model_regularized.compile(optimizer="rmsprop",
                          loss="binary_crossentropy",
                          metrics=["accuracy"])

history_regularized = model_regularized.fit(partial_x_train,
                                               partial_y_train,
                                               epochs=20,
                                               batch_size=512,
                                               validation_data=(x_val, y_val),
                                               verbose=0)

# Evaluate the model with regularization and dropout

results_regularized = model_regularized.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy (Regularized): {results_regularized[1]}")

