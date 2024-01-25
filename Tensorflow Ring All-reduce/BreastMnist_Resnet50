import tensorflow as tf
import os
import json
import numpy as np
import logging
import sys


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

os.environ['TF_CONFIG'] = json.dumps({ 'cluster': { 'worker': ["Hostname:port","Hostname:port",'Hostname:port',"Hostname:port"]},
 'task': {'type': 'worker', 'index':int(sys.argv[1])}})

strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver())
num_workers = 4

# Define the batch size for training and validation
batch_size = 16

global_batch_size = batch_size * num_workers

# Load BreastMNIST data from .npz file
data = np.load('breastmnist.npz')
pathmnist_train_images = data['train_images']
pathmnist_train_labels = data['train_labels']
pathmnist_test_images = data['test_images']
pathmnist_test_labels = data['test_labels']

pathmnist_labels = []
pathmnist_images = []
pathmnist_labels_test = []
pathmnist_images_test = []

# Process train images and labels
for i in range(len(pathmnist_train_images)):
    if pathmnist_train_images[i] is not None and pathmnist_train_labels[i] is not None:
        image = pathmnist_train_images[i]  # Get image
        image = np.expand_dims(image, axis=-1)  # Add extra dimension to make it 3-dimensional
        resized_image = tf.image.resize(image, (224, 224))  # Resize image
        resized_image = tf.image.grayscale_to_rgb(resized_image)  # Convert grayscale to RGB
        normalized_image = resized_image / 255.0  # Normalize image
        pathmnist_images.append(normalized_image)  # Add resized and normalized image to the list
        pathmnist_labels.append(np.array([pathmnist_train_labels[i]]))  # Convert label to numpy array

# Process test images and labels
for i in range(len(pathmnist_test_images)):
    if pathmnist_test_images[i] is not None and pathmnist_test_labels[i] is not None:
        image = pathmnist_test_images[i]  # Get image
        image = np.expand_dims(image, axis=-1)  # Add extra dimension to make it 3-dimensional
        resized_image = tf.image.resize(image, (224, 224))  # Resize image
        resized_image = tf.image.grayscale_to_rgb(resized_image)  # Convert grayscale to RGB
        normalized_image = resized_image / 255.0  # Normalize image
        pathmnist_images_test.append(normalized_image)  # Add resized and normalized image to the list
        pathmnist_labels_test.append(np.array([pathmnist_test_labels[i]]))  # Convert label to numpy array

# Convert lists to numpy arrays
pathmnist_images = np.array(pathmnist_images)
pathmnist_labels = np.array(pathmnist_labels)

pathmnist_images_test = np.array(pathmnist_images_test)
pathmnist_labels_test = np.array(pathmnist_labels_test)

# One-hot encode labels
pathmnist_labels = tf.one_hot(pathmnist_labels, depth=1, dtype=tf.float32)
pathmnist_labels_test = tf.one_hot(pathmnist_labels_test, depth=1, dtype=tf.float32)

# Remove the last two dimensions from the labels
pathmnist_labels = tf.squeeze(pathmnist_labels, axis=1)
pathmnist_labels = tf.squeeze(pathmnist_labels, axis=1)

pathmnist_labels_test = tf.squeeze(pathmnist_labels_test, axis=1)
pathmnist_labels_test = tf.squeeze(pathmnist_labels_test, axis=1)

# Convert labels to binary format (0 or 1)

dataset = tf.data.Dataset.from_tensor_slices((tf.cast(pathmnist_images, tf.float32), tf.cast(pathmnist_labels, tf.int64)))
dataset = dataset.repeat().shuffle(len(pathmnist_images)).batch(64)

datasettest = tf.data.Dataset.from_tensor_slices((tf.cast(pathmnist_images_test, tf.float32), tf.cast(pathmnist_labels_test, tf.float32)))
datasettest = dataset.shuffle(len(pathmnist_images)).batch(64)


def build_and_compile_cnn_model():
    model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,  # Exclude the original top layer
        weights=None,  # Do not load pre-trained weights
        input_shape=(224,224,3),  # Adjust input shape according to your data
        pooling='avg'  # Optional: Add pooling layer before the final classification layer
    )
    # Add a custom top layer for binary classification
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(model.output)  # Use sigmoid activation for binary classification
    model = tf.keras.Model(inputs=model.input, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', 'mse']
    )
    return model

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()
  multi_worker_model.fit(dataset, epochs=10, steps_per_epoch=len(pathmnist_train_images)//64)
batch_size = 64
number_of_steps = len(pathmnist_test_images) // batch_size

# Evaluate the model on the test dataset with the specified number of steps
results = multi_worker_model.evaluate(x=pathmnist_images_test,y=pathmnist_labels_test, steps=number_of_steps)



