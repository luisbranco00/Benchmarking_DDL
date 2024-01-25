import tensorflow as tf
import os
import json
import numpy as np
import logging
import sys


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

os.environ['TF_CONFIG'] = json.dumps({ 'cluster': { 'worker': ["cloud98:8050","cloud99:8052",'cloud103:8000',"cloud104:8000"]},
 'task': {'type': 'worker', 'index':int(sys.argv[1])}})

strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver())
data = np.load('breastmnist.npz')
pathmnist_train_images = data['train_images']
pathmnist_train_labels = data['train_labels']
pathmnist_test_images = data['test_images']
pathmnist_test_labels = data['test_labels']

# Train data
train_images = []
train_labels = []

# Process train images and labels
for i in range(len(pathmnist_train_images)):
    if pathmnist_train_images[i] is not None and pathmnist_train_labels[i] is not None:
        image = pathmnist_train_images[i]
        image = np.expand_dims(image, axis=-1)
        resized_image = tf.image.resize(image, (224, 224))
        resized_image = tf.image.grayscale_to_rgb(resized_image)
        normalized_image = resized_image / 255.0
        train_images.append(normalized_image)
        train_labels.append(np.array([pathmnist_train_labels[i]]))

train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_labels = tf.one_hot(train_labels, depth=1, dtype=tf.float32)
train_labels = tf.squeeze(train_labels, axis=1)
train_labels = tf.squeeze(train_labels, axis=1)

# Test data
test_images = []
test_labels = []

# Process test images and labels
for i in range(len(pathmnist_test_images)):
    if pathmnist_test_images[i] is not None and pathmnist_test_labels[i] is not None:
        image = pathmnist_test_images[i]
        image = np.expand_dims(image, axis=-1)
        resized_image = tf.image.resize(image, (224, 224))
        resized_image = tf.image.grayscale_to_rgb(resized_image)
        normalized_image = resized_image / 255.0
        test_images.append(normalized_image)
        test_labels.append(np.array([pathmnist_test_labels[i]]))

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_labels = tf.one_hot(test_labels, depth=1, dtype=tf.float32)
test_labels = tf.squeeze(test_labels, axis=1)
test_labels = tf.squeeze(test_labels, axis=1)

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32), tf.cast(train_labels, tf.int64)))
train_dataset = train_dataset.repeat().shuffle(len(train_images)).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(test_images, tf.float32), tf.cast(test_labels, tf.int64)))
test_dataset = test_dataset.repeat().shuffle(len(test_images)).batch(64)


print('Dataset preparado, A preparar o modelo--->')

def build_and_compile_cnn_model():
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,  # Exclude the original top layer
        weights=None,  # Do not load pre-trained weights
        input_shape=(224, 224, 3),  # Adjust input shape according to your data
        pooling='avg'  # Optional: Add pooling layer before the final classification layer
    )
    # Add a custom top layer for binary classification
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(base_model.output)  # Use sigmoid activation for binary classification
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', 'mse']
    )
    return model

    

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
	multi_worker_model = build_and_compile_cnn_model()
print('Iniciando a primeira epoca:')


# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(train_dataset, steps_per_epoch=(len(train_images)//64),epochs=10)
batch_size = 64
number_of_steps = len(pathmnist_test_images) // batch_size

# Evaluate the model on the test dataset with the specified number of steps
results = multi_worker_model.evaluate(test_dataset ,steps=number_of_steps)
