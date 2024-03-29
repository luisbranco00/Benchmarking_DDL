import tensorflow as tf
import os
import json
import numpy as np
import logging
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["Hostname:port", "Hostname:port", 'Hostname:port', "Hostname:port"]
    },
    'task': {'type': 'worker', 'index': int(sys.argv[1])}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver()
)
num_workers = 4


def load_data(file_path):
    data = np.load(file_path)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']


def process_image(image, label):
    if image is not None and label is not None:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        if len(image.shape) < 3:
            image = tf.expand_dims(image, axis=-1)
            image = tf.image.grayscale_to_rgb(image)
        resized_image = tf.image.resize(image, (224, 224))
        normalized_image = resized_image / 255.0

        # Convert labels to integers instead of one-hot encoding
        label = tf.convert_to_tensor(label, dtype=tf.int64)
        return normalized_image, label  # Ensure label shape (None,)
    else:
        return None, None


# Load data
train_images, train_labels, test_images, test_labels = load_data('pathmnist.npz')
def data_generator(images, labels, batch_size):
    i = 0
    while True:
        processed_images = []
        processed_labels = []
        for _ in range(batch_size):
            if i >= len(images):
                i = 0
            img, label = process_image(images[i], labels[i])  # Process the image
            if img is not None and label is not None:  # Check if the image and label are not None
                processed_images.append(img)
                processed_labels.append(label)
            i += 1

        if processed_images:  # Check if there are processed images
            processed_images = np.array(processed_images)
            processed_labels = np.array(processed_labels)
            processed_labels = np.reshape(processed_labels, (-1,))
            yield processed_images, processed_labels  # Yield the batch
# Create separate training and testing datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_labels, 128),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_images, test_labels, 128),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )
)

# Build and compile the model
def build_and_compile_cnn_model():
    model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    outputs = tf.keras.layers.Dense(9, activation='softmax')(model.output)
    model = tf.keras.Model(inputs=model.input, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', 'mse']
    )
    return model


with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

# Train the model using the training dataset
multi_worker_model.fit(train_dataset, epochs=10, steps_per_epoch=(len(train_images)//128), verbose=1)

# Evaluate the model using the testing dataset
evaluation = multi_worker_model.evaluate(test_dataset,steps==(len(test_images)//128))

