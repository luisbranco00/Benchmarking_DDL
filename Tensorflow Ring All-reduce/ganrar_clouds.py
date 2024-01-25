
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, initializers

import argparse
import os
import json
import logging
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img


class Train(keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
    

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, real_images):

        # save loss values
        G_loss=[]
        D_loss=[]

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (we should *not* update the weights of the discriminator)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        G_loss.append(self.g_loss_metric.result())
        D_loss.append(self.d_loss_metric.result())
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }



# saves generated images
class GANMonitor(keras.callbacks.Callback):
    
    def __init__(self, latent_dim, num_img):
        self.num_img = num_img
        self.latent_dim = latent_dim


    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        
        generated_images = self.model.generator(random_latent_vectors)
        
        generated_images *= 255
        generated_images.numpy()

        preview_dir = './dcgan'

        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(preview_dir+"/_img_%03d_%d.png" % (epoch, i))



def main():



    
    parser = argparse.ArgumentParser(description='Keras example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.')
    parser.add_argument('--n_hidden', '-n', type=int, default=128,
                        help='Number of hidden units (z)')
    parser.add_argument('--index', '-p', type=int, default=0,
                        help='worker index')

    args = parser.parse_args()

    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    os.environ['TF_CONFIG'] = json.dumps({ 'cluster': { 'worker': ["cloud98:8050","cloud99:8052","cloud103:8053","cloud104:8053"]},
     'task': {'type': 'worker', 'index': args.index}})
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver())
    # create dataset from folder
    dataset = keras.utils.image_dataset_from_directory(
        args.dataset, label_mode=None, image_size=(256, 256), batch_size=args.batchsize)

    # normalize the images 
    dataset = dataset.map(lambda x: x / 255.0)

    # create the generator
    with strategy.scope():
        generator = keras.models.Sequential(
            [
                keras.Input(shape=(args.n_hidden)),
                
                layers.Dense(4 * 4 * 1024, 
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #l0
                layers.BatchNormalization(synchronized=True), #bn0
                layers.Reshape((4, 4, 1024)),
                
                layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same", 
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc1
                layers.BatchNormalization(synchronized=True), #bn1
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc2
                layers.BatchNormalization(synchronized=True), #bn2
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc3
                layers.BatchNormalization(synchronized=True), #bn3
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc4
                layers.BatchNormalization(synchronized=True), #bn4
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc5
                layers.BatchNormalization(synchronized=True), #bn5
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc6
                layers.BatchNormalization(synchronized=True), #bn6
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same",
                                    kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc7
                layers.Dropout(0.2),
                layers.Activation(activations.tanh),
            ],
            name="generator",
            )
        generator.summary()


        # create the discriminator
        discriminator = keras.models.Sequential(
            [
                keras.Input(shape=(256, 256, 3)),

                layers.Conv2D(32, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c0_0
                layers.ReLU(),

                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"), #c0_1   
                layers.BatchNormalization(synchronized=True), #bn0_1
                layers.ReLU(),

                layers.Conv2D(128, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c1_0   
                layers.BatchNormalization(synchronized=True), #bn1_0
                layers.ReLU(),

                layers.Conv2D(256, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c1_1   
                layers.BatchNormalization(synchronized=True), #bn1_1
                layers.ReLU(),

                layers.Conv2D(256, kernel_size=3, strides=1, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c2_0   
                layers.BatchNormalization(synchronized=True), #bn2_0
                layers.ReLU(),

                layers.Conv2D(512, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c2_1   
                layers.BatchNormalization(synchronized=True), #bn2_1
                layers.ReLU(),

                layers.Conv2D(512, kernel_size=3, strides=1, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c3_0
                layers.BatchNormalization(synchronized=True), #bn3_0
                layers.ReLU(),

                layers.Conv2D(1024, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c3_1   
                layers.BatchNormalization(synchronized=True), #bn3_1
                layers.ReLU(),

                layers.Conv2D(1024, kernel_size=3, strides=1, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c4_0   
                layers.BatchNormalization(synchronized=True), #bn4_0
                layers.Activation(activations.sigmoid),

                layers.Flatten(),
                layers.Dense(1,
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)) #l5
            ],
            name="discriminator",
            )
        discriminator.summary()

        epochs = args.epoch

        # compile and train the model
        gan = Train(discriminator=discriminator, generator=generator, latent_dim=args.n_hidden)
        gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM),
        )

        gan.fit(
        dataset, epochs=epochs, verbose=1, steps_per_epoch=len(dataset)//args.batchsize, callbacks=[GANMonitor(num_img=1, latent_dim=args.n_hidden)]
        ) # verbose=1 gives progress bar


if __name__ == '__main__':
    main()