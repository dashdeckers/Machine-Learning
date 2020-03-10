import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.stats import norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Dense, Input, Lambda, Layer, Multiply
from tensorflow.keras.models import Model, Sequential

im_shape = (16, 16)
channels = 3
original_dim = im_shape[0] * im_shape[1] * channels  # 784
intermediate_dim = 256
latent_dim = 2
batch_size = 1024
epochs = 3
epsilon_std = 1.0


def nll(y_true, y_pred):
    """Negative log likelihood (Bernoulli)."""

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """Identity transform layer that adds KL divergence to the final loss."""

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

# Image input
x = Input(shape=(original_dim, ))
# Encoder
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = K.random_normal(stddev=epsilon_std,
                      shape=(K.shape(x)[0], latent_dim),
                      seed=42)
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(x, x_pred)

vae.compile(optimizer='rmsprop', loss=nll)


dataset, info = tfds.load(
    "stanford_dogs",
    split="train",
    with_info=True,
)

val_dataset = tfds.load(
    "stanford_dogs",
    split="test",
)

steps_per_epoch = math.ceil(info.splits["train"].num_examples / batch_size)
val_steps = math.ceil(info.splits["test"].num_examples / batch_size)


def preprocessing(data):
    image = tf.cast(data["image"], tf.float32)
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, im_shape)
    image = tf.reshape(image, [-1])
    return image, image


dataset = (
    dataset.cache()
    .shuffle(10 * batch_size)
    .repeat()
    .map(preprocessing)
    .batch(batch_size)
    .prefetch(5)
)

val_dataset = (
    val_dataset.cache()
    .repeat()
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(5)
)

vae.fit(dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps)


def preprocessing_y(data):
    image = tf.cast(data["image"], tf.float32)
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, im_shape)
    image = tf.reshape(image, [-1])
    print(image.shape)
    return image, data["label"]


val_yset = (
    tfds.load(
        "stanford_dogs",
        split="test",
    ).cache()
    .repeat()
    .map(preprocessing_y, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(225)
    .prefetch(5)
)

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(val_dataset, steps=1)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1])  # , c=1,
# alpha=.4, s=3**2, cmap='viridis')
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = im_shape[0]

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n*n, 2))
x_decoded = x_decoded.reshape(n, n, digit_size, digit_size, channels)

x = np.dstack((np.block(list(map(list, x_decoded[:, :, :, :, 0]))),
              np.block(list(map(list, x_decoded[:, :, :, :, 1]))),
              np.block(list(map(list, x_decoded[:, :, :, :, 2])))))
assert x.shape == (240, 240, 3)
plt.figure(figsize=(10, 10))
# plt.imshow(np.block(list(map(list, x_decoded[:, :, :, :, 0]))), cmap='gray')
plt.imshow(x)

# plt.imshow(x_decoded.reshape(15*im_shape[0], 15*im_shape[1], channels))
plt.show()


# fig = plt.figure(figsize=(n, n))
# i = 1
# for c in range(n):  # columns
#     for r in range(n):  # rows
#         img = x_decoded[c][r]
#         plt.subplot(n, n, i, sharex=True, sharey=True)
#         plt.imshow(img)
#         i += 1
# plt.show()


def make_image(val1, val2):
    inp = np.array([val1, val2]).reshape(-1, 2)
    plt.imshow(decoder.predict(inp).reshape(digit_size, digit_size),
               cmap='gray')
    plt.show()
