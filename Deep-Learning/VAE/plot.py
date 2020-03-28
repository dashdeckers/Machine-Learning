import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import get_data
from scipy.stats import norm
from VAE import mnist, stanford_dogs  # noqa

# Load experiment variables
experiment = stanford_dogs

model_path = experiment['model_path']
im_shape = experiment['im_shape']
channels = experiment['channels']
latent_dim = experiment['latent_dim']
dataset = experiment['dataset']

# assert latent_dim == 2, 'Currently only supports 2 latent variables'

# Load the saved model
encoder = tf.keras.models.load_model(os.path.join(model_path, 'encoder'))
decoder = tf.keras.models.load_model(os.path.join(model_path, 'decoder'))

# Get some labelled testing data
_, test, _ = get_data(
    batch_size=500,
    im_shape=im_shape,
    labels=True,
    dataset=dataset,
)
images, labels = list(test.take(1).as_numpy_iterator())[0]

# Display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(images, steps=1)
plt.figure(figsize=(6, 6))
plt.scatter(
    z_test[:, 0],
    z_test[:, 1],
    c=labels,
    alpha=.4,
    s=3**2,
    cmap='viridis'
)
plt.colorbar()
plt.show()

# Display a 2D manifold of the latent variables
n = 15              # Number of digits
size = im_shape[0]  # Size of each digit

# linearly spaced coordinates on the unit square are transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z.
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n * n, 2))

# If we only have one image channel, creating the image grid is easy
if channels == 1:
    x_decoded = x_decoded.reshape(n, n, size, size)
    image_grid = np.block(list(map(list, x_decoded)))
    assert image_grid.shape == (n * size, n * size)
    cmap = 'gray'

# With 3 channels, it is slightly different
elif channels == 3:
    x_decoded = x_decoded.reshape(n, n, size, size, channels)
    image_grid = np.dstack((
        np.block(list(map(list, x_decoded[:, :, :, :, 0]))),
        np.block(list(map(list, x_decoded[:, :, :, :, 1]))),
        np.block(list(map(list, x_decoded[:, :, :, :, 2]))),
    ))
    assert image_grid.shape == (n * size, n * size, 3)
    cmap = 'viridis'

# Plot the manifold
plt.figure(figsize=(10, 10))
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False
)
plt.imshow(image_grid, cmap=cmap)
plt.show()
