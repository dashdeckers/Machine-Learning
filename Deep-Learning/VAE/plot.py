import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import get_data
from scipy.stats import norm
from VAE import mnist, stanford_dogs  # noqa

# Load experiment variables
experiment = mnist
# experiment = stanford_dogs

model_path = experiment['model_path']
im_shape = experiment['im_shape']
channels = experiment['channels']
latent_dim = experiment['latent_dim']
dataset = experiment['dataset']

# Load the saved model
encoder = tf.keras.models.load_model(os.path.join(model_path, 'encoder'))
decoder = tf.keras.models.load_model(os.path.join(model_path, 'decoder'))


def plot_digit_classes_in_latent_space(encoder, im_shape, dataset):
    """Plot digit classes in 2D latent space.

    Currently only works for MNIST and can be used to visualize disentanglement
    in terms of separation of classes.

    To generalize to more latent dimensions, maybe use a Kohonen map or just
    compare the final metrics of clustering algorithms in N-d space.
    """
    assert experiment == mnist, 'Currently only supports MNIST'

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


def plot_2D_manifold_of_latent_variables(
            decoder,
            latent_dim,
            latent_indices,
            channels,
            im_shape
        ):
    """Plot a 2D manifold of latent variables.

    We can visualize the effect 2 of the latent variables by fixing the other
    variables to a constant (zeros), then varying the two around the Gaussian
    distribution, and finally showing the images that the decoder would produce
    in a 2D manifold.

    Conceptually, we are making a n*n grid with as many layers as latent
    dimensions, so that each point on the grid is a vector that can be passed
    into the decoder to create an image.

    Two of these layers can represent the (z1, z2) values, which are normally
    distributed from [-1, 1], of the latent variables that we wish to vary in
    the 2D space and visualize. The rest can be zeros.

    Then, we get an image prediction for each point in the grid and plot them.
    """
    assert im_shape[0] == im_shape[1], 'Only square images are supported'
    assert len(latent_indices) == 2, 'We can only plot a 2D manifold'
    assert latent_indices[0] != latent_indices[1], 'Plot both z1 and z2 please'
    assert (latent_dim == 2 and latent_indices == (0, 1)) or latent_dim > 2, \
        'If there are only 2 latent dimensions, the indices must be (0, 1)'

    # Number of images, size of each image
    n = 15
    size = im_shape[0]

    # Create the 2 layers for the values for (z1, z2)
    u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                                   np.linspace(0.05, 0.95, n)))
    # Make them normally distributed (from [0, 1] to [-1, -1])
    z_grid = norm.ppf(u_grid)

    # Put them together with the zero layers
    z_grid_1, z_grid_2 = np.dsplit(z_grid, 2)
    stack_list = list()

    for idx in range(latent_dim):
        if idx == latent_indices[0]:
            stack_list.append(z_grid_1)
        elif idx == latent_indices[1]:
            stack_list.append(z_grid_2)
        else:
            stack_list.append(np.ones(shape=(n, n)))

    z_grid = np.dstack(stack_list)
    assert z_grid.shape == (n, n, latent_dim)

    # Get predictions
    x_decoded = decoder.predict(z_grid.reshape(n * n, latent_dim))

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


def plot_all_2D_manifolds(decoder, latent_dim, channels, im_shape):
    """Plot all possible combinations of latent variables to vary."""
    for latent_indices in combinations(range(latent_dim), 2):
        plot_2D_manifold_of_latent_variables(
            decoder,
            latent_dim,
            latent_indices,
            channels,
            im_shape
        )


if __name__ == '__main__':
    # Use CTRL+C to quit early
    plot_all_2D_manifolds(decoder, latent_dim, channels, im_shape)
