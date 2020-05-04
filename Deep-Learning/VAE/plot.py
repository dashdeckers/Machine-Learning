import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from data import get_data
from model import get_model
from scipy.stats import norm

tf.keras.backend.set_floatx('float64')


def plot_digit_classes_in_latent_space(encoder, exp):
    """Plot digit classes in 2D latent space.

    Currently only works for MNIST and can be used to visualize disentanglement
    in terms of separation of classes.

    To generalize to more latent dimensions, maybe use a Kohonen map or just
    compare the final metrics of clustering algorithms in N-d space.
    """

    # Get some labelled testing data
    _, test, _ = get_data(
        batch_size=500,
        im_shape=exp['im_shape'],
        labels=True,
        dataset=exp['dataset'],
    )
    images, labels = list(test.take(1).as_numpy_iterator())[0]

    # Display a 2D plot of the digit classes in the latent space
    _, _, z_test = encoder(images)
    z_test = z_test.numpy()

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
    plt.savefig(exp['project_name']+"/digits.png")
    plt.close()


def make_latent_grid(
            latent_dim,
            latent_indices,
            n,
            show_all_dims
        ):
    """Make a grid of latent variables to be plotted.

    For either a grid exploring the interaction between 2 dimensions
    Or a grid exploring all different dimensions without interactions.

    This functions creates a grid of variables to fill a latent space.

    When showing all dimensions all latent spaces get set to 1,
    but in each row 1 latent variable explores a range.
    This make a row for each latent dim exploring their effect.
    This gives a latent_dim x n grid of images.

    When !show_all_dims 2 latent indices can be given.
    This indices will get a range set against eachother.
    All other variables will remain at 1.
    This gives a nxn grid of images.

    This functions returns the latent variables for each slot in a grid.
    As well as the number of images the grid is wide and high.
    """
    # Create the 2 layers for the values for (z1, z2)
    if show_all_dims:
        # We take 1 linspace as we only explore variable at each row
        u_grid = np.dstack(np.linspace(0.05, 0.95, n))
    else:
        u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                                       np.linspace(0.05, 0.95, n)))

    # Make them normally distributed (from [0, 1] to [-1, -1])
    z_grid = norm.ppf(u_grid)

    if show_all_dims:
        range_grid = z_grid[0, :, :]
        z_grid = np.ndarray(shape=[0, n, latent_dim])
        # For each dimension we add a row with a set of all dimensions
        for j in range(latent_dim):
            stack_list = list()
            for idx in range(latent_dim):
                if idx == j:
                    # Each row has 1 variable which gets the range
                    stack_list.append(range_grid)
                else:
                    # All other variables simply get 1 throughout in the row
                    stack_list.append(np.ones(shape=(1, n)))
            new = np.dstack(stack_list)
            # Put all the rows exploring different vars together
            z_grid = np.append(z_grid, new, axis=0)
        assert z_grid.shape == (latent_dim, n, latent_dim)
        return z_grid, latent_dim, n
    else:
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
        return z_grid, n, n


def plot_2D_manifold_of_latent_variables(
            decoder,
            exp,
            latent_indices,
            show_all_dims=False
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
    latent_dim = exp['latent_dim']
    channels = exp['channels']
    im_shape = exp['im_shape']

    assert im_shape[0] == im_shape[1], 'Only square images are supported'
    assert len(latent_indices) == 2, 'We can only plot a 2D manifold'
    assert latent_indices[0] != latent_indices[1], 'Plot both z1 and z2 please'
    assert (latent_dim == 2 and latent_indices == (0, 1)) or latent_dim > 2, \
        'If there are only 2 latent dimensions, the indices must be (0, 1)'

    # Number of images, size of each image
    n = 15
    size = im_shape[0]

    z_grid, width, height = make_latent_grid(latent_dim, latent_indices,
                                             n, show_all_dims)

    # Get predictions
    x_decoded = decoder(z_grid.reshape(width * height, latent_dim)).numpy()

    # If we only have one image channel, creating the image grid is easy
    if channels == 1:
        x_decoded = x_decoded.reshape(width, height, size, size)
        image_grid = np.block(list(map(list, x_decoded)))
        assert image_grid.shape == (width * size, height * size)
        cmap = 'gray'

    # With 3 channels, it is slightly different
    elif channels == 3:
        x_decoded = x_decoded.reshape(width, height, size, size, channels)
        image_grid = np.dstack((
            np.block(list(map(list, x_decoded[:, :, :, :, 0]))),
            np.block(list(map(list, x_decoded[:, :, :, :, 1]))),
            np.block(list(map(list, x_decoded[:, :, :, :, 2]))),
        ))
        assert image_grid.shape == (width * size, height * size, 3)
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
    plt.savefig(exp['project_name']+"/latent/"+str(latent_indices[0])+"-"+str(latent_indices[1])+".png")
    plt.close()


def plot_all_2D_manifolds(decoder, exp):
    """Plot all possible combinations of latent variables to vary."""
    for latent_indices in combinations(range(exp['latent_dim']), 2):
        plot_2D_manifold_of_latent_variables(decoder, exp, latent_indices)


def plot_independent_grid(decoder, exp):
    plot_2D_manifold_of_latent_variables(decoder, exp, (0, 1), True)


def plot_losses(project_name, losses):
    """Plot the losses per epoch.

    losses must be a list containing one or more of the following:
    ['decomp', 'main', 'val_decomp', 'val_main']
    """
    assert isinstance(losses, list)
    assert set(losses)
    decomp_losses = ['mi_loss', 'tc_loss', 'dw_kl_loss']
    main_losses = ['kl_loss', 'rec_loss']
    val_decomp_losses = ['val_' + loss for loss in decomp_losses]
    val_main_losses = ['val_' + loss for loss in main_losses]

    df = pd.read_csv(
        os.path.join(args.name, 'losses.csv'),
        sep=',',
    )

    loss_list = list()
    if 'decomp' in losses:
        loss_list.extend(decomp_losses)
    if 'main' in losses:
        loss_list.extend(main_losses)
    if 'val_decomp' in losses:
        loss_list.extend(val_decomp_losses)
    if 'val_main' in losses:
        loss_list.extend(val_main_losses)

    df[loss_list].plot()
    plt.xlabel('Epochs')
    plt.title('Loss values per epoch')
    plt.savefig(exp['project_name']+"/losses.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Plotter')
    parser.add_argument('name', type=str)
    parser.add_argument('--checkpoint', default='newest')
    args = parser.parse_args()

    # Load the model (exp must have the same architecture as the saved model)
    vae, exp = get_model(
        project_name=args.name,
        resume=True,
        checkpoint=args.checkpoint,
    )

    if not os.path.exists(args.name+"/latent"):
        os.makedirs(args.name+"/latent")

    # Use CTRL+C to quit early
    plot_losses(args.name, ['decomp', 'main'])
    plot_digit_classes_in_latent_space(vae.encoder, exp)
    # plot_independent_grid(vae.decoder, exp)
    plot_all_2D_manifolds(vae.decoder, exp)
