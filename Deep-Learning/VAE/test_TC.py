import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import get_data
from model import get_model
from scipy.special import logsumexp


def log_density_gaussian(x, mu, logvar):
    """Calculate log density of a Gaussian."""
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = np.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)

    return log_density


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculate log density of a Gaussian for all combination of batch pairs.

    I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    """
    batch_size, latent_dim = x.shape

    x = x.copy().reshape(batch_size, 1, latent_dim)
    mu = mu.copy().reshape(1, batch_size, latent_dim)
    logvar = logvar.copy().reshape(1, batch_size, latent_dim)

    matrix = log_density_gaussian(x, mu, logvar)
    assert matrix.shape == (batch_size, batch_size, latent_dim)

    return matrix


def get_probs(z_sample, z_mean, z_log_var):
    """Compute all the necessary probabilities."""
    batch_size, latent_dim = z_sample.shape
    zeros = np.zeros(shape=(batch_size, latent_dim))

    log_q_zx = log_density_gaussian(z_sample, z_mean, z_log_var).sum(axis=1)
    log_p_z = log_density_gaussian(z_sample, zeros, zeros).sum(axis=1)

    mat_log_q_z = matrix_log_density_gaussian(z_sample, z_mean, z_log_var)

    log_q_z = logsumexp(mat_log_q_z.sum(axis=2), axis=1)
    log_prod_q_zi = logsumexp(mat_log_q_z, axis=1).sum(axis=1)

    return log_q_zx, log_p_z, log_q_z, log_prod_q_zi


def reconstruction_loss(batch, reconstructed, distribution):
    batch_size, height, width, channels = batch.shape

    # Bernoulli distribution is for categorical classification,
    # (maybe) because BinaryCrossentropy is used for that.
    # This also requires the values to be in [0, 1], so use a
    # sigmoidal activation on the last decoder layer.
    if distribution == 'bernoulli':
        loss_func = tf.keras.losses.BinaryCrossentropy()
        loss = loss_func(reconstructed, batch)

    # Gaussian distribution is for continuous (pixel) variables
    # which corresponds to the L2 norm (MSE loss).
    elif distribution == 'gaussian':
        loss_func = tf.keras.losses.MeanSquaredError()
        loss = loss_func(reconstructed * 255, batch * 255) / 255

    loss = loss / batch_size

    return loss


def KL_loss_to_unit_normal(mu, logvar):
    """Calculate the KL divergence between a normal dist and a unit normal."""
    # Batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mu**2 + np.exp(logvar)).mean(axis=0)
    # Total kl
    return latent_kl.sum()


def main(n_iterations, project_name='testing'):
    # Init the model and exp
    vae, exp = get_model(project_name, resume=True)
    vae.predict(tf.zeros(exp['input_shape']))

    # Get the data iterator
    _, test, _ = get_data(
        batch_size=exp['batch_size'],
        im_shape=exp['im_shape'],
        labels=True,
        dataset=exp['dataset'],
    )
    numpy_iter = test.as_numpy_iterator()

    # Define metrics to store them later
    global losses, output, probs
    losses = {'kl_loss', 'rec_loss', 'mi_loss', 'tc_loss', 'dw_kl_loss', 'summed_KL_loss'}
    output = {'z_mean', 'z_log_var', 'z_sample'}
    probs = {'q_zx', 'p_z', 'q_z', 'prod_q_zi'}
    store = defaultdict(list)

    for i in range(n_iterations):
        try:
            # Get a batch of data
            images, labels = numpy_iter.next()
            print(labels)

        except StopIteration:
            # Unless theres no more
            break

        # Feed the batch to the model
        z_mean, z_log_var, z_sample = vae.encoder(images)
        reconstructed = vae.decoder(z_sample)

        # Convert from tensorflow to pure numpy
        z_sample = z_sample.numpy()
        z_log_var = z_log_var.numpy()
        z_mean = z_mean.numpy()

        # z is a batch of samples with dimension latent_dim
        assert z_sample.shape == (exp['batch_size'], exp['latent_dim'])

        # Compute losses
        rec_loss = reconstruction_loss(images, reconstructed, 'gaussian')
        rec_loss = rec_loss.numpy()

        kl_loss = KL_loss_to_unit_normal(z_mean, z_log_var)

        q_zx, p_z, q_z, prod_q_zi = get_probs(z_sample, z_mean, z_log_var)

        mi_loss = (q_zx - q_z).mean()
        tc_loss = (q_z - prod_q_zi).mean()
        dw_kl_loss = (prod_q_zi - p_z).mean()
        summed_KL_loss = mi_loss + tc_loss + dw_kl_loss

        # Store metrics
        var_dict = locals()
        for key, value in var_dict.items():
            if key in set.union(losses, output, probs):
                store[key].append(value)

    return store


# kl_loss_new == kl_loss_old * 2 (bit wierd, but not an issue)
# kl_loss_new =~ summed_KL_loss

n = 10
store = main(n)
for key in losses:
    plt.plot(range(n), store[key], label=key)

plt.legend()
plt.show()
