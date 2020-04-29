import math

import numpy as np
import tensorflow as tf
from data import get_data
from model import get_model
from scipy.special import logsumexp


def print_shit(**kwargs):
    for key, value in kwargs.items():
        print(f'{key}: {value}')


# Init the model and exp
project_name = 'testing'
vae, exp = get_model(project_name, resume=True)
vae.predict(tf.zeros(exp['input_shape']))

# Play with the batch size
exp['batch_size'] = 32

# Get a batch of data
_, test, _ = get_data(
    batch_size=exp['batch_size'],
    im_shape=exp['im_shape'],
    labels=True,
    dataset=exp['dataset'],
)
images, labels = list(test.take(1).as_numpy_iterator())[0]

# Feed it to the model
z_mean, z_log_var, z_sample = vae.encoder(images)
reconstructed = vae.decoder(z_sample)

# Convert from tensorflow to pure numpy
z_sample = z_sample.numpy()
z_log_var = z_log_var.numpy()
z_mean = z_mean.numpy()

# z is a batch of samples with dimension latent_dim
assert z_sample.shape == (exp['batch_size'], exp['latent_dim'])


def log_density_gaussian(x, mu, logvar):
    """Calculate log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.

    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = np.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)

    return log_density


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculate log density of a Gaussian for all combination of batch pairs of `x` and `mu`.

    I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch

    """
    batch_size, latent_dim = x.shape

    x = x.copy().reshape(batch_size, 1, latent_dim)
    mu = mu.copy().reshape(1, batch_size, latent_dim)
    logvar = logvar.copy().reshape(1, batch_size, latent_dim)

    matrix = log_density_gaussian(x, mu, logvar)
    assert matrix.shape == (batch_size, batch_size, latent_dim)

    return matrix


def get_probs(z_sample, z_mean, z_log_var, N_data):
    batch_size, latent_dim = z_sample.shape
    zeros = np.zeros(shape=(batch_size, latent_dim))

    log_q_zx = log_density_gaussian(z_sample, z_mean, z_log_var).sum(axis=1)
    log_p_z = log_density_gaussian(z_sample, zeros, zeros).sum(axis=1)

    mat_log_q_z = matrix_log_density_gaussian(z_sample, z_mean, z_log_var)

    # Using the SciPy implementation of logsumexp. Do they use a different
    # log base maybe??
    log_q_z = logsumexp(mat_log_q_z.sum(axis=2), axis=1)
    log_prod_q_zi = logsumexp(mat_log_q_z, axis=1).sum(axis=1)

    assert (
        log_q_zx.shape
        == log_p_z.shape
        == log_q_z.shape
        == log_prod_q_zi.shape
        == (mat_log_q_z.shape[0],)
        == (mat_log_q_z.shape[1],)
        == (batch_size,)
    )

    return log_q_zx, log_p_z, mat_log_q_z, log_q_z, log_prod_q_zi


def reconstruction_loss(inputs, reconstructed, distribution):
    batch_size, height, width, channels = inputs.shape

    if distribution == 'bernoulli':
        loss_func = tf.keras.losses.BinaryCrossentropy()
        loss = loss_func(reconstructed, inputs)

    elif distribution == 'gaussian':
        loss_func = tf.keras.losses.MeanSquaredError()
        loss = loss_func(reconstructed * 255, inputs * 255) / 255

    loss = loss / batch_size

    return loss


def KL_loss_to_unit_normal(mu, logvar):
    """Calculate the KL divergence between a normal dist and a unit normal.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    storer : dict
        Dictionary in which to store important variables for vizualisation.

    """
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mu**2 + np.exp(logvar)).mean(axis=0)
    total_kl = latent_kl.sum()

    return total_kl


# Old losses
kl_loss_old = (
    - 0.5
    * tf.reduce_mean(
        z_log_var
        - tf.square(z_mean)
        - tf.exp(z_log_var)
        + 1
    )
).numpy()

rec_loss_old = tf.reduce_sum(
    tf.square(images[0, :, :, :] - reconstructed[0, :, :, :])
).numpy()


# New losses
kl_loss_new = KL_loss_to_unit_normal(z_mean, z_log_var)

q_zx, p_z, mat_q_z, q_z, prod_q_zi = get_probs(z_sample, z_mean, z_log_var, 1)

rec_loss_new = reconstruction_loss(images, reconstructed, 'gaussian')

mi_loss = (q_zx - q_z).mean()
tc_loss = (q_z - prod_q_zi).mean()
dw_kl_loss = (prod_q_zi - p_z).mean()
summed_KL_loss = mi_loss + tc_loss + dw_kl_loss


print_shit(
    rec_loss_old=rec_loss_old,
    KL_loss_old=kl_loss_old,
    b1='',
    rec_loss_new=rec_loss_new,
    KL_loss_new=kl_loss_new,
    b2='',
    summed_KL_loss=summed_KL_loss,
)


"""
# Add KL divergence loss
# TODO: This calculation needs to be dissected into the 3 parts from
# the ELBO TC-Decomposition (See the VAE paper)
kl_loss_mean = (
    - 0.5
    * tf.reduce_mean(
        z_log_var
        - tf.square(z_mean)
        - tf.exp(z_log_var)
        + 1
    )
)

rec_loss = tf.reduce_sum(
    tf.square(inputs[0, :, :, :] - reconstructed[0, :, :, :])
)

new_loss = tf.reduce_mean(rec_loss + kl_loss)





kl_loss = mutual_info + total_corr + dim_wise_kl

This guy says mutual information can be computed via the
scipy library function:
https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python

Mutual information is the function

Total correlation is KL( q(z) || PROD q(z_j) )

Dimensionwise KL is SUM: KL( q(z_j) || p(z_j) )
"""

# Add negative log likelihood loss (likelihood of the data)
# This regularization term is currently set as the L2 norm (MSE)
# but it can also be categorical cross entropy if the sigmoid
# is used in the output layer so that all values are in [0, 1]
