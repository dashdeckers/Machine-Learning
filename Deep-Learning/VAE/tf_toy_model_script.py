import math

import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import kl_div, rel_entr
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from tensorflow.keras import layers


def show_reconstructed_image(model, test, im_idx=0):
    """Show a test image before and after reconstruction.

    This basic visualization just shows the reconstruction quality
    which does not have to be great for it to be a good model.
    """
    test_batch = list(test.take(1).as_numpy_iterator())[0][0]
    test_input = tf.reshape(test_batch[im_idx], model.exp['input_shape'])

    before = tf.reshape(test_input, model.im_shape).numpy()
    after = tf.reshape(model(test_input), model.im_shape).numpy()

    plt.imshow(before); plt.show()  # noqa
    plt.imshow(after); plt.show()  # noqa


def show_shapes(input_shape, layer_list, name):
    """Show the tensor shapes that flow through the model.

    Basic visualization of tensor shapes to track architecture and flow.
    Layers must be successive starting from input_shape or it won't work,
    and the Sampling layer does not work either, use dense_mean instead.
    """
    shapes = [input_shape]
    for layer in layer_list:
        shapes.append(layer.compute_output_shape(shapes[-1]))

    print(f'\n{name}:', ' --> '.join([str(s[1:]) for s in shapes]), '\n')


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z using the reparameterization trick.

    For z in range(laten_dim):
        sample a point from N(z_mean, z_log_var)
    """

    def __init__(self, latent_dim, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # assert z_mean.shape[1:] == z_log_var.shape[1:], z_log_var.shape
        # assert z_mean.shape[1:] == self.latent_dim, z_mean.shape

        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    """Maps an image to: (z_mean, z_log_var, z)."""

    def __init__(self, exp, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = exp['latent_dim']
        self.layer_list = exp['encoder_layers']
        self.exp = exp

    def build(self, input_shape):
        assert input_shape[1:] == self.exp['input_shape'][1:]

        self.flatten = layers.Flatten()
        self.dense_mean = layers.Dense(units=self.latent_dim)
        self.dense_log_var = layers.Dense(units=self.latent_dim)
        self.sampling = Sampling(latent_dim=self.latent_dim)

        show_shapes(
            input_shape,
            self.layer_list + [self.flatten, self.dense_mean],
            name='Encoder'
        )

    def call(self, inputs):
        output = inputs
        for layer in self.layer_list:
            output = layer(output)
        output = self.flatten(output)

        # Split to get (mean, variance) for each z
        z_mean = self.dense_mean(output)
        z_log_var = self.dense_log_var(output)

        # Sample from (mean, variance) to get z
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.Model):
    """Converts z back into an image."""

    def __init__(self, exp, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.latent_dim = exp['latent_dim']
        self.layer_list = exp['decoder_layers']
        self.exp = exp

    def build(self, input_shape):
        assert input_shape[1:] == self.latent_dim, input_shape[1:]
        show_shapes(input_shape, self.layer_list, name='Decoder')

    def call(self, inputs):
        output = inputs
        for layer in self.layer_list:
            output = layer(output)
        assert output.shape[1:] == self.exp['input_shape'][1:]
        return output


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model."""

    def __init__(self, exp, name='VAE', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.im_shape = exp['im_shape']
        self.beta = exp['beta']
        self.exp = exp

    def build(self, input_shape):
        assert len(input_shape) == 4, f'(batch, w, h, c): {input_shape}'
        self.encoder = Encoder(exp=self.exp)
        self.decoder = Decoder(exp=self.exp)

    """
    Just testing these methods.
    """
    @staticmethod
    def kl_divergence(p, q):
        assert len(p) == len(q), 'Must have the same length'
        assert sum(p) == sum(q) == 1, 'Must sum to 1 to be a probability dist.'
        return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))

    @staticmethod
    def kl_divergence_lib(p, q):
        return (rel_entr(p, q), kl_div(p, q), entropy(p, q))

    @staticmethod
    def mutual_information(p, q):
        return mutual_info_score(p, q)

    """
    Don't rely on them just yet.
    """

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Add KL divergence loss
        # TODO: This calculation needs to be dissected into the 3 parts from
        # the ELBO TC-Decomposition (See the VAE paper)
        kl_loss = (
            - 0.5
            * tf.reduce_mean(
                z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var)
                + 1
            )
        )

        """
        kl_loss = mutual_info + total_corr + dim_wise_kl

        This guy says mutual information can be computed via the
        scipy library function:
        https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python

        Mutual information is function()

        Total correlation is KL( q(z) || PROD q(z_j) )

        Dimensionwise KL is SUM: KL( q(z_j) || p(z_j) )
        """

        # Add negative log likelihood loss (likelihood of the data)
        # This regularization term is currently set as the L2 norm (MSE)
        # but it can also be categorical cross entropy if the sigmoid
        # is used in the output layer so that all values are in [0, 1]
        rec_loss = tf.reduce_sum(
            tf.square(inputs[0, :, :, :] - reconstructed[0, :, :, :])
        )

        self.add_loss(tf.reduce_mean(rec_loss + kl_loss), inputs=inputs)

        # breakpoint()

        # TODO: Save reconstructed image everytime to see the quality increase?
        return reconstructed
