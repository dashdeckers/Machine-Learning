"""Define the VAE model, project/experiment management, and GPU config."""
import glob
import json
import math
import os

import tensorflow as tf
from experiment import get_experiment
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')


def create_project_safely(name, exp):
    """Create a project directory and write the experiment file to it.

    Does not do anything if the directory already exists.
    """
    if not os.path.exists(name):
        print('\nCreating a new model at', name, '\n')
        os.makedirs(name)

        with open(os.path.join(name, 'experiment.json'), 'w') as file:
            file.write(json.dumps(
                exp,
                indent=4,
                default=lambda layer: layer.get_config())
            )


def get_model(project_name, resume, beta, tc, checkpoint='newest'):
    """Create and retrieve a VAE model, and load saved weights if needed."""
    exp = get_experiment(project_name, beta, tc, resume)
    create_project_safely(project_name, exp)

    vae = VariationalAutoEncoder(exp)
    vae.compile(optimizer=exp['optimizer'])

    if not resume:
        return vae, exp

    list_of_saved_models = glob.glob(os.path.join(project_name, '*.h5'))
    if not list_of_saved_models:
        print(f'No saved checkpoints found at {project_name}')
        return vae, exp

    # Create the weights if the model was not already initialized
    vae.predict(tf.zeros(vae.exp['input_shape'], dtype=tf.float64))

    # Load the latest checkpoint model
    if checkpoint == 'newest':
        latest_model = max(list_of_saved_models, key=os.path.getctime)
        print('Loading the model from checkpoint', latest_model, '\n')
        vae.load_weights(latest_model)

    else:
        print('Loading the model from checkpoint', checkpoint, '\n')
        vae.load_weights(os.path.join(project_name, checkpoint))

    return vae, exp


def gpu_configuration():
    """If possible, set the model to run on the GPU."""
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Dynamically allocate GPU memory use
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    except (ValueError, RuntimeError, IndexError):
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def show_shapes(input_shape, layer_list, name):
    """Show the tensor shapes that flow through the model.

    Basic visualization of tensor shapes to track architecture and flow.
    Layers must be successive starting from input_shape or it won't work,
    and the Sampling layer does not work either, use dense_mean instead.
    """
    try:
        shapes = [input_shape]
        for layer in layer_list:
            # print(layer.name)
            shapes.append(layer.compute_output_shape(shapes[-1]))

    except Exception:
        print(f'Problem with {layer.name}, input shape is {shapes[-1]}')

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
        self.alpha = exp['alpha']
        self.beta = exp['beta']
        self.gamma = exp['gamma']
        self.exp = exp

    def build(self, input_shape):
        assert len(input_shape) == 4, f'(batch, w, h, c): {input_shape}'
        self.encoder = Encoder(exp=self.exp)
        self.decoder = Decoder(exp=self.exp)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Compute main losses
        rec_loss = self.reconstruction_loss(
            inputs, reconstructed, self.exp['distribution']
        )
        kl_loss = self.KL_loss_to_unit_normal(z_mean, z_log_var)

        # Compute decomposed losses
        q_zx, p_z, q_z, prod_q_zi = self.get_probs(z, z_mean, z_log_var)

        mi_loss = tf.reduce_mean(q_zx - q_z)
        tc_loss = tf.reduce_mean(q_z - prod_q_zi)
        dw_kl_loss = tf.reduce_mean(prod_q_zi - p_z)

        # Add losses
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           self.gamma * dw_kl_loss)
        self.add_loss(loss, inputs=inputs)

        self.add_metric(rec_loss, aggregation='mean', name='rec_loss')
        self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
        self.add_metric(mi_loss, aggregation='mean', name='mi_loss')
        self.add_metric(tc_loss, aggregation='mean', name='tc_loss')
        self.add_metric(dw_kl_loss, aggregation='mean', name='dw_kl_loss')

        return reconstructed

    def reconstruction_loss(self, inputs, reconstructed, distribution):
        batch_size, height, width, channels = inputs.shape
        batch_size = self.exp['batch_size']

        # Bernoulli distribution is for categorical classification,
        # (maybe) because BinaryCrossentropy is used for that.
        # This also requires the values to be in [0, 1], so use a
        # sigmoidal activation on the last decoder layer.
        if distribution == 'bernoulli':
            loss_func = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            )

        # Gaussian distribution is for continuous (pixel) variables
        # which corresponds to the L2 norm (MSE loss).
        elif distribution == 'gaussian':
            loss_func = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM
            )

        loss = loss_func(inputs, reconstructed)
        return loss / batch_size

    def KL_loss_to_unit_normal(self, mu, logvar):
        """Calculate KL divergence between a normal dist and a unit normal."""
        return 0.5 * tf.reduce_mean(tf.reduce_sum(
            tf.square(mu) + tf.exp(logvar) - logvar - 1, [1]), name="kl_loss")

    def log_density_gaussian(self, x, mu, logvar):
        """Calculate log density of a Gaussian."""
        pi = tf.constant(math.pi, dtype=tf.float64)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.exp(-logvar)
        tmp = (x - mu)
        return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)

    def get_probs(self, z, mu, logvar):
        """Compute all the necessary probabilities."""
        batch_size, latent_dim = z.shape
        batch_size = self.exp['batch_size']

        zeros = tf.zeros(shape=(batch_size, latent_dim), dtype=tf.float64)

        log_q_zx = tf.reduce_sum(
            self.log_density_gaussian(z, mu, logvar), axis=1, keepdims=False
        )
        log_p_z = tf.reduce_sum(
            self.log_density_gaussian(z, zeros, zeros), axis=1, keepdims=False
        )

        mat_log_q_z = self.log_density_gaussian(
            tf.expand_dims(z, 1),
            tf.expand_dims(mu, 0),
            tf.expand_dims(logvar, 0)
        )

        log_q_z = tf.reduce_logsumexp(tf.reduce_sum(
            mat_log_q_z, axis=2, keepdims=False), axis=1, keepdims=False
        )

        log_prod_q_zi = tf.reduce_sum(tf.reduce_logsumexp(
            mat_log_q_z, axis=1, keepdims=False), axis=1, keepdims=False
        )

        return log_q_zx, log_p_z, log_q_z, log_prod_q_zi
