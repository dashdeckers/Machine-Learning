"""Define the VAE model, project/experiment management, and GPU config."""
import glob
import json
import math
import os

import numpy as np
import tensorflow as tf
from scipy.special import kl_div, rel_entr
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')


def get_experiment(project_name, resume=False):
    """Define the experiment here!.

    Everything about the VAE that can be changed is defined here. You can
    play with all the parameters you want from the exp dictionary.
    """
    if resume:
        funcs = {
            'dense': layers.Dense,
            'flatten': layers.Flatten,
            'reshape': layers.Reshape,
            'optimizer': tf.keras.optimizers.Adam,
        }
        # Get experiment from file
        with open(os.path.join(project_name, 'experiment.json')) as file:
            exp = json.load(file)

        # Convert layer configs into actual layers
        exp['optimizer'] = funcs['optimizer'].from_config(exp['optimizer'])
        for enc_dec in ['encoder_layers', 'decoder_layers']:
            for idx, config in enumerate(exp[enc_dec]):
                for fname in funcs.keys():
                    if config['name'].startswith(fname):
                        exp[enc_dec][idx] = funcs[fname].from_config(config)

        return exp

    # Define the experiment
    exp = {
        'project_name': project_name,
        'dataset': 'mnist',  # 'stanford_dogs'
        'input_shape': (1, 28, 28, 1),  # (1, 32, 32, 3)
        'batch_size': 64,
        'epochs': 10,

        'latent_dim': 2,
        'beta': 1.0,
        'optimizer': tf.keras.optimizers.Adam(),  # rmsprop
    }
    exp['im_shape'] = exp['input_shape'][1:3]
    exp['channels'] = exp['input_shape'][3]
    exp['col_dim'] = int(np.prod(exp['input_shape'][1:]))

    # Define the architecture
    exp['encoder_layers'] = [
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
    ]

    exp['decoder_layers'] = [
        layers.Dense(units=256, activation='relu'),
        layers.Dense(units=exp['col_dim'], activation='relu'),
        layers.Reshape(target_shape=exp['input_shape'][1:]),
    ]
    return exp


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


def get_model(project_name, resume, checkpoint='newest'):
    """Create and retrieve a VAE model, and load saved weights if needed."""
    exp = get_experiment(project_name, resume)
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

        Mutual information is the function

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
