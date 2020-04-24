import glob
import json
import math
from os import makedirs, path

import matplotlib.pyplot as plt
import tensorflow as tf
from data import get_data
from plot import plot_all_2D_manifolds, plot_independent_grid
from scipy.special import kl_div, rel_entr
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from tensorflow.keras import layers


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

        assert z_mean.shape[1:] == z_log_var.shape[1:], z_log_var.shape
        assert z_mean.shape[1:] == self.latent_dim, z_mean.shape

        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    """Maps an image to: (z_mean, z_log_var, z)."""

    def __init__(self, exp, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = exp['latent_dim']

    def build(self, input_shape):
        # First layers to capture image data
        self.layer_list = [
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
        ]
        # Then split with two dense outputs for (z_mean, z_log_var)
        self.dense_mean = layers.Dense(units=self.latent_dim)
        self.dense_log_var = layers.Dense(units=self.latent_dim)
        # Sample (z) from (z_mean, z_log_var)
        self.sampling = Sampling(latent_dim=self.latent_dim)

        show_shapes(
            input_shape,
            self.layer_list + [self.dense_mean],
            name='Encoder'
        )

    def call(self, inputs):
        output = inputs
        for layer in self.layer_list:
            output = layer(output)

        z_mean = self.dense_mean(output)
        z_log_var = self.dense_log_var(output)

        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.Model):
    """Converts z back into an image."""

    def __init__(self, exp, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.latent_dim = exp['latent_dim']
        self.im_dim = exp['im_shape'] + (exp['channels'], )
        self.col_dim = self.im_dim[0] * self.im_dim[1] * self.im_dim[2]

    def build(self, input_shape):
        assert input_shape[1:] == (self.latent_dim), input_shape[1:]

        self.layer_list = [
            layers.Dense(units=256, activation='relu'),
            layers.Dense(units=self.col_dim, activation='sigmoid'),
            layers.Reshape(target_shape=self.im_dim),
        ]
        show_shapes(input_shape, self.layer_list, name='Decoder')

    def call(self, inputs):
        output = inputs
        for layer in self.layer_list:
            output = layer(output)
        return output


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model."""

    def __init__(self, exp, name='VAE', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.latent_dim = exp['latent_dim']
        self.im_dim = exp['im_shape'] + (exp['channels'], )
        self.beta = exp['beta']
        self.model_path = exp['model_path']
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


def get_model(exp):
    """Load a model from checkpoint or create a new one.

    Manages loading from most recent checkpoint, creating the
    directories, returning the model.
    """
    model_path = exp['model_path']
    input_shape = (1,) + exp['im_shape'] + (exp['channels'], )

    vae = VariationalAutoEncoder(exp)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae(tf.zeros(input_shape))  # create the weights

    if path.exists(model_path):
        # Grab the newest .h5 model checkpoint file, if it exists
        list_of_saved_models = glob.glob(path.join(model_path, '*.h5'))

        if list_of_saved_models:
            latest_model = max(list_of_saved_models, key=path.getctime)

            # Load the latest checkpoint model
            print('Loading the model from checkpoint', latest_model, '\n')
            vae.load_weights(latest_model)

    else:
        # Create a new model folder and write the experiment dict to file
        print('Creating a new model at', model_path, '\n')
        makedirs(model_path)
        with open(path.join(model_path, 'experiment.json'), 'w') as file:
            file.write(json.dumps(exp, indent=4))

    return vae


def show_reconstructed_digit(vae, index=0):
    """Show a test image before and after reconstruction."""
    test_batch = list(test.take(1).as_numpy_iterator())[0][0]
    test_input = tf.reshape(test_batch[index], (1, 28, 28, 1))

    before = tf.reshape(test_input, (28, 28)).numpy()
    after = tf.reshape(vae(test_input), (28, 28)).numpy()

    plt.imshow(before)
    plt.show()
    plt.imshow(after)
    plt.show()


# Define the experiment
# exp = {
#     'dataset': 'mnist',
#     'im_shape': (28, 28),
#     'channels': 1,
#     'latent_dim': 2,
#     'batch_size': 64,
#     'epochs': 10,
#     'beta': 1.0,
#     'model_path': 'new_mnist',
#     'checkpoint_format': '{epoch:02d}-{val_loss:.2f}.h5',
# }

exp = {
    'dataset': 'stanford_dogs',  # 'mnist',
    'im_shape': (32, 32),  # (28, 28),
    'channels': 3,
    'latent_dim': 64,
    'batch_size': 16,
    'epochs': 10,
    'beta': 1.0,
    'model_path': 'new_dogs',
    'checkpoint_format': '{epoch:02d}-{val_loss:.2f}.h5',
}


# Load and preprocess the data
train, test, info = get_data(
    batch_size=exp['batch_size'],
    im_shape=exp['im_shape'],
    dataset=exp['dataset'],
)


# Create, train and evaluate the model
vae = get_model(exp)
vae.fit(
    train,
    epochs=exp['epochs'],
    steps_per_epoch=math.ceil(
        info.splits["train"].num_examples / exp['batch_size']),
    validation_data=test,
    validation_steps=math.ceil(
        info.splits["test"].num_examples / exp['batch_size']),
    callbacks=[
        # Log validation losses
        tf.keras.callbacks.CSVLogger(
            filename=path.join(exp['model_path'], 'losses.csv'),
            append=True
        ),
        # Create model checkpoints (Skip during exploratory testing)
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=path.join(exp['model_path'], exp['checkpoint_format']),
        #     monitor='val_loss',
        #     save_freq='epoch',
        #     verbose=1,
        # ),
    ],
)


plot_independent_grid(vae.decoder, 2, 1, (28, 28))
plot_all_2D_manifolds(vae.decoder, 2, 1, (28, 28))


show_reconstructed_digit(vae)
