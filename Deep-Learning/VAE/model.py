"""Define the VAE model, saving, loading, layers and custom callbacks."""
import os
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Add, Dense, Input, Lambda, Layer, Multiply
from tensorflow.keras.models import Model, Sequential


def nll(y_true, y_pred):
    """Negative log likelihood (Bernoulli)."""
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """Identity transform layer that adds KL divergence to the final loss."""

    def __init__(self, beta, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        # Multiply KLD component with Beta factor
        kl_batch *= self.beta

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class CustomCallback(Callback):
    """Descends from Callback."""

    def __init__(self, path, checkpoint, encoder, decoder):
        self.model_path = path
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.decoder = decoder

    def on_epoch_end(self, epoch, logs=None):
        # Checkpoint - Save every X epochs
        if self.checkpoint == 0:
            return

        if epoch % self.checkpoint == 0:
            print("\nSaving the model at epoch: ", epoch+1)
            save_model(
                vae=self.model,
                encoder=self.encoder,
                decoder=self.decoder,
                model_path=self.model_path,
            )

    # Catch if no callbacks are enabled
    lambda *_, **__: None


def gpu_configuration():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Dynamically allocate GPU memory use
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    except (ValueError, RuntimeError, IndexError):
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def load_model(
            original_dim,
            interm_dim,
            latent_dim,
            epochs,
            epsilon_std,
            beta,
            model_path,
            train
        ):
    try:
        # If there is already a model at the given folder, try resuming
        assert os.path.isdir(model_path), f'No directory at {model_path}'

        # Load the en/decoder, create the VAE and load its weights
        encoder = tf.keras.models.load_model(
            os.path.join(model_path, 'encoder')
        )
        decoder = tf.keras.models.load_model(
            os.path.join(model_path, 'decoder')
        )
        vae, _, _ = make_model(
            original_dim=original_dim,
            interm_dim=interm_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            epsilon_std=epsilon_std,
            beta=beta,
        )
        vae.compile(optimizer='rmsprop', loss=nll)

        # Load the state of the old model
        vae.load_weights(os.path.join(model_path, 'vae', ""))
        print(f"\nResuming from loaded model at {model_path}\n")

    except AssertionError as e:
        print(e)
        print(f"\nCreating a new model at {model_path}\n")
        return None, None, None

    except (ImportError, IOError, ValueError) as e:
        print(e)
        print(f"\nTried resuming from {model_path} but something went wrong\n")
        sys.exit()

    return vae, encoder, decoder


def save_model(
            vae,
            encoder,
            decoder,
            model_path
        ):
    # Reset metrics before saving so that loaded model has same state,
    # since metric states are not preserved by Model.save_weights
    vae.reset_metrics()
    # Save the model
    vae.save_weights(os.path.join(model_path, 'vae', ""), save_format='tf')
    encoder.save(os.path.join(model_path, 'encoder'))
    decoder.save(os.path.join(model_path, 'decoder'))


def make_model(
            original_dim,
            interm_dim,
            latent_dim,
            epochs,
            epsilon_std,
            beta
        ):
    """Define the Variational Autoencoder model and return it."""
    # Define the Decoder (Latent --> Reconstructed Image)
    decoder = Sequential([
        Dense(int(interm_dim / 2), input_dim=latent_dim, activation='relu'),
        Dense(interm_dim, input_dim=int(interm_dim / 2), activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    # Define the Encoder (Original Image --> Latent)
    # Input of the encoder
    x = Input(shape=(original_dim, ))
    # Hidden layer
    h1 = Dense(interm_dim, activation='relu')(x)
    h2 = Dense(int(interm_dim / 2), activation='relu')(h1)
    # Mean and log variance (output of the encoder)
    z_mu = Dense(latent_dim)(h2)
    z_log_var = Dense(latent_dim)(h2)

    z_mu, z_log_var = KLDivergenceLayer(beta=beta)([z_mu, z_log_var])
    # Normalize the log variance to std dev
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = K.random_normal(
        stddev=epsilon_std,
        shape=(K.shape(x)[0], latent_dim),
        seed=42
    )
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)

    # Combine the Encoder and Decoder to define the VAE
    vae = Model(x, x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)

    encoder = Model(x, z_mu)

    return vae, encoder, decoder
