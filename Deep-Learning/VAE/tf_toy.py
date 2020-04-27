import argparse
import glob
import json
import math
import os

import numpy as np
import tensorflow as tf
from data import get_data
from plot import plot_all_2D_manifolds, plot_independent_grid
from tensorflow.keras import layers

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('name', type=str)
args = parser.parse_args()

checkpoint_format = '{epoch:02d}-{val_loss:.2f}.h5'

# Define the experiment
exp = {
    'dataset': 'mnist',  # 'stanford_dogs'
    'input_shape': (1, 28, 28, 1),  # (1, 32, 32, 3)
    'batch_size': 64,
    'epochs': 10,

    'latent_dim': 2,
    'beta': 1.0,
}
exp['im_shape'] = exp['input_shape'][1:3]
exp['channels'] = exp['input_shape'][3]
exp['col_dim'] = np.prod(exp['input_shape'][1:])

exp['encoder_layers'] = [
    layers.Conv2D(
        input_shape=exp['input_shape'],
        data_format='channels_last',
        filters=32,
        kernel_size=2,
        padding='same',
        activation='relu',
        name='Conv-1',
    ),
    # layers.Conv2D(
    #     filters=64,
    #     kernel_size=2,
    #     padding='same',
    #     activation='relu',
    #     strides=(2, 2),
    #     name='Conv-2',
    # ),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dense(1024, activation='relu'),
    # layers.Dense(256, activation='relu'),
]

exp['decoder_layers'] = [
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=exp['col_dim'], activation='relu'),
    layers.Reshape(target_shape=exp['im_dim']),
    # layers.Conv2DTranspose(
    #     filters=64,
    #     kernel_size=3,
    #     strides=(1, 1),
    #     padding='same',
    #     activation='relu',
    #     name='DeConv-1',
    # ),
    # layers.Conv2DTranspose(
    #     filters=self.im_dim[3],
    #     kernel_size=3,
    #     strides=(1, 1),
    #     padding='same',
    #     activation='relu',
    #     name='DeConv-2',
    # ),
]


# Create a new project folder and write the experiment info to it
if not os.path.exists(args.name):
    print('Creating a new model at', args.name, '\n')
    os.makedirs(args.name)
    with open(os.path.join(args.name, 'experiment.json'), 'w') as file:
        file.write(json.dumps(exp, indent=4))


# Setup GPU options
gpu_configuration()


# Load and preprocess the data
train, test, info = get_data(
    batch_size=exp['batch_size'],
    im_shape=exp['im_shape'],
    dataset=exp['dataset'],
)


# Create and maybe resume the model
vae = VariationalAutoEncoder(exp)
vae.compile(
    optimizer=tf.keras.optimizers.Adam(),
    # optimizer='rmsprop',
)
if args.resume:
    # Grab the newest .h5 model checkpoint file, if it exists
    list_of_saved_models = glob.glob(os.path.join(args.name, '*.h5'))
    latest_model = max(list_of_saved_models, key=os.path.getctime)

    # Load the latest checkpoint model
    print('Loading the model from checkpoint', latest_model, '\n')
    vae(np.zeros(shape=exp['input_shape']))  # create weights
    vae.load_weights(latest_model)


# Train and evaluate the model
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
            filename=os.path.join(args.name, 'losses.csv'),
            append=True
        ),
        # Create model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.name, checkpoint_format),
            monitor='val_loss',
            save_freq='epoch',
            period=5,
            verbose=1,
        ),
    ],
)


plot_independent_grid(
    vae.decoder,
    exp['latent_dim'],
    exp['channels'],
    exp['im_shape']
)
plot_all_2D_manifolds(
    vae.decoder,
    exp['latent_dim'],
    exp['channels'],
    exp['im_shape']
)
