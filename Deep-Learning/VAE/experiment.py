import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
            'conv2d': layers.Conv2D,
            'max': layers.MaxPooling2D,
            'dropout': layers.Dropout,
            'up': layers.UpSampling2D,
            'deconv2d': layers.Conv2DTranspose,

            'optimizer': tf.keras.optimizers.Adam,

            # Add more layer functions here as needed
        }
        # Get experiment from file
        with open(os.path.join(project_name, 'experiment.json')) as file:
            exp = json.load(file)

        # Convert layer configs into actual layers
        exp['optimizer'] = funcs['optimizer'].from_config(exp['optimizer'])
        for enc_dec in ['encoder_layers', 'decoder_layers']:
            for idx, config in enumerate(exp[enc_dec]):
                name = config['name'].split('_')
                if len(name) > 1 and name[1] == 'transpose':
                    exp[enc_dec][idx] = funcs['deconv2d'].from_config(config)
                else:
                    exp[enc_dec][idx] = funcs[name[0]].from_config(config)

        return exp

    # Define the experiment
    exp = {
        'project_name': project_name,
        'dataset': 'mnist',  # 'stanford_dogs'
        'input_shape': (1, 28, 28, 1),  # (1, 32, 32, 3)
        'batch_size': 64,
        'epochs': 1,

        'latent_dim': 2,
        'alpha': 1.0,
        'beta': 2.0,
        'gamma': 1.0,
        'distribution': 'gaussian',

        'optimizer': tf.keras.optimizers.Adam(),
    }
    exp['im_shape'] = exp['input_shape'][1:3]
    exp['channels'] = exp['input_shape'][3]
    exp['col_dim'] = int(np.prod(exp['input_shape'][1:]))

    # Define the architecture
    exp['encoder_layers'] = [
        layers.Conv2D(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            input_shape=exp['input_shape'],
            activation='relu',
            data_format='channels_last'
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            activation='relu',
            data_format='channels_last'
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(3, 3),
            activation='relu',
            data_format='channels_last'
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(
            units=exp['input_shape'][1] * 2,
            activation='relu'
        ),
        layers.Dropout(0.5),
        layers.Dense(
            units=exp['latent_dim'],
            activation='softmax'
        ),
    ]

    exp['decoder_layers'] = [
        layers.Dense(
            units=exp['latent_dim'],
            activation='softmax'
        ),
        layers.Dense(
            units=int(exp['latent_dim'] * 2),  # * instead of /
            activation='relu'
        ),
        layers.Dense(
            units=int(exp['latent_dim'] * 4),  # because why /?
            activation='relu'
        ),
        layers.Dense(
            units=256,
            activation='relu'
        ),
        layers.Reshape(target_shape=(2, 2, int(256 / 4))),  # -1 doesn't work
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            activation='relu',
            data_format='channels_last'
        ),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            activation='relu',
            data_format='channels_last'
        ),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2D(
            filters=exp['channels'],  # changed this to Conv2D (idk, but works)
            kernel_size=(1, 1),
            activation='relu',
            data_format='channels_last'
        ),
        layers.Reshape(target_shape=exp['input_shape'][1:]),
    ]
    return exp
