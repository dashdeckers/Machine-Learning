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
        'dataset': 'stanford_dogs',  # 'mnist'
        'input_shape': (1, 32, 32, 3),  # (1, 28, 28, 1)
        'batch_size': 64,
        'epochs': 50,

        'latent_dim': 20,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'distribution': 'gaussian',

        'optimizer': tf.keras.optimizers.Adam(),
    }
    exp['im_shape'] = exp['input_shape'][1:3]
    exp['channels'] = exp['input_shape'][3]
    exp['col_dim'] = int(np.prod(exp['input_shape'][1:]))

    # Define the architecture
    exp['encoder_layers'] = [
        layers.Flatten(),
        layers.Dense(
            units=int(6144 * 0.75),
            activation='relu',
        ),
        layers.Dense(
            units=int(6144 * 0.5),
            activation='relu',
        ),
        layers.Dense(
            units=int(6144 * 0.25),
            activation='relu',
        ),
    ]

    exp['decoder_layers'] = [
        layers.Dense(
            units=int(6144 * 0.25),
            activation='relu',
        ),
        layers.Dense(
            units=int(6144 * 0.5),
            activation='relu',
        ),
        layers.Dense(
            units=int(6144 * 0.75),
            activation='relu',
        ),
        layers.Dense(
            units=int(6144),
            activation='relu',
        ),
        layers.Reshape(target_shape=exp['input_shape'][1:]),
    ]
    return exp
