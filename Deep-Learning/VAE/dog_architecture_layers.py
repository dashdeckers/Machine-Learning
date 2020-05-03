#  https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
#  Encoder
exp['encoder_layers'] = [       # loss = binary crossentropy
        # Dog breed image classification from:
        # https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
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
#  Decoder
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
            units=int(exp['latent_dim'] * 4),  # because why /? #Good catch!
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
            kernel_size=(4, 4),
            activation='relu',
            data_format='channels_last'
        ),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2DTranspose(
            filters=exp['channels'],
            kernel_size=(3, 3),
            activation='relu',
            data_format='channels_last'
        ),
        layers.Reshape(target_shape=exp['input_shape'][1:]),
    ]

# Architecture 3 - Bigger Images
# Define the experiment
    exp = {
        'project_name': project_name,
        'dataset': 'stanford_dogs',  # 'mnist'
        'input_shape': (1, 64, 64, 3),  # (1, 28, 28, 1)
        'batch_size': 64,
        'epochs': 100,

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
        layers.Conv2D(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            input_shape=exp['input_shape'],
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2D(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2D(
            filters=exp['input_shape'][1] * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Flatten(),
        layers.Dense(
            units=exp['input_shape'][1] * 4,
            activation='relu'
        ),
    ]

    exp['decoder_layers'] = [
        layers.Dense(
            units=exp['input_shape'][1] * 4,
            activation='relu'
        ),
        layers.Dense(
            units=16384,
            activation='relu'
        ),
        layers.Reshape(target_shape=(8, 8, 256)),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2DTranspose(
            filters=exp['channels'],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
    ]

#  Architecture 4, The Big Boy Edition
    exp = {
        'project_name': project_name,
        'dataset': 'stanford_dogs',  # 'mnist'
        'input_shape': (1, 128, 128, 3),  # (1, 28, 28, 1)
        'batch_size': 64,
        'epochs': 500,

        'latent_dim': 120,
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
        layers.Conv2D(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            input_shape=exp['input_shape'],
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2D(
            filters=exp['input_shape'][1],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2D(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2D(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Flatten(),
        layers.Dense(
            units=exp['input_shape'][1] * 4,
            activation='relu'
        ),
    ]

    exp['decoder_layers'] = [
        layers.Dense(
            units=exp['input_shape'][1] * 4,
            activation='relu'
        ),
        layers.Dense(
            units=16384,
            activation='relu'
        ),
        layers.Reshape(target_shape=(8, 8, 256)),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1] * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2DTranspose(
            filters=exp['input_shape'][1],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
        layers.Conv2DTranspose(
            filters=exp['channels'],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='relu',
            data_format='channels_last'
        ),
    ]