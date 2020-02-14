import numpy as np
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)

# Seed the rng's for reproducible results
seed = 1000
np.random.seed(seed)


def build_experiment(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
            batch_size=64,
            epochs=50
        ):
    return {
        'loss': loss,
        'optimizer': optimizer,
        'metrics': metrics,
        'batch_size': batch_size,
        'epochs': epochs,
    }


def build_model(architecture='cnn', dropout='medium', activation='relu'):
    # Valid architectures are:
        # 'cnn', 'short cnn', 'long cnn', 'small model' & 'minimal model'
    # Valid dropouts are:
        # 'none', 'medium', and 'high'
    # Valid activations are:
        # 'relu', 'elu', 'selu', 'linear', 'softplus', 'sigmoid'

    if dropout == 'medium':
        conv_drop = 0.25
        dense_drop = 0.5
    elif dropout == 'none':
        conv_drop = 0
        dense_drop = 0
    elif dropout == 'high':
        conv_drop = 0.5
        dense_drop = 0.85

    if architecture == 'cnn':
        # Keras CNN model intended for CIFAR
        # https://keras.io/examples/cifar10_cnn/
        model = [
            # First group
            Conv2D(
                filters=32,
                input_shape=(32, 32, 3),
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+1
            ),

            # Second group
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+2
            ),

            # Dense
            Flatten(),
            Dense(
                units=512,
                activation=activation
            ),
            Dropout(
                dense_drop,
                seed=seed+3
            ),

            Dense(
                units=10,
                activation='softmax'
            ),
        ]
    if architecture == 'long cnn':
        model = [
            # First group
            Conv2D(
                filters=32,
                input_shape=(32, 32, 3),
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+1
            ),

            # Second group
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+1
            ),

            # Third group
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+2
            ),

            # Fourth group
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+2
            ),

            # Dense
            Flatten(),
            Dense(
                units=512,
                activation=activation
            ),
            Dropout(
                dense_drop,
                seed=seed+3
            ),

            Dense(
                units=10,
                activation='softmax'
            ),
        ]
    if architecture == 'short cnn':
        model = [
            # First group
            Conv2D(
                filters=32,
                input_shape=(32, 32, 3),
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                activation=activation,
            ),
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='valid',
                activation=activation,
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(1, 1)
            ),
            Dropout(
                conv_drop,
                seed=seed+2
            ),

            # Dense
            Flatten(),
            Dense(
                units=512,
                activation=activation
            ),
            Dropout(
                dense_drop,
                seed=seed+3
            ),

            Dense(
                units=10,
                activation='softmax'
            ),
        ]
    if architecture == 'small model':
        model = [
            Conv2D(
                filters=96,
                input_shape=(32, 32, 3),
                kernel_size=(2, 2),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            ),
            Flatten(),
            Dense(
                units=10,
                activation='softmax'
            ),
        ]
    if architecture == 'minimal model':
        model = [
            Flatten(),
            Dense(
                units=10,
                activation='softmax'
            )
        ]
    return model
