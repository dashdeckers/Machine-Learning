import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

# Seed the rng's for reproducible results
seed = 1000
np.random.seed(seed)

# Define experiments
standard_experiment = {
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': ['accuracy'],
    'batch_size': 64,
    'epochs': 10,
}


# Define models

# Original AlexNet model
# Intended for 224x224 images, does not work with CIFARs 32x32
AlexNet = [
    # 1st Conv Layer
    Conv2D(filters=96,
           input_shape=(224, 224, 3),
           kernel_size=(11, 11),
           strides=(4, 4),
           padding='valid',
           activation='relu'),

    MaxPooling2D(pool_size=(3, 3),
                 strides=(2, 2),
                 padding='valid'),

    # 2nd Conv Layer
    Conv2D(filters=256,
           kernel_size=(5, 5),
           strides=(1, 1),
           padding='valid',
           activation='relu'),

    MaxPooling2D(pool_size=(3, 3),
                 strides=(2, 2),
                 padding='valid'),

    # 3rd Conv Layer
    Conv2D(filters=384,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='valid',
           activation='relu'),

    # 4th Conv Layer
    Conv2D(filters=384,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='valid',
           activation='relu'),

    # 5th Conv Layer
    Conv2D(filters=256,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='valid',
           activation='relu'),

    MaxPooling2D(pool_size=(3, 3),
                 strides=(2, 2),
                 padding='valid'),

    # Flatten
    Flatten(),

    # 1st Fully Connected Layer
    Dense(units=4096,
          input_shape=(224*224*3,),
          activation='relu'),
    Dropout(rate=0.5,
            seed=seed+1),

    # 2nd Fully Connected Layer
    Dense(units=4096,
          activation='relu'),
    Dropout(rate=0.5,
            seed=seed+2),

    # 3rd Fully Connected Layer
    Dense(units=4096,
          activation='relu'),
    Dropout(rate=0.5,
            seed=seed+3),

    # Output Layer
    Dense(units=10,
          activation='softmax')
]

# Keras CNN model intended for CIFAR
# https://keras.io/examples/cifar10_cnn/
cnn = [
    # First group
    Conv2D(filters=32,
           input_shape=(32, 32, 3),
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),
    Dropout(0.25,
            seed=seed+1),

    # Second group
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),
    Dropout(0.25,
            seed=seed+2),

    # Dense
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5,
            seed=seed+3),

    Dense(10, activation='softmax'),
]

# CNN model adapted to remove all dropout
# Can be used to compare against cnn and cnn_high_dropout
cnn_no_dropout = [
    # First group
    Conv2D(filters=32,
           input_shape=(32, 32, 3),
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),

    # Second group
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),

    # Dense
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax'),
]


# CNN model adapted to increase dropout
# Can be used to compare against cnn and cnn_no_dropout
cnn_high_dropout = [
    # First group
    Conv2D(filters=32,
           input_shape=(32, 32, 3),
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),
    Dropout(0.5,
            seed=seed+1),

    # Second group
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           ),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='valid',
           activation='relu',
           ),
    MaxPooling2D(
        pool_size=(2, 2),
        stride=(1, 1)),
    Dropout(0.5,
            seed=seed+2),

    # Dense
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.85,
            seed=seed+3),

    Dense(10, activation='softmax'),
]

small_model = [
    Conv2D(filters=96,
           input_shape=(224, 224, 3),
           kernel_size=(2, 2),
           strides=(1, 1),
           padding='same',
           activation='relu'),

    MaxPooling2D(pool_size=(2, 2),
                 strides=(2, 2),
                 padding='valid'),

    Flatten(),

    Dense(units=10,
          activation='softmax')
]

minimal_model = [
    Flatten(),

    Dense(units=10,
          activation='softmax')
]
