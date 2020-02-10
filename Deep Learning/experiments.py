from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D  # noqa

# Define experiments
standard_experiment = {
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': ['accuracy'],
    'batch_size': 64,
    'epochs': 10,
}


# Define models
small_model = [
    Conv2D(filters=96,
           input_shape=(32, 32, 3),
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
