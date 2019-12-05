from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import (
    Activation,
    Flatten,
    Dense,
)
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Define model type and input shape
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, width, height)

        # Define the first and only: CONV => RELU
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))

        # Make it a softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

