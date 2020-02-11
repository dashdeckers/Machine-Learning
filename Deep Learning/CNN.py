# import numpy as np
import os
import pickle

import tensorflow as tf
from experiments import minimal_model, small_model, standard_experiment  # noqa
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.backend import resize_images

# Turn off Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_cifar(enlarged=True):
    if enlarged:
        try:
            file = open("cifar224.pkl", 'rb')
            (x_train, y_train), (x_test, y_test) = pickle.load(file)
            file.close()
        except Exception:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # One hot encode
            y_test = to_categorical(y_test, 10)
            y_train = to_categorical(y_train, 10)

            # Normalize
            x_train = (x_train.astype('float16') / 255)
            x_test = (x_test.astype('float16') / 255)

            # Scale 32x32 to 244x244
            x_train = resize_images(x_train, height_factor=7, width_factor=7,
                                    data_format="channels_last").numpy()
            x_test = resize_images(x_test, height_factor=7, width_factor=7,
                                   data_format="channels_last").numpy()

            file = open("cifar224.pkl", 'wb')
            pickle.dump(((x_train, y_train), (x_test, y_test)), file)
            file.close()
    else:
        # Load data, one-hot-encode labels
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)

        # Normalize
        x_train = (x_train.astype('float32') / 255)
        x_test = (x_test.astype('float32') / 255)

    return (x_train, y_train), (x_test, y_test)


# May want to move this elsewhere
(x_train, y_train), (x_test, y_test) = load_cifar(enlarged=True)


def run_experiment(experiment, architecture, verbose=1):
    model = Sequential(architecture)
    model.compile(loss=experiment['loss'],
                  optimizer=experiment['optimizer'],
                  metrics=experiment['metrics'])
    model.build(x_train.shape[1:])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=experiment['batch_size'],
              epochs=experiment['epochs'],
              verbose=verbose)
    print(model.evaluate(x_test, y_test, verbose=verbose))
    return model


if __name__ == '__main__':
    run_experiment(standard_experiment, small_model)
