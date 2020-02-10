# import numpy as np
import os

import tensorflow as tf
from experiments import minimal_model, small_model, standard_experiment  # noqa
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical

# Turn off Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load data, one-hot-encode labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)

x_train = (x_train.astype('float32') / 255)
x_test = (x_test.astype('float32') / 255)


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
