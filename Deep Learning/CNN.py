# import numpy as np
import datetime
import os

import tensorflow as tf
from experiments import build_model  # noqa
from experiments import standard_experiment
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Turn off Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TensorBoard stuff
logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def load_data(preprocess=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if preprocess:
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()


def run_experiment(experiment, model, verbose=1):
    model = Sequential(model)
    model.compile(loss=experiment['loss'],
                  optimizer=experiment['optimizer'],
                  metrics=experiment['metrics'])
    model.build(x_train.shape[1:])
    model.summary()

    model.fit(x=x_train,
              y=y_train,
              batch_size=experiment['batch_size'],
              epochs=experiment['epochs'],
              verbose=verbose,
              validation_split=0.1,
              callbacks=[tensorboard_callback])

    return model


if __name__ == '__main__':
    run_experiment(standard_experiment, build_model(dropout='none'))
