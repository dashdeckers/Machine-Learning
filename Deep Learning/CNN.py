# import numpy as np
import argparse
import datetime

import tensorflow as tf
from experiments import (AlexNet, minimal_model, options, small_model,  # noqa
                         standard_experiment)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser(description='Deep Learning with CIFAR10')
parser.add_argument(
    'name',
    type=str,
    help='an appropriate name to record the logs to'
)
parser.add_argument(
    '--opt',
    default=options['optimizers'][0],
    choices=options['optimizers'],
    type=str,
    help='the optimizer to use (default: %(default)s)'
)
parser.add_argument(
    '--loss',
    default=options['losses'][0],
    choices=options['losses'],
    type=str,
    help='the loss to minimize (default: %(default)s)'
)
parser.add_argument(
    '--model',
    default=options['models'][0],
    choices=options['models'],
    type=str,
    help='the model architecture to use (default: %(default)s)'
)
parser.add_argument(
    '--activation',
    default=options['activations'][0],
    choices=options['activations'],
    type=str,
    help='the activation function to use (default: %(default)s)'
)
parser.add_argument(
    '--dropout',
    default=options['dropouts'][0],
    choices=options['dropouts'],
    type=str,
    help='the dropout probability (default: %(default)s)'
)
parser.add_argument(
    '--epochs',
    default=50,
    type=int,
    help='the number of epochs to run for (integer, default: %(default)s)'
)

args = parser.parse_args()
print(args)


# TensorBoard stuff
logdir = "logs/" + datetime.datetime.now().strftime("%d-%m %H:%M:%S")
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    profile_batch=0,
)


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
    model.compile(
        loss=experiment['loss'],
        optimizer=experiment['optimizer'],
        metrics=experiment['metrics']
    )
    model.build(x_train.shape[1:])
    model.summary()

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=experiment['batch_size'],
        epochs=experiment['epochs'],
        verbose=verbose,
        validation_split=0.1,
        callbacks=[tensorboard]
    )

    return model


if __name__ == '__main__':
    pass
    # run_experiment(standard_experiment, small_model)
