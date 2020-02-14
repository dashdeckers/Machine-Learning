# import numpy as np
import datetime

import tensorflow as tf
from experiments import build_experiment, build_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

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
    run_experiment(build_experiment(), build_model(dropout='none'))
1
