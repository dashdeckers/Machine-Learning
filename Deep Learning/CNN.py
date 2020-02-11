# import numpy as np
import os

import numpy as np
import tensorflow as tf
from experiments import minimal_model, small_model, standard_experiment  # noqa
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import Sequence, to_categorical
from skimage.transform import resize

# Turn off Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CIFAR10(Sequence):
    def __init__(self, X, y, batch_size, preprocess=True, rescale=True):
        self.X, self.y = X, y
        self.rescale = rescale
        self.batch_size = batch_size

        if preprocess:
            self.X = (X.astype('float32') / 255)
            self.y = to_categorical(y, 10)

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return (
            np.array([resize(img, (224, 224)) for img in batch_x]),
            np.array(batch_y)
        )


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def run_experiment(experiment, model, verbose=1):
    model = Sequential(model)
    model.compile(loss=experiment['loss'],
                  optimizer=experiment['optimizer'],
                  metrics=experiment['metrics'])
    model.build(x_train.shape[1:])
    model.summary()

    batch_size = experiment['batch_size']

    train_generator = CIFAR10(x_train, y_train, batch_size=batch_size)
    test_generator = CIFAR10(x_test, y_test, batch_size=batch_size)

    model.fit_generator(train_generator,
                        epochs=experiment['epochs'],
                        verbose=verbose,
                        validation_data=test_generator,
                        workers=6,
                        use_multiprocessing=True)

    print(model.evaluate_generator(test_generator,
                                   verbose=verbose,
                                   workers=6,
                                   use_multiprocessing=True))
    return model


if __name__ == '__main__':
    run_experiment(standard_experiment, small_model)
