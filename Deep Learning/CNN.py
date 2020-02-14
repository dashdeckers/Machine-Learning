import argparse
from datetime import datetime

import tensorflow as tf
from experiments import build_model, options
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Argument Parsing
parser = argparse.ArgumentParser(
    description='Deep Learning with CIFAR10',
    epilog='The default values for optional flags are the first choices,'
    + ' the --epoch flag must be an integer, and the name should be meaningful'
    + ' so that the results of this model can be easily found in TensorBoard.'
    + ' To see the data in TensorBoard, run `tensorboard --logdir logs` in a'
    + ' separate terminal from the same directory as you ran this file and'
    + ' then visit `localhost:6006` in your browser.',
)
parser.add_argument(
    'name',
    type=str,
    help='an appropriate name to record the logs to'
)
parser.add_argument(
    '--optimizer',
    default=options['optimizers'][0],
    choices=options['optimizers'],
    type=str,
)
parser.add_argument(
    '--loss',
    default=options['losses'][0],
    choices=options['losses'],
    type=str,
)
parser.add_argument(
    '--architecture',
    default=options['architectures'][0],
    choices=options['architectures'],
    type=str,
)
parser.add_argument(
    '--activation',
    default=options['activations'][0],
    choices=options['activations'],
    type=str,
)
parser.add_argument(
    '--dropout',
    default=options['dropouts'][0],
    choices=options['dropouts'],
    type=str,
)
parser.add_argument(
    '--epochs',
    default=50,
    type=int,
)
args = parser.parse_args()
print(args)


# TensorBoard stuff
logdir = f'logs/{args.name}' + datetime.now().strftime(" %d-%m %H:%M:%S")
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    profile_batch=0,
)


def load_data(preprocess=True):
    """Load the data and do some basic preprocessing."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if preprocess:
        # Normalize the data
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # One-hot encode the labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def run_experiment(architecture, verbose=1):
    """Load the data, define the model architecture, run the experiment."""
    (x_train, y_train), (x_test, y_test) = load_data()
    model = Sequential(architecture)
    model.compile(
        loss=args.loss,
        optimizer=args.optimizer,
        metrics=['accuracy', 'categorical_accuracy']
    )
    model.build(x_train.shape[1:])
    model.summary()

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=args.epochs,
        verbose=verbose,
        validation_split=0.1,
        callbacks=[tensorboard]
    )

    # HERE: evaluate the model on the lockboxed (x_test, y_test)

    return model


if __name__ == '__main__':
    run_experiment(build_model(
        architecture=args.architecture,
        dropout=args.dropout,
        activation=args.activation,
    ))
