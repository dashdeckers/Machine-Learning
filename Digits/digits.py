"""A Linear Regression baseline algorithm for predicting handwritten digits."""

import multiprocessing as mp
import pickle as pkl
from functools import partial
from random import gauss

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(filename='mfeat-pix.txt'):
    """Load the data, make the labels."""
    data = np.loadtxt(filename)

    labels = np.zeros(data.shape[0], dtype=np.int)
    digitreps = int(data.shape[0] / 10)
    for digit in range(10):
        labels[digit * digitreps: (digit + 1) * digitreps] = digit

    return data, labels


def results(model, data, labels, show_list=[]):
    """Show and/or return the results of a model on some data given some labels.

    Print metrics by passing the names of the metrics via show_list like so:
    show_list=['prec', 'rec', 'F1', 'acc', 'MR', 'matrix', 'report']

    Always returns the following metrics:
    ['prec', 'rec', 'F1', 'acc', 'MR']

    """
    # Don't allow invalid metrics in show_list
    valid_metrics = ['prec', 'rec', 'F1', 'acc', 'MR', 'matrix', 'report']
    assert all(metric in valid_metrics for metric in show_list)

    # Gather the data
    pred = model.predict(data)  # This is the line that takes so long...
    precs, recs, F1s, _ = precision_recall_fscore_support(labels, pred)
    acc = accuracy_score(labels, pred)

    if show_list:
        # Helper function to print the mean and standard deviation of a list
        def show_mean_std(n, l):
            print(f'{n.title()}: mean={np.mean(l):.3f}, std={np.std(l):.3f}')

        # Possible metrics as partial functions
        show_dict = {
            'prec': partial(show_mean_std, 'precision', precs),
            'rec': partial(show_mean_std, 'recall', recs),
            'F1': partial(show_mean_std, 'f1-score', F1s),
            'acc': partial(print, f'Accuracy: {acc:.3f}'),
            'MR': partial(print, f'Misclassification Rate: {1 - acc:.3f}'),
            'matrix': partial(confusion_matrix, labels, pred),
            'report': partial(classification_report, labels, pred)
        }

        for metric in show_list:
            show_dict[metric]()

    return {
        'prec': [np.mean(precs), np.std(precs)],
        'rec': [np.mean(recs), np.std(recs)],
        'F1': [np.mean(F1s), np.std(F1s)],
        'acc': acc,
        'MR': 1 - acc,
    }


def clamp(value, minimum, maximum):
    """Clamp a value between a min and a max."""
    if minimum > value:
        value = minimum
    if maximum < value:
        value = maximum
    return value


def add_noise(x_train, y_train, spread=0, copies=1, keep_original=False):
    """Generate more data by adding noise.

    This function takes data = (x_train, y_train) and generates more data by
    making copies of the data with added Gaussian noise and the same label.
    """
    x_noise = []
    y_noise = []
    for digit, label in zip(x_train, y_train):
        # Keep the original if we want to
        if keep_original:
            x_noise.append(digit)
            y_noise.append(label)

        # Create copies with added noise
        for c in range(copies):
            # Every pixel is given a Gaussian noise component
            x_noise.append([clamp(pixel + gauss(0, spread), 0, 6)
                            for pixel in digit])
            y_noise.append(label)

    return x_noise, y_noise


def cross_val(pipeline, data, labels,
              n_splits=10, n_repeats=1,
              noise_spread=0, noise_copies=1):
    """Perform cross-validation while adding noise.

    This function brings the noise generation to the cross-validation to make
    sure that we are using noisy data for training only and not for testing.

    After each split, expand the training data by adding noise and then
    evaluate and record the performance.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=1,
    )

    # Let the CV split the data correctly, the correct number of times
    perf = list()
    for train_indices, test_indices in cv.split(data, labels):
        x_train, x_test = data[train_indices], data[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]

        # Expand the data by adding noise to (x_train, y_train)
        x_train, y_train = add_noise(x_train, y_train,
                                     noise_spread, noise_copies)

        # Fit the model and collect the results (its performance)
        pipeline.fit(x_train, y_train)
        perf.append(results(pipeline, x_test, y_test))

    return perf


def execute_thread(pipeline, data, labels, params, classifier):
    """Delegate a set of params to multiprocessing."""
    if classifier == 'LR':
        pipeline.set_params(**{
            'pca__n_components': params['m'],
            'model__alpha': params['alpha'],
        })
    elif classifier == 'KNN':
        pipeline.set_params(**{
            'pca__n_components': params['m'],
            'model__n_neighbors': params['K'],
        })
    else:
        assert classifier in ['LR', 'KNN']

    print(params)

    perf = cross_val(pipeline, data, labels,
                     noise_spread=params['noise_spread'],
                     noise_copies=params['noise_copies'])

    return params, perf


def perform_experiment(exp, filename):
    """Perform the full experiment.

    Does a full parameter sweep for the experiment passed as a dictionary and
    saves the result to file. The experiment exp should define a pipeline, the
    parameters to sweep for, and give the name of the classifier.
    """
    # Load the data and labels
    data, labels = load_data()

    # Generate all the arguments for function calls in this parameter sweep
    args = [(exp['pipeline'], data, labels, params, exp['classifier'])
            for params in exp['parameters']]

    # Use multiprocessing to delegate functions calls across processors
    print(f'Delegating {len(args)} tasks to {mp.cpu_count()} cores')
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(execute_thread, args)

    # Save results to file
    with open(filename, 'wb') as f:
        pkl.dump(results, f)

    return results


experiment_LR = {
    # Set the classifier name (must be one of {'LR', 'KNN'})
    'classifier': 'LR',

    # Define the pipeline
    'pipeline': Pipeline([
        ('normalize', StandardScaler()),
        ('pca', PCA()),
        ('model', RidgeClassifier()),
    ]),

    # Generate the parameter combinations to sweep for
    'parameters': list(ParameterGrid({
        'm': list(range(20, 51)),
        'alpha': [0],
        'noise_spread': np.arange(0, 3.25, 0.25),
        'noise_copies': [10],
    })),
}

baby_experiment = {
    # Set the classifier name (must be one of {'LR', 'KNN'})
    'classifier': 'KNN',

    # Define the pipeline
    'pipeline': Pipeline([
        ('normalize', StandardScaler()),
        ('pca', PCA()),
        ('model', KNeighborsClassifier()),
    ]),

    # Generate the parameter combinations to sweep for
    'parameters': list(ParameterGrid({
        'm': [30],
        'K': [3, 5],
        'noise_spread': np.arange(1, 2.25, 0.5),
        'noise_copies': [10],
    })),
}


if __name__ == '__main__':
    perform_experiment(baby_experiment, 'trash_file')
