"""A Linear Regression baseline algorithm for predicting handwritten digits."""

import pathlib
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def timestamp(start_time):
    """Return the elapsed time in seconds since start_time."""
    return str(round(time.time() - start_time, 3)) + 's'


def show_digit(digit, col_vector=False):
    """Show the digit as an image.

    Usage:
        digit = data[:,0] # get column vector
        digit = digit.reshape(16, 15) # resize into 2D
        show_digit(digit)

        digit = data[:,0]
        show_digit(digit, col_vector=True)

    """
    if col_vector:
        assert digit.ndim == 1, (digit, digit.shape)
        assert digit.shape == (240,), (digit, digit.shape)
        digit = digit.reshape(16, 15)
    else:
        assert digit.ndim == 2, (digit, digit.shape)
        assert digit.shape == (16, 15), (digit, digit.shape)

    plt.imshow(digit, cmap='Greys')
    plt.show()


def label_data(datafile):
    # Load the transposed datafile to get each image in a col vector
    data = np.loadtxt(datafile).T  # (240, 100c)

    # Label every 10% as a new digit
    digitreps = int(data.shape[1] / 10)

    labels = np.zeros(data.shape[1], dtype=np.int)
    for digit in range(10):
        labels[digit * digitreps: (digit+1) * digitreps] = digit

    return data, labels


def load_data(s=0.0, c=1):
    """Load the data from file.

    Loads the 'Project Digits' dataset from file.
    The data is already split into a noisy train and noiseless test set.
    The array of labels is created in label_data.

    Test data will have 1000 examples, 100 of each class.
    Train data will have 1000*c examples, 100*c of each class.

    Each example consists of a 240 dimensional column vector, representing
    a 16x15 dimensional image, and a a single integer representing the label.

    """
    # Load the transposed datafile to get each image in a col vector
    testpath = pathlib.Path(__file__).parent / 'testdata/testdata.txt'
    trainpath = pathlib.Path(__file__).parent / \
        ('traindata/traindata_s_' + str(s) + '_c_' + str(c) + '.txt')

    x_train, y_train = label_data(trainpath)
    x_test, y_test = label_data(testpath)

    return (x_train, y_train), (x_test, y_test)


def results(model, data, labels, show_list=[]):
    """Show and/or return the results.

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
            'MR': partial(print, f'Misclassification Rate: {1-acc:.3f}'),
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
        'MR': 1-acc,
    }


def cross_val(pipeline, data, labels, n_splits=10, n_repeats=1):
    """Perform cross-validation while adding noise."""
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

        # Add noise + labels to x_train + y_train here, like:
        # x_train, y_train = add_noise(x_train, y_train)

        # Fit the model and collect the results (its performance)
        pipeline.fit(x_train, y_train)
        perf.append(results(pipeline, x_test, y_test))

    return perf


def param_sweep_LR(pipeline, data, labels, m_vals=[33], alphas=[0]):
    """Perform a parameter sweep on the Linear Regression pipeline.

    Pass in the full dataset and labels, and specify which parameters
    to sweep for as lists. To use this on a KNN pipeline, simply change
    the keyword arguments in set_params() accordingly.
    """
    # Generate all possible combinations, producing a list of dicts
    params = list(ParameterGrid({
        'm': m_vals,
        'alpha': alphas,
    }))

    # Perform cross validation for each parameter combination
    results = list()
    for param_set in params:
        pipeline.set_params(pca__n_components=param_set['m'])
        pipeline.set_params(LR__alpha=param_set['alpha'])

        perf = cross_val(pipeline, data, labels)
        results.append((param_set, perf))

    return results


def do_everything():
    """Perform the full experiment.

    This is just to show how the old code in the main could
    be replaced.
    """
    # Load the data, make the labels
    data = np.loadtxt('mfeat-pix.txt')
    labels = np.zeros(data.shape[0], dtype=np.int)
    digitreps = int(data.shape[0] / 10)
    for digit in range(10):
        labels[digit * digitreps: (digit+1) * digitreps] = digit

    # Define the pipeline
    pipeline = Pipeline([
        ('normalize', StandardScaler()),
        ('pca', PCA()),
        ('LR', RidgeClassifier()),
    ])

    # Do paramsweep with cross validation
    results = param_sweep_LR(pipeline, data, labels)

    return results

#########################################################################


if __name__ == '__main__':
    t0 = time.time()

    # Load data
    if len(sys.argv) <= 1:
        (x_train, y_train), (x_test, y_test) = load_data()
    else:
        (s, c) = (sys.argv[1], sys.argv[2])
        (x_train, y_train), (x_test, y_test) = load_data(s, c)

    # Initialize data structures to capture all necessary info
    LR_train = list()
    LR_test = list()
    KNN_train = list()
    KNN_test = list()

    # Parameter sweep loop
    alpha = 0
    m_vals = list(range(1, 241))
    for m in m_vals:
        print(f'\n\nSetting m={m}:')

        # Preprocess the data
        preprocessing = Pipeline([
            ('normalize', StandardScaler()),
            ('pca', PCA(n_components=m)),
        ])

        x_train_processed = preprocessing.fit_transform(x_train.T)
        x_test_processed = preprocessing.transform(x_test.T)
        print(f'Preprocessed the data ({timestamp(t0)})')

        # Fit the models (Only LR runs in 20s, both run in 170s)
        LR_model = RidgeClassifier().fit(x_train_processed, y_train)
        KNN_model = KNeighborsClassifier().fit(x_train_processed, y_train)

        # Evaluate the models
        LR_train.append(results(LR_model, x_train_processed, y_train))
        LR_test.append(results(LR_model, x_test_processed, y_test))
        KNN_train.append(results(KNN_model, x_train_processed, y_train))
        KNN_test.append(results(KNN_model, x_test_processed, y_test))
        print(f'Crunched the numbers ({timestamp(t0)})')

    # Plot the results
    MR_LR_train = [np.log10(res['MR']) for res in LR_train]
    MR_LR_test = [np.log10(res['MR']) for res in LR_test]
    MR_KNN_train = [np.log10(res['MR']) for res in KNN_train]
    MR_KNN_test = [np.log10(res['MR']) for res in KNN_test]

    plt.plot(m_vals, MR_LR_train, c='blue', linestyle='-', label='LR_train')
    plt.plot(m_vals, MR_LR_test, c='red', linestyle='-', label='LR_test')
    plt.plot(m_vals, MR_KNN_train, c='blue', linestyle='--', label='KNN_train')
    plt.plot(m_vals, MR_KNN_test, c='red', linestyle='--', label='KNN_test')
    plt.xlabel('m')
    plt.ylabel('MR (log10)')
    plt.title(f'MR vs chosen m (with alpha={alpha})')
    plt.legend()
    plt.show()
