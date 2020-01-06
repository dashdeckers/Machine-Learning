"""A Linear Regression baseline algorithm for predicting handwritten digits."""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


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


def try_it_out(W, Um, digit):
    """Visually / Manually try out the classifier.

    W is the classifier itself that can make a prediction given a feature
    vector Um is the feature 'mapper' that can reduce the dimensionality of raw
    input digit is the raw input (as a column vector) of a digit

    Usage:
        ...
        # Keep a decent classifier + feature mapping to try out later
        if m == 30:
            W_keep = W
            Um_keep = Um
        ...

        digit = x_test[:, arbitrary_index]
        try_it_out(W_keep, Um_keep, digit)

    """
    # Reduce dimensionality of the digit (=get feature vector for digit)
    f_digit = np.dot(Um.T, digit)
    # Make prediction on the feature vector (=get probability vector)
    pred = np.dot(W, f_digit)
    # Print best guess and show the digit
    print(f'Best guess: {np.argmax(pred)}')
    show_digit(digit, col_vector=True)


def load_data(filename='mfeat-pix.txt'):
    """Load the data from file.

    Loads the 'Project Digits' dataset from file and splits it into
    test and train as per the project instructions. Also creates the array
    of labels.

    Both test and training data will have 1000 examples, 100 of each class.
    Each example consists of a 240 dimensional column vector, representing
    a 16x15 dimensional image, and a a single integer representing the label.

    """
    with open(filename, 'r') as datafile:
        # Load the transposed datafile to get each image in a col vector
        data = np.loadtxt(datafile).T  # (240, 2000)

        # Create an array of labels (each 200 elements is a digit)
        labels = np.zeros(data.shape[1], dtype=np.int)
        for digit in range(10):
            labels[digit * 200: (digit+1) * 200] = digit  # (2000,)

        # Split the data into train and test by first determining the indices
        even = np.array([np.arange(i*100, (i+1)*100) for i in range(0, 20, 2)])
        odd = np.array([np.arange(i*100, (i+1)*100) for i in range(1, 20, 2)])
        even = even.reshape(-1)  # reshape to get a single,
        odd = odd.reshape(-1)    # long array of indices: (1000, )

        # And then selecting via the array of indices
        x_train, y_train = data[:, even], labels[even]
        x_test,  y_test = data[:, odd],  labels[odd]

        return (x_train, y_train), (x_test, y_test)


def preprocess_data(data, subtract_mean=False):
    """Preprocess the data.

    Preprocess the data from load_data by normalizing the values to the
    range [0,1] and centering the data by subtracting the mean.

    Centering the data actually decreases performance.. lowest MSE_train ~ -0.5

    """
    (x_train, y_train), (x_test, y_test) = data

    # Normalize the data to range [0,1] (assuming range [0,6])
    x_train /= 6
    x_test /= 6

    assert np.max(x_train) <= 1, np.max(x_train)
    assert np.max(x_test) <= 1, np.max(x_test)
    assert np.min(x_train) >= 0, np.min(x_train)
    assert np.min(x_test) >= 0, np.min(x_test)

    # Center the data by subtracting the mean
    if subtract_mean:
        mean = x_train.mean(axis=1).reshape(-1, 1)  # (240, 1) (=column vector)
        assert mean.shape == (240, 1)
        x_train -= mean
        x_test -= mean

    return (x_train, y_train), (x_test, y_test)


def compute_first_m_PCs_of_x(x, m):
    """Compute the first m principle components of the data-matrix x.

    Um.shape == (240, m)

    """
    # Compute covariance matrix
    C = np.cov(x)
    # Get the SVD of C
    U, S, V = np.linalg.svd(C)
    # Return the first m columns of U
    return U[:, :m]


def one_hot_encode_labels(labels):
    """Return a One-Hot encoded representation of digit labels.

    Returns a matrix of one-hot encoded vectors for the given array
    of values. Assumes that the labels in y correspond to the indices in
    [0, 9] that should be set to 1 in the resulting matrix.

        label_matrix.shape == (10, n_labels)

    """
    assert labels.ndim == 1, (labels, labels.shape)

    N = len(labels)
    # Create a matrix of zeros
    label_matrix = np.zeros(shape=(10, N))
    # Entries with (row, col) == (label, c_index) should equal 1
    label_matrix[labels, range(N)] = 1

    return label_matrix


def compute_LR_classifier(F, V, alpha=0):
    """Compute the Linear Regression classifier according to lecture notes.

    It includes the Ridge Regression regularization term (eq. 47).

    W' = (F F' + a^2 * I)^-1 F V'

    """
    Reg_Term = alpha**2 * np.identity(F.shape[0])

    F_Fprime = np.dot(F, F.T)           # F*F'
    RidgeAdd = F_Fprime + Reg_Term      # (F*F' + a2*I)
    F_Fp_inv = np.linalg.inv(RidgeAdd)  # (F*F' + a2*I)^-1

    Triple_F = np.dot(F_Fp_inv, F)      # (F*F' + a2*I)^-1 * F
    W_transp = np.dot(Triple_F, V.T)    # (F*F' + a2*I)^-1 * F * V'

    return W_transp.T                   # W'


def compute_MSE(V, F, W):
    """Compute the Mean Squared Error.

    Find the difference between predictions (=W*F) and label-vectors, square
    the result to avoid negative values, then find the mean squared error for
    each example. Finally, sum these means and divide by n_examples.

    """
    n_examples = V.shape[1]
    squared_error = np.square(V - np.dot(W, F))  # (10, 1000)
    mean_squared_error = squared_error.mean(axis=0)  # (1000, )
    return mean_squared_error.sum() / n_examples  # (float)


def compute_MR(V, F, W):
    """Compute the Misclassification Rate.

    Calculate all prediction vectors, then find the argmax of each vector to
    get an array containing the best guesses for each example in F.

    Similarly, find the argmax of each one-hot encoded label vector in V to get
    an array containing all the true answers for each example in F.

    Then, count the number of times the answer was incorrect and divide by
    n_examples.

    """
    n_examples = V.shape[1]

    pred = np.dot(W, F)  # (10, n_examples)
    best = np.argmax(pred, axis=0)  # (n_examples, )
    true = np.argmax(V, axis=0)  # (n_examples, )
    return (n_examples - np.count_nonzero(true == best)) / n_examples


def show_results(y_test, pred):
    """Show all the results.

    First print the confusion matrix, and then the precision and recall values.

    """
    print(confusion_matrix(y_test, pred))

    print('')
    print('Class - Precision - Recall')
    precs, recs, *_ = precision_recall_fscore_support(y_test, pred)
    for digit, precision, recall in zip(range(10), precs, recs):
        print(f' {str(digit):4s} - {str(round(precision, 3)):9s} - {recall}')

    print('')
    print(f'Precision (mean, std): {np.mean(precs):.3f}, {np.std(precs):.3f}')
    print(f'Recall (mean, std): {np.mean(recs):.3f}, {np.std(recs):.3f}')


def get_LR_pred(W, F_test):
    """Get the predictions from the weights of the Linear Regression model."""
    pred = np.dot(W, F_test)
    return np.argmax(pred, axis=0)


def get_KNN_pred(x_train, y_train, x_test):
    """Fit a KNN model on the data and return the predictions."""
    y_train = LabelEncoder().fit_transform(y_train)
    KNN_model = KNeighborsClassifier().fit(x_train.T, y_train)
    return KNN_model.predict(x_test.T)


if __name__ == '__main__':
    t0 = time.time()

    # Step 0: Load and preprocess Data
    (x_train, y_train), (x_test, y_test) = preprocess_data(load_data())
    print(f'Loaded and preprocessed data ({timestamp(t0)})')

    MSE_trains = list()
    MSE_tests = list()
    MR_trains = list()
    MR_tests = list()
    m_vals = list(range(241))

    alpha = 0
    for m in m_vals:
        print(f'Setting m={m}:')

        # Step 1: PCA
        Um = compute_first_m_PCs_of_x(x_train, m)
        F_train = np.dot(Um.T, x_train)
        F_test = np.dot(Um.T, x_test)
        print(f'Computed PCA feature vectors ({timestamp(t0)})')

        # Step 2: One-Hot Encode Labels
        V_train = one_hot_encode_labels(y_train)
        V_test = one_hot_encode_labels(y_test)
        print(f'One-Hot encoded labels ({timestamp(t0)})')

        # Step 3: Compute LR Classifier
        W = compute_LR_classifier(F_train, V_train, alpha)
        print(f'Computed linear regression weight matrix ({timestamp(t0)})')

        # Step 4: Compute the Errors
        MSE_trains.append(np.log10(compute_MSE(V_train, F_train, W)))
        MSE_tests.append(np.log10(compute_MSE(V_test, F_test, W)))
        MR_trains.append(np.log10(compute_MR(V_train, F_train, W)))
        MR_tests.append(np.log10(compute_MR(V_test, F_test, W)))

        print(f'Computed the Errors ({timestamp(t0)})')
        print(f'\tMSE_train error (for m={m}): {MSE_trains[-1]}')
        print(f'\tMSE_test  error (for m={m}): {MSE_tests[-1]}\n')

    # If only one m_val has been chosen, then just compare the results and exit
    if len(m_vals) == 1:
        print('\nResults for the Linear Regression model:\n')
        show_results(y_test, get_LR_pred(W, F_test))
        print('\nResults for the KNN model:\n')
        show_results(y_test, get_KNN_pred(x_train, y_train, x_test))
        print('')
        sys.exit()

    # Step 5: Plot the results
    plt.plot(m_vals, MSE_trains, c='blue', linestyle='--', label='MSE_train')
    plt.plot(m_vals, MSE_tests, c='red', linestyle='--', label='MSE_test')
    plt.plot(m_vals, MR_trains, c='blue', linestyle='-', label='MR_train')
    plt.plot(m_vals, MR_tests, c='red', linestyle='-', label='MR_test')
    plt.xlabel('m')
    plt.ylabel('MSE/MR (log10)')
    plt.title(f'MSE/MR vs chosen m (with alpha={alpha})')
    plt.legend()
    plt.show()
