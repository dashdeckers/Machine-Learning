"""A Rosenblatt Perceptron and means of visualizing its behaviour."""
import numpy as np


def generate_data(P, N=2, mean=0, variance=1, teacher=None, clamped=False):
    """Generate the data.

    Generates P randomly generated N-dimensional feature
    vectors and corresponding labels. Also generates the weight
    vector, initialized to zeros.

    The feature vector values are sampled from a Gaussian
    distribution with mean and variance, and the binary
    labels are either randomly selected from {-1, 1} with
    an even distribution, or set according to the teacher
    perceptron w.

    If clamped is set to True, then append -1 to each datapoint
    and append 0 to the weight vector. This results in a final
    dimension of N+1. This allows for inhomogeneous solutions
    with an offset (in practice by incrementing dimensionality).
    """
    mean = [mean] * N
    covar = np.identity(N) * variance

    data = np.random.multivariate_normal(mean, covar, P)
    weights = np.zeros(shape=(N,))

    # Randomly assign labels, or assign them according to w*
    if teacher is not None:

        # assert that the dimensionality of teacher and data is compatible
        assert len(teacher) == len(data[0])
        # something like loop through data and label = sign(datapoint)
        labels = [sign(teacher, xi) for xi in data]
    else:
        labels = np.random.choice([-1, 1], P)

    # Clamp the generated data to add a degree of freedom
    if clamped:
        # Add a column vector of -1's to the data
        clamped_col_vec = np.array([-1] * P).reshape(-1, 1)
        data = np.concatenate((data, clamped_col_vec), axis=1)
        # Add a theta value to the weights
        weights = np.append(weights, [0])

    return data, labels, weights


def sign(w, xi, theta=0):
    """Compute the Response of the perceptron.

    S_w(xi) = {
        +1 if dot(w, xi) >= theta,
        -1 if dot(w, xi) <  theta
    }
    """
    response = np.sign(np.dot(w, xi) - theta)
    # np.sign(0) == 0, but we want response=1 in this case
    return 1 if response == 0 else response
