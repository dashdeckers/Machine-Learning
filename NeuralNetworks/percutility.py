"""Utility functions and classes for perceptron algorithms."""

import numpy as np


def generate_teacher(N, clamped):
    """Create a teacher perceptron w_opt.

    Create a random teacher perceptron w_opt and normalize it such that:
    |w|^2 = N  <==>  |w| = sqrt(N)
    """
    if clamped:
        N += 1

    w_opt = np.random.uniform(low=-1, high=1, size=N)
    w_opt = w_opt / np.linalg.norm(w_opt) * np.sqrt(N)

    return w_opt


def generate_C_eta(xi, S):
    """Compute the correlation matrix C and an appropriate learning rate eta.

    C.shape == (P, P), with C[u, v] == 1/N * S_u * S_v * np.dot(xi_u, xi_v)
    0 < eta < C[u, u] for all u
    """
    P, N = xi.shape
    C = np.zeros(shape=(P, P))
    for i in range(P):
        for j in range(P):
            C[i, j] = (S[i] * S[j] * np.dot(xi[i], xi[j])) / N

    lower_b, upper_b = 0, min(C.diagonal())
    eta = (lower_b + upper_b) / 2

    return C, eta


def embedding_to_weights(xi, S, x, N):
    """Compute the weight vector via embdedding strengths.

    Formula:
    w = 1/N * SUM_v(x_mu * xi_mu * S_mu)
    """
    P, N = xi.shape
    w = np.zeros(xi.shape[1])
    for v in range(P):
        w += (xi[v] * S[v] * x[v])

    return w / N


def generalization_error(w, w_opt):
    """Compute the generalization error.

    The generalization error between between a student perceptron w and the
    teacher perceptron w_opt can be computed by finding the angle between them:

    e = 1/pi * arccos( (w .* w_opt) / (|w| * |w_opt|) )

    If either w or w_opt is a zero vector, this function returns 1.
    """
    if sum(w) == 0 or sum(w_opt) == 0:
        return 1
    return (1 / np.pi) * np.arccos(
        np.dot(w, w_opt) / (Vector(w).magnitude() * Vector(w_opt).magnitude())
    )


def sign(xi_v, w, theta=0):
    """Compute the response of a perceptron w to a datapoint xi_v.

    S_w(xi_v) = {
        +1 if dot(xi_v, w) >= theta,
        -1 if dot(xi_v, w) <  theta
    }
    """
    response = np.sign(np.dot(xi_v, w) - theta)
    # np.sign(0) == 0, but we want response=1 in this case
    return 1 if response == 0 else response


def generate_data(P=20, N=2, mean=0, variance=1, w_opt=None, clamped=False):
    """Generate the data.

    Generates P randomly generated N-dimensional feature vectors xi and
    corresponding labels S. Also generates the weight vector w, initialized
    to zeros (tabula rasa).

    The feature vector values xi are sampled from a Gaussian distribution with
    the given mean and variance, and the binary labels S (element {-1, 1}) are
    either randomly (uniformly) selected, or set according to the teacher
    perceptron w_opt.

    If clamped is set to True, then -1 is appended to each datapoint and 0 is
    appended to the weight vector. This results in a final dimension of N+1,
    which allows for inhomogeneous solutions.
    """
    mean = [mean] * N
    covar = np.identity(N) * variance

    xi = np.random.multivariate_normal(mean, covar, P)
    w = np.zeros(shape=(N,))

    # Clamp the generated data to add a degree of freedom
    if clamped:
        # Add a column vector of -1's to the data
        clamped_col_vec = np.array([-1] * P).reshape(-1, 1)
        xi = np.concatenate((xi, clamped_col_vec), axis=1)
        # Add a theta value to the weights
        w = np.append(w, [0])

    # Randomly assign labels, or according to teacher perceptron w*
    if w_opt is not None:
        S = [sign(w_opt, xi_v) for xi_v in xi]
    else:
        S = np.random.choice([-1, 1], P)

    return xi, S, w


class Vector:
    """Vector Class.

    A helper class to make easier to work with Vectors in
    future projects. Can be generalized to more dimensions.
    """

    def __init__(self, coords):
        self.coords = np.array(coords)
        # Set x and y for convenience in 2D
        if len(self.coords) == 2:
            self.x = coords[0]
            self.y = coords[1]

    def __sub__(self, other):
        return Vector(np.subtract(self.coords, other.coords))

    def __add__(self, other):
        return Vector(np.add(self.coords, other.coords))

    def __mul__(self, other):
        # works for Vector * Vector and Vector * Scalar
        if isinstance(other, Vector):
            return Vector(np.multiply(self.coords, other.coords))
        else:
            return Vector(np.multiply(self.coords, other))

    def __str__(self):
        if len(self.coords) == 2:
            return f'({self.x}, {self.y})'
        else:
            return f'Vector([{self.coords}])'

    def __repr__(self):
        return self.__str__()

    def dot(self, other):
        return np.dot(self.coords, other.coords)

    def magnitude(self):
        return np.sqrt(np.sum(np.square(self.coords)))

    def normalized(self):
        assert self.magnitude != 0
        return Vector(np.divide(self.coords, self.magnitude()))

    def perpendicular(self):
        assert len(self.coords) == 2, 'Perpendicular only works in 2D'

        # If this is a zero vector, return self
        if not self.coords.any():
            return self

        return Vector((self.y, -self.x))
