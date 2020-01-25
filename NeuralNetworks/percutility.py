"""Utility functions and classes for perceptron algorithms."""

import numpy as np


def generalization_error(w, w_opt):
    """Compute the generalization error.

    The generalization error between between a student perceptron w and the
    teacher perceptron w_opt can be computed by finding the angle between them:

    e = 1/pi * arccos( (w .* w_opt) / (|w| * |w_opt|) )
    """
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

    # Randomly assign labels, or according to teacher perceptron w*
    if w_opt is not None:
        if clamped:
            assert len(w_opt) == N+1
            xi_clamped = [np.append(xi_v, 0) for xi_v in xi]
            S = [sign(w_opt, xi_v) for xi_v in xi_clamped]
        else:
            assert len(w_opt) == N
            S = [sign(w_opt, xi_v) for xi_v in xi]
    else:
        S = np.random.choice([-1, 1], P)

    # Clamp the generated data to add a degree of freedom
    if clamped:
        # Add a column vector of -1's to the data
        clamped_col_vec = np.array([-1] * P).reshape(-1, 1)
        xi = np.concatenate((xi, clamped_col_vec), axis=1)
        # Add a theta value to the weights
        w = np.append(w, [0])

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
        if not any(self.coords):
            return self

        return Vector((self.y, -self.x))
