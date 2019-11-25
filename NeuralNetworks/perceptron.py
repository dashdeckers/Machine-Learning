import time
import numpy as np
import matplotlib.pyplot as plt

class Vector:
    ''' A helper class to make easier to work with Vectors in
    future projects. Can be generalized to more dimensions.
    '''
    def __init__(self, coords):
        self.coords = np.array(coords)
        # Set x and y for convenience in 2D
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
        # Works for 2D only
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return self.__str__()

    def magnitude(self):
        return np.sqrt(np.sum(np.square(self.coords)))

    def normalized(self):
        return Vector(np.divide(self.coords, self.magnitude()))

    def perpendicular(self):
        # Works for 2D only
        assert len(self.coords) == 2
        return Vector((1, -self.x / self.y))

def generate_data(P, N=2):
    ''' Generates P randomly generated N-dimensional feature
    vectors and corresponding binary labels.

    The feature vector values are sampled from a Gaussian
    distribution with mean 0 and variance 1, and the binary
    labels are randomly selected from {-1, 1} with an even
    distribution.
    '''
    mean = [0] * N
    covar = np.identity(N)

    data = np.random.multivariate_normal(mean, covar, P)
    labels = np.random.choice([-1, 1], P)

    return data, labels

def plot(x, y, labels, weight_vec=None):
    ''' Plots the data points in x and y in two different colors
    depending on the labels, and shows the weight vector from the
    origin as well as the linear separation line orthogonal to it.
    '''
    assert len(x) == len(y), 'x and y must be equally long'
    # Create a figure and plot the points
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=labels)
    fig.canvas.draw()

    # If weights are given
    if weight_vec is not None and len(weight_vec) == 2:
        # Draw the weight vector
        origin = (0, 0)
        Q = ax.quiver(
            *origin,
            *weight_vec,
            color=['black'],
            angles='xy',
            scale_units='xy',
            scale=1
        )
        # Build a normalized, perpendicular vector to w
        w = Vector(weight_vec)
        w_perp_normalized = w.perpendicular().normalized()

        # If it really is perpendicular, the dot product is zero
        assert np.dot(w.coords, w_perp_normalized.coords) == 0

        # Choose two points on that vector to draw the line
        P1 = w + w_perp_normalized * 2
        P2 = w - w_perp_normalized * 2

        # Draw the linear separation orthogonal to weight vector
        ax.plot(
            P1.coords,
            P2.coords,
            c='black',
            marker='.',
            linestyle=':'
        )

    # Show the plot
    plt.axis('equal')
    plt.show(block=False)
    fig.canvas.draw()
    time.sleep(1)

    # for i in range(10):
    #     Q.set_offsets(Q.get_offsets() + np.array([0.1*i, 0.1*i]))
    #     fig.canvas.draw()
    #     time.sleep(0.1)


def sign(x, theta=0):
    ''' The sign function:

    sign(x - theta) = {
            +1 for x >= theta
            -1 for x <  theta
    }

    Guaranteed to return a numpy array
    '''
    res = np.array(np.sign(np.array(x) - theta))
    # np.sign(0) == 0, correct for this
    res[res == 0] = 1
    return res

def response(w, xi, theta):
    '''
    '''
    return sign(np.dot(w, xi), theta)

# Number of Dimensions
N = 2
# Number of Datapoints
P = 100

# Data
xi, labels = generate_data(P, N)
x, y = xi.T

# Params
weights = np.zeros(shape=(N,))


plot(x, y, labels, [5, 5])
