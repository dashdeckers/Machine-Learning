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
        assert self.magnitude != 0
        return Vector(np.divide(self.coords, self.magnitude()))

    def perpendicular(self):
        # Works for 2D only
        assert len(self.coords) == 2, 'Perpendicular only works in 2D'

        # If this is a zero vector, return self
        if not any(self.coords):
            return self

        return Vector((self.y, -self.x))

def generate_data(P, N=2, mean=0, variance=1, labels='random'):
    ''' Generates P randomly generated N-dimensional feature
    vectors and corresponding labels.

    The feature vector values are sampled from a Gaussian
    distribution with mean and variance, and the binary
    labels are either randomly selected from {-1, 1} with
    an even distribution, or equal to the value of labels.
    '''
    mean = [mean] * N
    covar = np.identity(N) * variance

    data = np.random.multivariate_normal(mean, covar, P)

    if labels == 'random':
        labels = np.random.choice([-1, 1], P)
    else:
        labels = np.array([labels] * P)

    return data, labels

def add_quiver(ax, weights, verbose=False):
    ''' Add a Quiver showing the weight vector to the plot. Only
    works in 2D and not if the weight vector is a zero vector.

    Also adds a line perpendicular to the weight vector, which
    goes through the origin.
    '''
    assert len(weights) == 2, 'Weights must be 2D'
    assert np.any(weights), 'Weights cannot be zero vector'

    # Get origin, weight vector and perpendicular vector
    origin = Vector((0, 0))
    weight_vec = Vector(weights).normalized()
    perp_vec = weight_vec.perpendicular().normalized()

    # Draw the weight vector quiver
    Q = ax.quiver(
        *origin.coords,
        *weight_vec.coords,
        color=['black'],
        angles='xy',
        scale_units='xy',
        scale=1
    )

    # Choose two points on the perpendicular vector
    P1 = perp_vec
    P2 = perp_vec * -1

    # Print stuff if verbose is set to True
    if verbose:
        print(f'Weight Vector: {weight_vec.coords}')
        print(f'Perp   Vector: {perp_vec.coords}')
        print(f'P1: {P1.coords}, P2: {P2.coords}')

    # Draw a line between the two points
    lines = ax.plot(
        [P1.x, P2.x],
        [P1.y, P2.y],
        c='black',
        marker='.',
        linestyle=':'
    )
    return Q, lines

def make_plot(xi, labels, xlim=[-3, 3], ylim=[-3, 3]):
    ''' Creates a 2D plot with the (x, y) coordinates in xi, in two 
    different colors depending on the labels. The plot is interactive
    to allow for iterative updating.
    '''
    assert xi.shape[1] == 2, 'Can only plot in 2D'

    # Create a figure and plot the points
    plt.ion()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    x, y = xi.T
    ax.scatter(x, y, c=labels)
    fig.canvas.draw()

    # Show the plot and return it
    plt.axis('equal')
    plt.show(block=False)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    fig.canvas.draw()
    return ax, fig, plt

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

def response(w, xi, theta=0):
    ''' The Error term E

    E = ???
    '''
    return sign(np.dot(w, xi), theta)

def run_rosenblatt(N=2, P=5, n_max=5, verbose=False):
    ''' Rosenblatt learning algorithm, where:

    N is the number of dimensions
    P is the number of datapoints
    n_max is the number of Epochs to run for

    if verbose is set to True, it will print more than
    just the Epoch progress.
    '''
    # Generate data and plot
    xi, labels = generate_data(P, N)

    # Initialize Perceptron parameters
    weights = np.zeros(shape=(N,))

    # Initialize plot data (only available in 2D)
    if N == 2:
        ax, fig, _ = make_plot(xi, labels)
        Q, lines = (None, None)

    # Epoch loop
    for epoch in range(n_max):
        print(f'Epoch {epoch}/{n_max}')
        stop_early = True

        # Data loop
        for xi_t, label_t in zip(xi, labels):
            # Get the error
            E_t = response(weights, xi_t) * label_t

            # If condition, update weights and don't stop early        
            if E_t <= 0:
                weights += (xi_t * label_t)/N
                stop_early = False

            # Only plot in 2D
            if N == 2:
                # If if there is a Quiver, remove it
                if Q is not None and lines is not None:
                    Q.remove()
                    lines.pop(0).remove()

                # If weights is not a zero vector, draw a Quiver
                if np.any(weights):
                    Q, lines = add_quiver(ax, weights, verbose)
                    fig.canvas.draw()
                    time.sleep(0.5)

        # If we haven't updated any weight in this data loop, success
        if stop_early:
            return (True, weights)

    # If the stop early condition never happened, failure
    return (False, weights)

if __name__ == '__main__':
    run_rosenblatt()




















### Random irrelevant stuff

def test_dataplot():
    ''' Testing the new generate_data
    '''
    xi1, labels1 = generate_data(50, 2, mean=1, variance=1, labels=-1)
    xi2, labels2 = generate_data(50, 2, mean=3, variance=1, labels=1)
    ax, fig, plt = make_plot(xi1, labels1, xlim=[-2, 6], ylim=[-2, 6])

    x2, y2 = xi2.T
    ax.scatter(x2, y2, c='red')

    return ax, fig, plt

def get_solution(phi, y):
    ''' The shortcut solution from the linear regression part of the
    lecture notes.
    '''
    return np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), y)