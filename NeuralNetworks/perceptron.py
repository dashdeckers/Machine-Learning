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
        assert len(self.coords) == 2

        # If this is a zero vector, return self
        if not any(self.coords):
            return self

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

def add_quiver(ax, weights):
    ''' Add a Quiver showing the weight vector to the plot. Only
    works in 2D and not if the weight vector is a zero vector.

    Also adds a line perpendicular to the weight vector, which
    goes through the origin.
    '''
    assert len(weights) == 2
    assert np.any(weights)

    # Get origin, weight vector and perpendicular vector
    origin = Vector((0, 0))
    weight_vec = Vector(weights).normalized()
    perp_vec = weight_vec.perpendicular().normalized()
    assert np.dot(weight_vec.coords, perp_vec.coords) <= 1e-5

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
    P1 =  perp_vec
    P2 =  perp_vec * -1

    debug.append((P1, P2))

    # Make sure the line between the points goes through the origin
    assert (P1.x * (P2.y - P1.y) == P1.y * (P2.x - P1.x))
    print(f'W: {weights}')
    print(f'P1: {P1.coords}, P2: {P2.coords}')

    # Draw a line between the two points
    lines = ax.plot(
        P1.coords,
        P2.coords,
        c='black',
        marker='.',
        linestyle=':'
    )
    return Q, lines

def make_plot(xi, labels):
    ''' Creates a 2D plot with the (x, y) coordinates in xi, in two 
    different colors depending on the labels. The plot is interactive
    to allow for iterative updating.
    '''
    x, y = xi.T
    assert len(x) == len(y), 'x and y must be equally long'

    # Create a figure and plot the points
    plt.ion()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.scatter(x, y, c=labels)
    fig.canvas.draw()

    # Show the plot and return it
    plt.axis('equal')
    plt.show(block=False)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    fig.canvas.draw()
    return ax, fig

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
    '''
    '''
    return sign(np.dot(w, xi), theta)


# Set global variables
N = 2                       # Number of Dimensions
P = 5                       # Number of Datapoints
n_max = 5                   # Number of Epochs
Q, lines = (None, None)     # Initialize plot data

# Generate data and plot
xi, labels = generate_data(P, N)
ax, fig = make_plot(xi, labels)

# Initialize Perceptron parameters
weights = np.zeros(shape=(N,))

debug = list()
import pprint
pp = pprint.PrettyPrinter()

# Epoch loop
for epoch in range(n_max):
    stop_early = True

    # Data loop
    for xi_t, label_t in zip(xi, labels):
        # Get the error
        E_t = response(weights, xi_t) * label_t

        # If condition, update weights and don't stop early        
        if E_t <= 0:
            weights += (xi_t * label_t)/N
            stop_early = False

        # If if there is a Quiver, remove it
        if Q is not None and lines is not None:
            Q.remove()
            pp.pprint(lines[0].__dict__)
            lines.pop(0).remove()

        # If weights is not a zero vector, draw a Quiver
        if np.any(weights):
            Q, lines = add_quiver(ax, weights)
            fig.canvas.draw()
            time.sleep(0.5)

    # If we have not updated any weight in this data loop, stop
    if stop_early:
        break






# def is_parallel(P1, P2, P3, P4):
#     return abs((P2.x - P1.x)*(P4.y - P3.y) - (P4.x - P3.x)*(P2.y - P1.y)) < 1e-10

# def check_parallel():
#     ref_point = debug.pop()
#     for point in debug:
#         if not is_parallel(*ref_point, *point):
#             print('Not Parallel!')
#         else:
#             print('Parallel')
