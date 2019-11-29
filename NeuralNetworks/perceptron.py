import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import groupby
from operator import itemgetter

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

def plot(xi, labels, verbose=False):
    ''' Creates a 2D plot with the (x, y) coordinates in xi, in two 
    different colors depending on the labels. The plot is interactive
    to allow for iterative updating.
    '''
    try:
        assert xi.shape[1] == 2, 'Can only plot in 2D'

        # Create a figure and plot the points
        plt.ion()
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        x, y = xi.T
        ax.scatter(x, y, c=labels)
        fig.canvas.draw()

        # Prevent a messy tkinter error when manually closing the plot
        global running
        running = True
        def handle_close(evt):
            global running
            running = False
        fig.canvas.mpl_connect('close_event', handle_close)

        # Show the plot and return it
        plt.axis('equal')
        plt.show()
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        fig.canvas.draw()

        # Initialize perpendicular arrow and line
        Q, lines = (None, None)

        while running:
            weights = yield

            # If if there is a Quiver, remove it
            if Q is not None and lines is not None:
                Q.remove()
                lines.pop(0).remove()

            # If weights is not a zero vector, draw a Quiver
            if np.any(weights):
                Q, lines = add_quiver(ax, weights, verbose)
                fig.canvas.draw()
                plt.pause(0.0005)
                time.sleep(0.5)

    except AssertionError:
        # If not 2D, become a bogus generator
        while True:
            _ = yield

def generate_data(P, N=2, mean=0, variance=1, labels='random', clamped=False):
    ''' Generates P randomly generated N-dimensional feature
    vectors and corresponding labels. Also generates the weight
    vector, initialized to zeros.

    The feature vector values are sampled from a Gaussian
    distribution with mean and variance, and the binary
    labels are either randomly selected from {-1, 1} with
    an even distribution, or equal to the value of labels.

    If clamped is set to True, then append -1 to each datapoint
    and append 0 to the weight vector. This results in a final
    dimension of N+1. This allows for inhomogeneous solutions
    with an offset (in practice by increasing dimensionality).
    '''
    mean = [mean] * N
    covar = np.identity(N) * variance

    data = np.random.multivariate_normal(mean, covar, P)
    weights = np.zeros(shape=(N,))

    # Randomly assign labels, or assign them all the value of labels
    if labels == 'random':
        labels = np.random.choice([-1, 1], P)
    else:
        labels = np.array([labels] * P)

    # Clamp the generated data to add a degree of freedom
    if clamped:
        # Add a column vector of -1's to the data
        clamped_col_vec = np.array([-1] * P).reshape(-1, 1)
        data = np.concatenate((data, clamped_col_vec), axis=1)
        # Add a theta value to the weights
        weights = np.append(weights, [0])

    return data, labels, weights

def response(w, xi, theta=0):
    ''' The Response of the perceptron.

    S_w(xi) = {
        +1 if dot(w, xi) >= theta,
        -1 if dot(w, xi) <  theta
    }

    The points given by xi are linearly separated by
    the hyperplane given by dot(w, xi) - theta.
    '''
    response = np.sign(np.dot(w, xi) - theta)
    # np.sign(0) == 0, but we want response=1 in this case
    if response == 0:
        return 1
    else:
        return response

def run_rosenblatt(N=2, P=5, n_max=5, clamped=True, verbose=False):
    ''' Rosenblatt learning algorithm, where:

    N is the number of dimensions
    P is the number of datapoints
    n_max is the number of Epochs to run for

    if verbose is set to True, it will print stuff.
    '''
    # Generate data and weights
    xi, labels, weights = generate_data(P, N, clamped=clamped)

    # Initialize plotter, if applicable
    plotter = plot(xi, labels)
    next(plotter)

    # Epoch loop
    for epoch in range(n_max):
        if verbose: print(f'Epoch {epoch}/{n_max}')

        # Data loop
        stop_early = True
        for xi_t, label_t in zip(xi, labels):
            # Get the error via Hebbian learning:
            # If response == label: Error = 1, else Error = -1
            E_t = response(weights, xi_t) * label_t

            # If Error == -1, update weights and don't stop early
            if E_t == -1:
                weights += (xi_t * label_t) / N
                stop_early = False

            # Send the new weights to the plotter
            plotter.send(weights)

        # If we haven't updated any weight in this data loop, success
        if stop_early:
            return (True, weights)

    # If the stop early condition never happened, failure
    return (False, weights)

# Functions to execute the actions that individual threads need to take
def run_experiment(alpha, N, clamped):
    Pa = 0
    repetitions = 50
    for i in range(repetitions):
        P = int(alpha * N)
        result, _ = run_rosenblatt(N=N, P=P, n_max=100, clamped=clamped)
        Pa += int(result)

    return (N, alpha, Pa / repetitions)

def collect_data(clamped=False):
    # Create the arguments to run
    alphaset = np.arange(0.75,3,0.25)
    dimensions = [5, 20] # [150, 20, 5]
    args = [(a, N, clamped) for N in dimensions for a in alphaset]

    # Determine the number of threads available
    pool = mp.Pool(mp.cpu_count())
    print(f'CPUs = {mp.cpu_count()}')

    # Have each thread execute on a subset of the various alphas
    output = pool.starmap(run_experiment, args)
    out_lists = [list(g) for _, g in groupby(output, itemgetter(0))]
    print(out_lists)
    pool.close()

    # Plot results
    colours = ["red", "orange", "green", "blue", "purple"]
    for colour, tup_list in zip(colours, out_lists):
        prob_vals = [tup[2] for tup in tup_list]
        plt.plot(alphaset, prob_vals, 'r--', c=colour, label=tup_list[0][0])

    plt.legend(title="Number of dimensions")
    plt.title("Probability of linear separability for Data points/Dimension ratio")
    plt.xlabel(r'$\alpha$ defined as the ratio of Data Points per Dimension')
    plt.ylabel("Probability of being linearly seperable (%)")
    plt.show()

if __name__ == '__main__':
    # collect_data()
    pass
