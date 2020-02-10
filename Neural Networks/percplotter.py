"""Plotting functions for perceptron algorithms."""

import time

import matplotlib.pyplot as plt
import numpy as np
from percutility import Vector


def add_quiver(ax, weights):
    """Add quiver to plot.

    Add a Quiver showing the weight vector to the plot. Only
    works in 2D and not if the weight vector is a zero vector.

    Also adds a line perpendicular to the weight vector, which
    goes through the origin.
    """
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

    # Draw a line between the two points
    lines = ax.plot(
        [P1.x, P2.x],
        [P1.y, P2.y],
        c='black',
        marker='.',
        linestyle=':'
    )
    return Q, lines


def plot(xi, labels):
    """Create the plot.

    Plots the 2D coordinates in xi with colors according to labels.

    Returns a generator with which you can iteratively plot a weight
    vector and its perpendicular separation line.

    Usage:
    # Create dummy data. Make sure xi is a numpy.ndarray
    xi = np.array([[0, 1], [1, 1]])
    labels = [0, 1]

    # Create the plotter
    plotter = plot(xi, labels)

    # Initialize the plotter and show the first plot
    next(plotter)

    # Iteratively plot the weight vector
    weight_vec = [0, 1]
    for i in [1, 2, 3, 4, 5, 6]:
        weight_vec[0] += i
        plotter.send(weight_vec)
    """
    try:
        assert xi.shape[1] == 2, 'Can only plot in 2D'

        # Create a figure and plot the points
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
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

        # Show the plot
        plt.axis('equal')
        plt.show()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
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
                Q, lines = add_quiver(ax, weights)
                fig.canvas.draw()
                plt.pause(0.0005)
                time.sleep(0.5)

    except AssertionError:
        # If not 2D (or any issue), become a bogus generator
        while True:
            yield
