"""An implementation of the Adatron algorithm."""

import numpy as np
from percplotter import plot
from percutility import (embedding_to_weights, generalization_error,
                         generate_C_eta, generate_data, generate_teacher)


def run_adatron(P=5, N=2, t_max=100, clamped=False, use_teacher=False):
    """Adatron algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Generate teacher perceptron if use_teacher is set to True
    w_opt = None
    if use_teacher:
        w_opt = generate_teacher(N, clamped=clamped)

    # Generate data and weights
    xi, S, _ = generate_data(P, N, w_opt=w_opt, clamped=clamped)

    # Initialize embedding strengths, learning rate and C
    x = np.zeros(shape=xi.shape[0])
    C, eta = generate_C_eta(xi, S)

    # Initialize plotter, if applicable
    plotter = plot(xi, S)
    next(plotter)

    # Epoch loop
    for t in range(t_max):
        # Data loop
        for v, (xi_v, S_v) in enumerate(zip(xi, S)):
            E_v = np.dot(C, x)[v]
            x[v] = max(0, x[v] + eta * (1 - E_v))

        # Send the new weights to the plotter
        plotter.send(embedding_to_weights(xi, S, x, N))

    if use_teacher:
        return generalization_error(embedding_to_weights(xi, S, x, N), w_opt)
    else:
        return False
