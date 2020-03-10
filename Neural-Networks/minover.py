"""An implementation of the Minover algorithm."""

import numpy as np
from percplotter import plot
from percutility import generalization_error, generate_data, generate_teacher


def run_minover(P=5, N=2, t_max=100, clamped=False, use_teacher=False):
    """Minover algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Generate teacher perceptron if use_teacher is set to True
    w_opt = None
    if use_teacher:
        w_opt = generate_teacher(N, clamped=clamped)

    # Generate data and weights
    xi, S, w = generate_data(P, N, w_opt=w_opt, clamped=clamped)

    # Initialize plotter, if applicable
    plotter = plot(xi, S)
    next(plotter)

    # Epoch loop
    for t in range(t_max):
        # Data loop
        # Find the datapoint v = (xi_v, S_v) with minimal stability
        min_stability = None
        for xi_v, S_v in zip(xi, S):
            # Min stability = min local potential = abs(w .* xi_v * S_v)
            stability = np.dot(w, xi_v * S_v)

            if min_stability is None or stability < min_stability[2]:
                min_stability = (xi_v, S_v, stability)

        # Update the weight vector with v = (xi_v, S_v)
        min_xi_v, min_S_v, stability = min_stability  # type: ignore
        w += (min_xi_v * min_S_v) / N

        # Send the new weights to the plotter
        plotter.send(w)

    if use_teacher:
        return generalization_error(w, w_opt)
    else:
        return min_stability[2]
