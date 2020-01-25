"""An implementation of the Minover algorithm."""

import numpy as np
from percplotter import plot
from percutility import generalization_error, generate_data


def run_minover(P=5, N=2, t_max=100, clamped=True, verbose=False):
    """Minover algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Create a random teacher perceptron w_opt
    w_opt = [1, 1]  # TODO: Random generation according to |w^2| = N

    # Generate data and weights
    xi, S, w = generate_data(P, N, w_opt=w_opt, clamped=clamped)

    # Initialize plotter, if applicable
    plotter = plot(xi, S)
    next(plotter)

    # Epoch loop
    for t in range(t_max):
        if verbose:
            print(f'Epoch {t}/{t_max}')

        # Data loop
        min_stability = None
        for xi_v, S_v in zip(xi, S):
            # Find the datapoint v = (xi_v, S_v) with minimal stability
            # Min stability = min local potential = abs(w .* xi_v * S_v)
            stability = abs(np.dot(w, xi_v * S_v))

            if min_stability is None or stability < min_stability[2]:
                min_stability = (xi_v, S_v, stability)

        # Update the weight vector with (xi_v, S_v)
        min_xi_v, min_S_v, stability = min_stability
        w += (min_xi_v * min_S_v) / N  # TODO: Does not converge???

        # DEBUG
        print(f'w: {w}')
        print(f'xi_v: {xi_v}, stability: {stability}\n')

        # Send the new weights to the plotter
        plotter.send(w)

    return generalization_error(w, w_opt)
