"""An implementation of the Rosenblatt algorithm."""

from percplotter import plot
from percutility import (generalization_error, generate_data, generate_teacher,
                         sign)


def run_rosenblatt(P=5, N=2, t_max=100, clamped=False, use_teacher=False):
    """Rosenblatt algorithm.

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
        stop_early = True
        for xi_v, S_v in zip(xi, S):
            # Get the error via Hebbian learning:
            # If response == label: Error = 1, else Error = -1
            E_v = sign(xi_v, w) * S_v

            # If Error == -1, update weights and don't stop early
            if E_v == -1:
                w += (xi_v * S_v) / N
                stop_early = False

            # Send the new weights to the plotter
            plotter.send(w)

        # If we haven't updated any weight in this data loop, success
        if stop_early:
            if use_teacher:
                return generalization_error(w, w_opt)
            else:
                return True

    # If the stop early condition never happened, failure
    if use_teacher:
        return generalization_error(w, w_opt)
    else:
        return False
