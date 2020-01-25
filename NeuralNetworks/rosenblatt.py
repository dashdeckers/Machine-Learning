"""An implementation of the Rosenblatt algorithm."""

import multiprocessing as mp
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from percplotter import plot
from percutility import generate_data, sign


def run_rosenblatt(P=5, N=2, t_max=100, clamped=True, verbose=False):
    """Rosenblatt algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Generate data and weights
    xi, S, w = generate_data(P, N, clamped=clamped)

    # Initialize plotter, if applicable
    plotter = plot(xi, S)
    next(plotter)

    # Epoch loop
    for t in range(t_max):
        if verbose:
            print(f'Epoch {t}/{t_max}')

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
            return True, w

    # If the stop early condition never happened, failure
    return False, w


# Function to execute the actions that individual threads need to take
def run_experiment(alpha, N, clamped):
    Pa = 0
    repetitions = 100
    for i in range(repetitions):
        P = int(alpha * N)
        result, _ = run_rosenblatt(P=P, N=N, clamped=clamped)
        Pa += int(result)

    return N, alpha, Pa / repetitions


def collect_data(clamped=False):
    # Create the arguments to run
    alphaset = np.arange(0.75, 5, 0.1)
    dimensions = [5, 20, 150]
    args = [(a, N, clamped) for N in dimensions for a in alphaset]

    # Determine the number of threads available
    print(f'CPUs = {mp.cpu_count()}')
    pool = mp.Pool(mp.cpu_count())

    # Have each thread execute on a subset of the various alphas
    output = pool.starmap(run_experiment, args)
    out_lists = [list(g) for _, g in groupby(output, itemgetter(0))]
    pool.close()

    # Plot results
    if clamped:
        colours = ["blue", "purple", "black"]
        text = ', clamped'
    else:
        colours = ["red", "orange", "green"]
        text = ', not clamped'

    for colour, tup_list in zip(colours, out_lists):
        prob_vals = [tup[2] for tup in tup_list]
        plt.plot(alphaset, prob_vals, c=colour,
                 label="N= " + str(tup_list[0][0]) + text)

    plt.legend(title='Number of dimensions')
    plt.title(r'Probability of linear separability depending on $\alpha$')
    plt.xlabel(r'$\alpha$ defined as the ratio of Datapoints per Dimension')
    plt.ylabel('Probability of being linearly seperable (%)')
    plt.show()


if __name__ == '__main__':
    collect_data()
    pass
