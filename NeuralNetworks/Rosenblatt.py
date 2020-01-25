"""A Rosenblatt Perceptron and means of visualizing its behaviour."""

import multiprocessing as mp
from itertools import groupby
from operator import itemgetter
import perceptron
import percplotter

import matplotlib.pyplot as plt
import numpy as np


def run_rosenblatt(N=2, P=5, n_max=5, clamped=True, verbose=False):
    """Rosenblatt learning algorithm.

    N is the number of dimensions
    P is the number of datapoints
    n_max is the number of Epochs to run for

    if verbose is set to True, it will print stuff.
    """
    # Generate data and weights
    xi, labels, weights = perceptron.generate_data(P, N, clamped=clamped)

    # Initialize plotter, if applicable
    plotter = percplotter.plot(xi, labels)
    next(plotter)

    # Epoch loop
    for epoch in range(n_max):
        if verbose:
            print(f'Epoch {epoch}/{n_max}')  # noqa

        # Data loop
        stop_early = True
        for xi_t, label_t in zip(xi, labels):
            # Get the error via Hebbian learning:
            # If response == label: Error = 1, else Error = -1
            E_t = perceptron.sign(weights, xi_t) * label_t

            # If Error == -1, update weights and don't stop early
            if E_t == -1:
                weights += (xi_t * label_t) / N
                stop_early = False

            # Send the new weights to the plotter
            plotter.send(weights)

        # If we haven't updated any weight in this data loop, success
        if stop_early:
            return True, weights

    # If the stop early condition never happened, failure
    return False, weights


# Functions to execute the actions that individual threads need to take
def run_experiment(alpha, N, clamped):
    Pa = 0
    repetitions = 100
    for i in range(repetitions):
        P = int(alpha * N)
        result, _ = run_rosenblatt(N=N, P=P, n_max=100, clamped=clamped)
        Pa += int(result)

    return N, alpha, Pa / repetitions


def collect_data(clamped=False):
    # Create the arguments to run
    alphaset = np.arange(0.75, 5, 0.1)
    dimensions = [5]  # , 20, 150]  # [150, 20, 5]
    args = [(a, N, clamped) for N in dimensions for a in alphaset]

    # Determine the number of threads available
    print(f'CPUs = {mp.cpu_count()}')
    pool = mp.Pool(mp.cpu_count())

    # Have each thread execute on a subset of the various alphas
    output = pool.starmap(run_experiment, args)
    out_lists = [list(g) for _, g in groupby(output, itemgetter(0))]
    print(out_lists)
    pool.close()

    # Plot results
    plt.ion()
    if clamped:
        colours = ["blue", "purple", "black"]
    else:
        colours = ["red", "orange", "green"]
    for colour, tup_list in zip(colours, out_lists):
        prob_vals = [tup[2] for tup in tup_list]
        if clamped:
            plt.plot(alphaset, prob_vals, c=colour,
                     label="N= " + str(tup_list[0][0]) + ", clamped")
        else:
            plt.plot(alphaset, prob_vals, c=colour,
                     label="N= " + str(tup_list[0][0]) + ", not clamped")

    plt.legend(title='Number of dimensions')
    plt.title(r'Probability of linear separability depending on $\alpha$')
    plt.xlabel(r'$\alpha$ defined as the ratio of Datapoints per Dimension')
    plt.ylabel('Probability of being linearly seperable (%)')
    plt.show()


if __name__ == '__main__':
    collect_data()
    pass
