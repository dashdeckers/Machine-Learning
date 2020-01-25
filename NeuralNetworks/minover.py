"""An implementation of the Minover algorithm."""

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp
from itertools import groupby
from operator import itemgetter

from percplotter import plot
from percutility import generalization_error, generate_data


def run_minover(P=5, N=2, t_max=100, clamped=False, verbose=False):
    """Minover algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Create a random teacher perceptron w_opt
    if clamped:
        w_opt = np.random.uniform(low=-1, high=1, size=N+1)
    else:
        w_opt = np.random.uniform(low=-1, high=1, size=N)
    # Normalize it so that |w|**2 = N i.e. |w| = sqrt(N)
    w_opt = w_opt / np.linalg.norm(w_opt) * np.sqrt(N)

    if P < 1:
        P = 1

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
            # Min stability = w .* xi_v * S_v / |w|

            # We can't determine a norm when w = 0
            if w.sum() == 0:
                stability = np.dot(w, xi_v * S_v)
            else:
                stability = np.dot(w, xi_v * S_v) / np.linalg.norm(w)

            if min_stability is None or stability < min_stability[2]:
                min_stability = (xi_v, S_v, stability)

        # Update the weight vector with (xi_v, S_v)
        min_xi_v, min_S_v, stability = min_stability
        w += (min_xi_v * min_S_v) / N

        # Send the new weights to the plotter
        plotter.send(w)

    return generalization_error(w, w_opt)


# Function to execute the actions that individual threads need to take
def run_experiment(alpha, N):
    pa = 0
    repetitions = 40
    for i in range(repetitions):
        p = int(alpha * N)
        result = run_minover(P=p, N=N)
        pa += result

    return N, alpha, pa / repetitions


def collect_data(clamped=False):
    # Create the arguments to run
    alphaset = np.arange(0.1, 10, 0.1)
    dimensions = [5, 20, 150]
    args = [(a, N) for N in dimensions for a in alphaset]

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
        # text = ', clamped'
    else:
        colours = ["red", "orange", "green"]
        # text = ', not clamped'

    for colour, tup_list in zip(colours, out_lists):
        prob_vals = [tup[2] for tup in tup_list]
        plt.plot(alphaset, prob_vals, c=colour,
                 label="N= " + str(tup_list[0][0]))

    plt.legend(title='Number of dimensions')
    plt.title(r'Learning curve depending on $\alpha$')
    plt.xlabel(r'$\alpha$ defined as the ratio of Datapoints per Dimension')
    plt.ylabel('Average generalization error')
    plt.show()
