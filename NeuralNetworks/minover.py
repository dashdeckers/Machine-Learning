"""An implementation of the Minover algorithm."""

import multiprocessing as mp
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from percplotter import plot
from percutility import generalization_error, generate_data
from rosenblatt import run_rosenblatt


def run_minover(P=5, N=2, t_max=100, clamped=False, verbose=False,
                random_labeler=False, Rosenblatt=False):
    """Minover algorithm.

    N is the number of dimensions
    P is the number of datapoints
    t_max is the maximum number of Epochs to run for
    """
    # Create a random teacher perceptron w_opt and normalize it so
    # that: |w|^2 = N  <==>  |w| = sqrt(N)
    if clamped:
        w_opt = np.random.uniform(low=-1, high=1, size=N+1)
        w_opt = w_opt / np.linalg.norm(w_opt) * np.sqrt(N+1)
    else:
        w_opt = np.random.uniform(low=-1, high=1, size=N)
        w_opt = w_opt / np.linalg.norm(w_opt) * np.sqrt(N)

    if P < 1:
        P = 1

    # Generate data and weights
    if random_labeler:
        xi, S, w = generate_data(P, N, clamped=clamped)
    else:
        xi, S, w = generate_data(P, N, w_opt=w_opt, clamped=clamped)

    if Rosenblatt:
        data = (xi, S, w)
        (success, w) = run_rosenblatt(P, N, t_max, clamped, verbose, data=data)
        return generalization_error(w, w_opt)

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
            if random_labeler and sum(w) != 0:
                stability = np.dot(w, xi_v) * S_v / np.linalg.norm(w)

            if min_stability is None or stability < min_stability[2]:
                min_stability = (xi_v, S_v, stability)

        # Update the weight vector with v = (xi_v, S_v)
        min_xi_v, min_S_v, stability = min_stability
        w += (min_xi_v * min_S_v) / N

        # Send the new weights to the plotter
        plotter.send(w)

    if random_labeler:
        return min_stability[2]
    else:
        return generalization_error(w, w_opt)


# Function to execute the actions that individual threads need to take
def run_experiment(alpha, N, random_labeler, Rosenblatt=False):
    Pa = 0
    repetitions = 40
    for i in range(repetitions):
        P = int(alpha * N)
        result = run_minover(P=P, N=N, random_labeler=random_labeler,
                             Rosenblatt=Rosenblatt)
        Pa += result

    return N, alpha, Pa / repetitions


def collect_data(random_label=False, Rosenblatt=False):
    # Create the arguments to run
    alphaset = np.arange(0.25, 6.25, 0.75)
    dimensions = [5, 20, 150]
    args = [(a, N, random_label, Rosenblatt)
            for N in dimensions for a in alphaset]

    # Determine the number of threads available
    print(f'CPUs = {mp.cpu_count()}')
    pool = mp.Pool(mp.cpu_count())

    # Have each thread execute on a subset of the various alphas
    output = pool.starmap(run_experiment, args)
    print(output)
    out_lists = [list(g) for _, g in groupby(output, itemgetter(0))]
    print(out_lists)
    pool.close()

    # Plot results
    if Rosenblatt:
        colours = ["blue", "purple", "black"]
        text = ', Rosenblatt'
    else:
        colours = ["red", "orange", "green"]
        text = ', Minover'

    for colour, tup_list in zip(colours, out_lists):
        prob_vals = [tup[2] for tup in tup_list]
        plt.plot(alphaset, prob_vals, c=colour,
                 label="N= " + str(tup_list[0][0]) + text)

    plt.legend(title='Number of dimensions')
    plt.title(r'Learning curve depending on $\alpha$')
    plt.xlabel(r'$\alpha$ defined as the ratio of Datapoints per Dimension')
    plt.ylabel('Average generalization error')

    if Rosenblatt:
        collect_data(False, False)
        plt.show()
    else:
        plt.show()
