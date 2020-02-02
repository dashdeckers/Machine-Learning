import multiprocessing as mp
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from adatron import run_adatron
from minover import run_minover
from rosenblatt import run_rosenblatt

experiment1 = {
    'algorithms': ['rosenblatt'],
    'alphaset': np.arange(0.75, 5, 0.1),
    'dimensions': [5, 20, 150],
    'repetitions': 100,
    't_max': 100,
    'clamped': [True, False],
    'use_teacher': [False],

    'plot title': r'Probability of linear separability depending on $\alpha$',
    'x label': r'$\alpha$ defined as the ratio of Datapoints per Dimension',
    'y label': 'Probability of being linearly seperable (%)',
    'legend title': 'Number of dimensions',
}

experiment2 = {
    'algorithms': ['rosenblatt', 'minover'],
    'alphaset': np.arange(0.25, 6.25, 0.75),
    'dimensions': [5, 20, 150],
    'repetitions': 40,
    't_max': 100,
    'clamped': [False, True],
    'use_teacher': [True],

    'plot title': r'Learning curve depending on $\alpha$',
    'x label': r'$\alpha$ defined as the ratio of Datapoints per Dimension',
    'y label': 'Average generalization error',
    'legend title': 'Number of dimensions',
}

experiment3 = {
    'algorithms': ['minover', 'adatron'],
    'alphaset': np.arange(0.25, 6.25, 0.75),
    'dimensions': [5, 20, 150],
    'repetitions': 40,
    't_max': 30,
    'clamped': [False],
    'use_teacher': [True],

    'plot title': r'Learning curve depending on $\alpha$',
    'x label': r'$\alpha$ defined as the ratio of Datapoints per Dimension',
    'y label': 'Average generalization error',
    'legend title': 'Number of dimensions',
}


def get_label_and_vals(tup_list, experiment, algorithm):
    """Get the values and the labels from the tuple list.

    For each combination of clamped and use_teacher in [True, False],
    return a list of tuples that are ready to plot with.

    tup = (N, alpha, Pa / repetitions, clamped, use_teacher, algorithm)
    """
    clamped = experiment['clamped']
    use_teacher = experiment['use_teacher']
    N = str(tup_list[0][0])

    out = list()
    if True in clamped and True in use_teacher:
        label = f'N={N}, clamped+teacher ({algorithm})'
        prob_vals = [tup[2] for tup in tup_list
                     if tup[3] and tup[4] and tup[5] == algorithm]
        out.append((label, prob_vals))

    if True in clamped and False in use_teacher:
        label = f'N={N}, clamped+random ({algorithm})'
        prob_vals = [tup[2] for tup in tup_list
                     if tup[3] and not tup[4] and tup[5] == algorithm]
        out.append((label, prob_vals))

    if False in clamped and True in use_teacher:
        label = f'N={N}, teacher ({algorithm})'
        prob_vals = [tup[2] for tup in tup_list
                     if not tup[3] and tup[4] and tup[5] == algorithm]
        out.append((label, prob_vals))

    if False in clamped and False in use_teacher:
        label = f'N={N}, random ({algorithm})'
        prob_vals = [tup[2] for tup in tup_list
                     if not tup[3] and not tup[4] and tup[5] == algorithm]
        out.append((label, prob_vals))

    return out


# Function to execute the actions that individual threads need to take
def run_experiment(alpha, N, algorithm, repetitions,
                   t_max, clamped, use_teacher):
    if algorithm == 'rosenblatt':
        alg = run_rosenblatt
    if algorithm == 'minover':
        alg = run_minover
    if algorithm == 'adatron':
        alg = run_adatron

    Pa = 0
    for i in range(repetitions):
        P = int(alpha * N)
        result = alg(P=P, N=N, t_max=t_max, clamped=clamped,
                     use_teacher=use_teacher)
        Pa += result

    return N, alpha, Pa / repetitions, clamped, use_teacher, algorithm


def collect_data(experiment):
    """Collect data from multiple experiments and plot the results."""
    algorithms = experiment['algorithms']
    repetitions = experiment['repetitions']
    t_max = experiment['t_max']
    clamped = experiment['clamped']
    use_teacher = experiment['use_teacher']
    dimensions = experiment['dimensions']
    alphaset = experiment['alphaset']

    # Create the arguments to run
    args = [(alpha, N, algorithm, repetitions, t_max, c, t)
            for N in dimensions
            for alpha in alphaset
            for c in clamped
            for t in use_teacher
            for algorithm in algorithms]

    # Determine the number of threads available
    print(f'CPUs = {mp.cpu_count()}')
    pool = mp.Pool(mp.cpu_count())

    # Have each thread execute on a subset of the various alphas
    output = pool.starmap(run_experiment, args)
    pool.close()

    # Plot results
    # Loop over groups of results for the same dimension (N)
    for tups in [list(g) for _, g in groupby(output, itemgetter(0))]:
        # Loop over groups of results depending on the algorithm used
        for algo in algorithms:
            # Loop over groups of results depending on clamped and use_teacher
            for label, prob_vals in get_label_and_vals(tups, experiment, algo):
                plt.plot(alphaset, prob_vals, label=label)

    plt.legend(title=experiment['legend title'])
    plt.title(experiment['plot title'])
    plt.xlabel(experiment['x label'])
    plt.ylabel(experiment['y label'])
    plt.show()
