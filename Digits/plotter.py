import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

if __name__ == '__main__':
    with open('LR_m70', 'rb') as f:
        data = pkl.load(f)

    """
    data = [datapoint]
    datapoint = (params, results)
    params = {'param': value}
    results = [single_result]
    single_result = {'measure': value}
    """

    # Set the performance measure to plot.
    # Can be one of: ['prec', 'rec', 'F1', 'acc', 'MR']
    measure = 'acc'

    # Aggregate the performance results together:
    # Each datapoint has a list of results (k results from k-fold cross-val)
    if measure in ['F1', 'prec', 'rec']:
        # These results come in pairs of (mean, std), we only want the mean
        res_list = [[res[measure][0] for res in dp[1]] for dp in data]
    else:
        # The other results are simple values
        res_list = [[res[measure] for res in dp[1]] for dp in data]

    # Plot the noise spread and the PCA-m on the x- and y- axes
    x = [dp[0]['noise_spread'] for dp in data]
    y = [dp[0]['m'] for dp in data]

    # Take the mean of each list of results to plot on the z-axis
    z = [sum(results) / len(results) for results in res_list]

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=np.negative(z))
    ax.set_xlabel('Noise spread')
    ax.set_ylabel('Number of Principle Components')
    ax.set_zlabel(f'Performance ({measure})')
    plt.show()
