import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

if __name__ == '__main__':

    with open('KNN_fullsweep', 'rb') as f:
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
    performance_measure = 'MR'

    x_measure = 'noise_spread'
    y_measure = 'm'
    omitted_measure = 'knn_k'
    omitted_target = 5

    # Aggregate the performance results together:
    # Each datapoint has a list of results (k results from k-fold cross-val)
    if performance_measure in ['F1', 'prec', 'rec']:
        # These results come in pairs of (mean, std), we only want the mean
        res_list = [[res[performance_measure][0]
                     for res in dp[1]] for dp in data
                    if dp[0][omitted_measure] == omitted_target]
    else:
        # The other results are simple values
        res_list = [[res[performance_measure]
                     for res in dp[1]] for dp in data
                    if dp[0][omitted_measure] == omitted_target]

    # Plot the noise spread and the PCA-m on the x- and y- axes
    x = [dp[0][x_measure] for dp in data
         if dp[0][omitted_measure] == omitted_target]
    y = [dp[0][y_measure] for dp in data
         if dp[0][omitted_measure] == omitted_target]

    # Take the mean of each list of results to plot on the z-axis
    z = [sum(results) / len(results) for results in res_list]

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=np.negative(z))

    ax.set_xlabel(x_measure)
    ax.set_ylabel(y_measure)
    ax.set_zlabel(f'Performance ({performance_measure})')

    if omitted_measure != 'alpha':
        plt.title("Misclassification rates for "
                  + omitted_measure + " = " + str(omitted_target))
    else:
        plt.title("Misclassification rates")

    plt.show()
