import pickle as pkl

import numpy as np
import scipy.linalg  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def fit_plane(x, y, z, order=2):
    data = np.c_[x, y, z]
    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(20, 51, 0.5), np.arange(0, 3.1, 0.5))
    XX = X.flatten()
    YY = Y.flatten()

    order = 2    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])    # coefficients

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2],
                  np.prod(data[:, :2], axis=1), data[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY,
                         XX**2, YY**2], C).reshape(X.shape)
    return X, Y, Z


def K_planes(data):
    # Set the performance measure to plot.
    # Can be one of: ['prec', 'rec', 'F1', 'acc', 'MR']
    performance_measure = 'MR'

    x_measure = 'm'
    y_measure = 'noise_spread'
    # Only use this if you want the z-measure to be another parameter
    # z_measure = 'm'
    # To include all data use alpha 0 as omitted set
    omitted_measure = 'knn_k'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fakeLegendLines = []
    colours = ['r', 'g', 'b', 'y']
    for c, omitted_target in enumerate(range(3, 11, 2)):

        # Aggregate the performance results together:
        if performance_measure in ['F1', 'prec', 'rec']:
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
        # z = [dp[0][z_measure] for dp in data
        #      if dp[0][omitted_measure] == omitted_target]

        # Take the mean of each list of results to plot on the z-axis
        p = [sum(results) / len(results) for results in res_list]

        X, Y, Z = fit_plane(x, y, p, 2)

        # Plot the results

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        alpha=0.5, color=colours[c])
        fakeLegendLines.append(
            mpl.lines.Line2D([0], [0], linestyle="none",
                             c=colours[c], marker='o'))

    measure_names = {
        "noise_spread": "Variance of noise",
        "m": "Number of principle components",
        "knn_k": "Number of neighbors (k)",
        "MR": "Misclassification rate",
    }

    ax.set_xlabel(measure_names[x_measure])
    ax.set_ylabel(measure_names[y_measure])
    ax.set_zlabel(f'Performance ({measure_names[performance_measure]})',
                  labelpad=20)

    if omitted_measure != 'alpha' and False:
        plt.title("Misclassification rates for "
                  + measure_names[omitted_measure] +
                  " = " + str(omitted_target))
    else:
        plt.title("Misclassification rates for KNN")
    ax.legend(fakeLegendLines, ['k=3', 'k=5', 'k=7', 'k=9'])
    plt.show()


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
    performance_measure = 'MR'

    x_measure = 'm'
    y_measure = 'noise_spread'
    # Only use this if you want the z-measure to be another parameter
    # z_measure = 'm'
    # To include all data use alpha 0 as omitted set
    omitted_measure = 'knn_k'
    omitted_target = 9

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
    # z = [dp[0][z_measure] for dp in data
    #      if dp[0][omitted_measure] == omitted_target]

    # Take the mean of each list of results to plot on the z-axis
    p = [sum(results) / len(results) for results in res_list]

    X, Y, Z = fit_plane(x, y, p, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the results
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)

    cmap = mpl.cm.get_cmap('jet_r')
    plot = ax.scatter(x, y, p, c=p, cmap=cmap)

    measure_names = {
        "noise_spread": "Variance of noise",
        "m": "Number of principle components",
        "knn_k": "Number of neighbors (k)",
        "MR": "Misclassification rate",
    }

    ax.set_xlabel(measure_names[x_measure])
    ax.set_ylabel(measure_names[y_measure])
    ax.set_zlabel(f'Performance ({measure_names[performance_measure]})',
                  labelpad=20)
    # ax.set_zlabel(measure_names[z_measure])
    cbar = fig.colorbar(plot, orientation="horizontal",
                        label=f'Performance ('
                              f'{measure_names[performance_measure]})')

    if omitted_measure != 'alpha':
        plt.title("Misclassification rates for "
                  + measure_names[omitted_measure] +
                  " = " + str(omitted_target))
    else:
        plt.title("Misclassification rates for KNN")

    plt.show()
