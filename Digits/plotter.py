import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    with open('KNN_fullsweep', 'rb') as f:
        data = pkl.load(f)

    x_measure = 'noise_spread'
    y_measure = 'm'
    omitted_measure = 'alpha'
    omitted_target = 0

    results_set = np.zeros((len(data), 3))
    MR_means = []
    for i, datapoint in enumerate(data):
        if datapoint[0][omitted_measure] == omitted_target or omitted_measure == None:
            mean = 0
            for i in range(0, len(datapoint[1])):
                mean += datapoint[1][i]['MR']
            mean /= len(datapoint[1])
            MR_means.append(mean)

    x = [datapoint[0][x_measure] for datapoint in data if datapoint[0][omitted_measure] == omitted_target]
    y = [datapoint[0][y_measure] for datapoint in data if datapoint[0][omitted_measure] == omitted_target]
    z = MR_means

    # plt.scatter(x, y, c=z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=np.negative(z))
    ax.set_xlabel(x_measure)
    ax.set_ylabel(y_measure)
    ax.set_zlabel('Misclassification rate')
    plt.title("Misclassification rates for " +omitted_measure + " = " + str(omitted_target))

    # ax = fig.add_subplot(111)
    # ax.scatter(x, y, c=np.negative(z))
    # ax.set_xlabel("Variance of noise")
    # ax.set_ylabel("Number of principle components")
    # #ax.set_ylabel("Misclassficiation rate")
    # plt.title("Misclassification rates for Linear Regression with different numbers of principle components and levels of noise")

plt.show()
