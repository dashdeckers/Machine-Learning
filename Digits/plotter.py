import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    with open('resultsLR', 'rb') as f:
        data = pkl.load(f)

    results_set = np.zeros((len(data), 3))
    MR_means = []
    for i, datapoint in enumerate(data):
        mean = 0
        for i in range(0, len(datapoint[1])):
            mean += datapoint[1][i]['MR']
        mean /= len(datapoint[1])
        MR_means.append(mean)
        results_set[i] = [datapoint[0]['noise_spread'], datapoint[0]['m'], mean]

    x = [datapoint[0]['noise_spread'] for datapoint in data]
    y = [datapoint[0]['m'] for datapoint in data]
    z = MR_means

    #plt.scatter(x, y, c=z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=np.negative(z))
    ax.set_xlabel('Noise spread')
    ax.set_ylabel('Number of Principle Components')
    ax.set_zlabel('Misclassification rate')

plt.show()
