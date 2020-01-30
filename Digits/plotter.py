import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('resultsLR', 'rb') as f:
        data = pkl.load(f)
    print(len(data))
    print(len(data[0]))

    x = np.linspace(-1, 1, 21)
    y = np.linspace(-1, 1, 21)
    z = np.array([i*i+j*j for j in y for i in x])

    X, Y = np.meshgrid(x, y)
    Z = z.reshape(21, 21)

    plt.pcolor(X, Y, Z)
    plt.show()