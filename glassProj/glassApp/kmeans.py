import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def clus():
    df = pd.read_csv('glass.csv')
    df.head()

    X = df.iloc[:, [3, 4]].values
    # print (X)

    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of features. Here n=2n_iter=100

    K = 7
    n_iter = 100
    Centroids = np.array([]).reshape(n, 0)

    for i in range(K):
        rand = rd.randint(0, m - 1)
        Centroids = np.c_[Centroids, X[rand]]

    Output = {}

    EuclidianDistance = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    C = np.argmin(EuclidianDistance, axis=1) + 1

    for i in range(n_iter):
        # step 2.a
        EuclidianDistance = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
            EuclidianDistance = np.c_[EuclidianDistance, tempDist]
        C = np.argmin(EuclidianDistance, axis=1) + 1
        # step 2.b
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(2, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]

        for k in range(K):
            Y[k + 1] = Y[k + 1].T

        for k in range(K):
            Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        Output = Y

    color = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
    labels = ['bw_fp', 'bw_nfp', 'v', 'v_nfp', 'containers', 'tableware', 'headlamps']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[1, :], s=100, c='grey', label='Centroids')
    plt.xlabel('RI')
    plt.ylabel('Na')
    plt.legend()
    plt.show()
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in range(K):
        ax.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    ax.scatter(Centroids[0, :], Centroids[1, :], s=100, c='grey', label='Centroids')

    ax.set_xlabel('RI')
    ax.set_ylabel('Na')
    ax.set_zlabel('Mg')
    plt.legend()
    plt.show()
    import pickle
    pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

clus()


