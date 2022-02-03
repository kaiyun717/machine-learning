import matplotlib.pyplot as plt
import numpy as np
import os


def plot_data(path_to_file):
    # path_to_file = os.path.join(os.getcwd(), 'homwework4', 'hw4_data', 'circle.npz')
    data = np.load(path_to_file)
    # plt.figure()
    X = data['x']
    Y = data['y']

    for i in range(len(Y)):
        if Y[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], color="orange")
        else:
            plt.scatter(X[i, 0], X[i, 1], color="blue")
    plt.show()



if __name__ == "__main__":
    dir_path = 'C:\\Users\\kaiyu\\OneDrive\\Documents\\77.Git\\machine_learning\\homework4\\hw4_data\\'
    npz_files = ['circle.npz', 'heart.npz', 'asymmetric.npz']
    for i in range(len(npz_files)):
        plot_data(path_to_file=dir_path+npz_files[i])