import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def least_squares_fit(x, y, degree):
    """
    y = Xw where w is the weight vector.
    :param x: feature vector of masses (size n x 1)
    :param y: `y_train` data
    :param degree: Degree of the polynomial
    :return: w
                Weight vector of size (degree + 1) x 1
             y_estimate
                Estimate of the y-values using the newly acquired weights and features.
    """
    n = x.shape[0]
    X = np.zeros((n, degree + 1))

    for i in range(n):
        X[i] = [x[i]**degree for degree in range(degree+1)]

    w = np.linalg.pinv(X).dot(y)
    y_estimate = np.dot(X, w)

    return w, y_estimate


def training_error(x, y):
    """
    Returns the list of degrees and list of errors.
    :param x: `x_train` data
    :param y: `y_train` data
    :return: D
                List of degrees.
             errors
                List of errors, each corresponding to each degree, in order.
    """
    x = x.T
    n = x.shape[0]
    errors = []

    D = [i for i in range(n)]
    for d in D:
        w_d, f_d = least_squares_fit(x=x, y=y, degree=d)
        error = np.sum(np.square(y - f_d)) / n
        errors.append(error)

    return D, errors


def plot_training_errors(x, y_list, title):
    """
    Plots the average training errors as a function of degree D for each y-data given in the y_list.
    :param x: `x_train` data
    :param y_list: (list of) y-data
    :param title: title of the plot
    :return: N/A (Plots the average training errors)
    """
    xticks = []
    for y in y_list:
        D, errors = training_error(x, y)
        plt.plot(D, errors, "o")

        if len(D) > len(xticks):
            xticks = D

    plt.xticks(xticks)
    plt.title(title)
    plt.xlabel("Degrees")
    plt.ylabel("Training Error")
    plt.show()


if __name__ == "__main__":

    poly_mat_dir = os.path.join("hw2_data_code", "1D_poly.mat")
    poly_1d = loadmat(poly_mat_dir)
    x_train = poly_1d["x_train"]

    # == Part (b) ==
    y_train = poly_1d["y_train"]
    title_b = "Average Training Error for Different Polynomial Degrees\n" \
              "CS 189 - Homework 1, Q5(b)"
    plot_training_errors(x=x_train, y_list=[y_train], title=title_b)

    # == Part (d) ==
    y_fresh = poly_1d["y_fresh"]
    title_d = "y_train VS y_fresh\n" \
              "CS 189 - Homework 1, Q5(d)"
    plot_training_errors(x=x_train, y_list=[y_train, y_fresh], title=title_d)

    # print("y_tra in      |   y_fresh")
    # for i in range(len(y_fresh)):
    #     print(f"{y_train[i]}   {y_fresh[i]}")
