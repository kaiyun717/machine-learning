"""
UC Berkeley CS 189 Fall 2021
Homework 1, Question 6
Eigenvectors of the Gaussian Covariance Matrix

Author: Kai Yun

* Citations:
    - https://stackoverflow.com/questions/17990845/how-to-equalize-the-scales-of-x-axis-and-y-axis-in-matplotlib
    - https://stackoverflow.com/questions/52781433/how-to-get-the-unit-vector-from-a-numpy-array
    - https://stackoverflow.com/questions/42281966/how-to-plot-vectors-in-python-using-matplotlib/42284007
    - https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
"""

import math

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    np.random.seed(seed=9)

    mu_1 = 3
    var_1 = 9
    std_1 = math.sqrt(var_1)

    mu_2 = 4
    var_2 = 4
    std_2 = math.sqrt(var_2)

    sample_1 = np.random.normal(mu_1, std_1, 100)
    sample_2 = 1/2 * sample_1 + np.random.normal(mu_2, std_2, 100)

    # Part (a)
    mean_1 = np.mean(sample_1)
    mean_2 = np.mean(sample_2)

    print(f"Mean of sample: {mean_1, mean_2}\n")

    # Part (b)
    sample_matrix = np.array([sample_1, sample_2])
    covariance = np.cov(sample_matrix)

    print(f"Covariance (2x2):\n"
          f"{covariance}\n")

    # Part (c)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    print(f"Eigenvalues:  {eigenvalues}\n"
          f"Eigenvectors: {eigenvectors}\n")

    # Part (d)(i)
    plt.plot(sample_1, sample_2, 'ro', markersize=3)

    # Part (d)(ii)
    origin = [mean_1, mean_2]

    unit_eigenvector_1 = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    unit_eigenvector_2 = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])

    plt.quiver(*origin, *unit_eigenvector_1, color=['b'], scale=eigenvalues[0])
    plt.quiver(*origin, *unit_eigenvector_2, color=['g'], scale=eigenvalues[1])

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title("Eigenvectors of the Gaussian Covariance Matrix")
    plt.xlabel("$X_{1}$")
    plt.ylabel("$X_{2}$")
    plt.show()

    # Part (e)
    rotation_matrix = eigenvectors          # Matrix U

    centered_sample_1 = sample_1 - mean_1   # Centered X1
    centered_sample_2 = sample_2 - mean_2   # Centered X2
    centered_sample = np.array([sample_1, sample_2])    # Shape: 2 x 100

    rotated_sample = rotation_matrix.T.dot(centered_sample)

    rotated_sample_1 = rotated_sample[0, :]
    rotated_sample_2 = rotated_sample[1, :]

    plt.plot(rotated_sample_1, rotated_sample_2, 'bo', markersize=3)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title("Rotated Sample Points")
    plt.xlabel("$X_{1}$")
    plt.ylabel("$X_{2}$")
    plt.show()


