"""
UC Berkeley CS 189 - Homework 2
Part 1.2(a)
Finite Difference Checker

Author: Kai Yun (kaiyun799@berkeley.edu)
"""
import copy
import numpy as np
from homework2.neural_network import forward_pass, loss_fn


def partial_derivative_approx(func, a, index, eps=1e-5):
    a_1 = np.copy(a)
    a_2 = np.copy(a)
    a_1[index] = a[index] + eps
    a_2[index] = a[index] - eps
    a_1_out, _ = func(a_1)
    a_2_out, _ = func(a_2)

    partial_approx = (a_1_out - a_2_out) / (2*eps)

    return partial_approx


# Should implement this with Numpy and get rid of the `for-loop`.
def finite_difference(func, a, eps=1e-5):
    a_length = a.shape[0]       # `a` is a vector
    a_diff = np.zeros(a_length)

    for i in range(a_length):
        a_diff[i] = partial_derivative_approx(func=func, a=a, index=i, eps=eps)

    return a_diff


def nn_finite_difference_checker(X_batch, y_batch, weight_matrices, biases, activations, eps=1e-5):
    y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))
    L = len(weight_matrices)    # number of weight matrices

    weight_diff = copy.deepcopy(weight_matrices)
    # biases_diff = np.copy(biases)

    for l in range(L):
        weight_matrix = weight_matrices[l]
        num_neurons = weight_matrix.shape[0]
        for n in range(num_neurons):
            weights = weight_matrix[n]
            num_weights = weights.shape[0]
            for w in range(num_weights):
                weight_matrices_plus = copy.deepcopy(weight_matrices)
                weight_matrices_minus = copy.deepcopy(weight_matrices)
                weight_matrices_plus[l][n][w] += eps
                weight_matrices_minus[l][n][w] -= eps

                plus_output, _ = forward_pass(X_batch=X_batch, weight_matrices=weight_matrices_plus,
                                              biases=biases, activations=activations)
                minus_output, _ = forward_pass(X_batch=X_batch, weight_matrices=weight_matrices_minus,
                                               biases=biases, activations=activations)

                print(plus_output)
                print(minus_output)

                plus_loss, _ = loss_fn(y_estimate=plus_output, y_batch=y_batch)
                minus_loss, _ = loss_fn(y_estimate=minus_output, y_batch=y_batch)

                weight_diff[l][n][w] = (plus_loss.mean() - minus_loss.mean()) / (2*eps)


    print(weight_diff)


