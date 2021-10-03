"""
UC Berkeley CS 189 - Homework 2
Part 1.2(a)
Finite Difference Checker

Author: Kai Yun (kaiyun799@berkeley.edu)
"""
import copy
import numpy as np
from homework2.neural_network import forward_pass, loss_fn


def nn_finite_difference_checker(X_batch, y_batch, weight_matrices, biases, activations, eps=1e-5):
    """
    Checks if the back-propagation of neural network is accurately calculating the gradients
    by utilizing finite difference:
        For function f: R^d --> R, the finite difference is
             df         f(..., a_k + eps, ...) - f(..., a_k - eps, ...)
            ---- (a) = -------------------------------------------------
            dx_k                            2 * eps

    :param X_batch: Input data set.
    :param y_batch: Labels for each data set.
    :param weight_matrices: List of weight matrices that connect each layer.
    :param biases: List of bias vectors for each layer connection.
    :param activations: List of activation function names as strings for each layer.
    :param eps: Epsilon value for finite difference checker. Default to 1e-5.
    :return: Finite difference values for each weight and bias.
    """
    y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))
    L = len(weight_matrices)    # number of weight matrices
    B = len(biases)             # number of bias vectors

    weight_diff = copy.deepcopy(weight_matrices)
    biases_diff = copy.deepcopy(biases)

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

                plus_loss, _ = loss_fn(y_estimate=plus_output, y_batch=y_batch)
                minus_loss, _ = loss_fn(y_estimate=minus_output, y_batch=y_batch)

                weight_diff[l][n][w] = (plus_loss.mean() - minus_loss.mean()) / (2*eps)

    for l in range(B):
        bias_vector = biases[l].T
        num_biases = bias_vector.shape[0]
        for b in range(num_biases):
            bias_vector_plus = copy.deepcopy(biases)
            bias_vector_minus = copy.deepcopy(biases)
            np.transpose(bias_vector_plus[l])[b] += eps
            np.transpose(bias_vector_minus[l])[b] -= eps

            plus_output, _ = forward_pass(X_batch=X_batch, weight_matrices=weight_matrices,
                                          biases=bias_vector_plus, activations=activations)
            minus_output, _ = forward_pass(X_batch=X_batch, weight_matrices=weight_matrices,
                                           biases=bias_vector_minus, activations=activations)

            plus_loss, _ = loss_fn(y_estimate=plus_output, y_batch=y_batch)
            minus_loss, _ = loss_fn(y_estimate=minus_output, y_batch=y_batch)

            np.transpose(biases_diff[l])[b] = (plus_loss.mean() - minus_loss.mean()) / (2*eps)

    return weight_diff, biases_diff
