"""
UC Berkeley CS 189 - Homework 2
Part 1.2(b)
Functions needed to implement the feed-forward part of training / testing.
    - logistic loss
    - sigmoid activation
    - relu activation
    - layer forward

Author: Kai Yun (kaiyun799@berkeley.edu)
"""
import numpy as np


def logistic_loss(g: np.array, y: np.array):
    """
    Implementation of logistic loss function; element-wise calculation.
        Loss = - log(g[i] ^ y[i] * (1 - g[i]) ^ (1 - y[i]))
        dL_dg = - y[i] / g[i] + (1 - y[i]) / (1 - g[i])
    :param g: Estimated values of y.
    :param y: True y values.
    :return: Loss and dL_dg (gradients of loss values) as defined above.
    """
    assert g.shape == y.shape, "Array of the outputs of the neural network and " \
                               "array of the true labels do not have the same shape."

    # g = np.where(g < 1e-324, g, 1e-323)
    loss = -np.multiply(y, np.log(g)) - np.multiply(1-y, np.log(1-g))
    g = np.array(g, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    dL_dg = -np.divide(y, g) + np.divide(1-y, 1-g)

    return loss, dL_dg


def relu_activation(input: np.array):
    """
    Relu activation function; element-wise calculation.
        Relu(x) = max(0, x)
        gradient(Relu(x)) = 1 if x > 0 and 0 otherwise
    :param input: `np.array` of x values
    :return: `np.array` of Relu(x) values and their gradients.
    """
    output = np.maximum(input, 0)
    gradient = (output > 0) * 1

    return output, gradient


def sigmoid_activation(input: np.array, eps: float = 1e-15):
    """
    Sigmoid activation function; element-wise calculation. Depending on the input value,
    difference forms are used.
        - If x < 0, sig(x) = e^x / (1 + e^x)
        - otherwise, sig(x) = 1 / (1 + e^-x)
    :param input: `np.array` of x values.
    :param eps: To avoid floating point limitations, clip the sigmoid values to (eps, 1 - eps).
    :return:
        - output: output of the sigmoid function given the input, element-wise
        - gradient: d_sigmoid = sigmoid * (1 - sigmoid), element-wise
    """
    output = np.where(input < 0,
                      np.exp(input, where=input < 0) / (1 + np.exp(input, where=input < 0)),
                      np.reciprocal(1 + np.exp(-input, where=input > 0)))
    output = np.clip(output, eps, 1 - eps)

    gradient = np.multiply(output, 1 - output)

    return output, gradient


def layer_forward(x, W, b, activation_fn):
    """
    Same function as `layer_forward`, but without incorporating biases into weight matrix W.
    :param x: n x d(l-1) dimensional matrix consisting of the mini-batch neurons for layer l-1
    :param W: d(l-1) x d(l) weight matrix for layer l
    :param b: 1 x d(l) dimensional basis vector for layer l
    :param activation_fn: Activation function (element-wise); Sigmoid or ReLU
    :return:
        - out: Output values (between (0, 1)) for each data set.
        - cache: Values needed for back-propagation.
    """
    assert x.shape[1] == W.shape[0], f"Number of columns of matrix x, {x.shape[1]}, does not match" \
                                     f"the number of rows of matrix W, {W.shape[0]}."
    assert b.shape[0] == 1, f"This bias vector b is not a row vector. It has {b.shape[0]} number of rows."
    assert W.shape[1] == b.shape[1], f"Number of columns of matrix W, {W.shape[1]}, does not match" \
                                     f"the number of columns of row vector b, {b.shape[1]}."\

    n = x.shape[0]          # Size of mini-batch, i.e., the number of datasets being trained.
    d_prev = x.shape[1]     # Dimension of previous layer l-1.
    d_curr = W.shape[1]     # Dimension of current layer l.

    b_matrix = np.tile(b, (n, 1))
    assert b_matrix.shape[0] == n, f"Basis matrix should have shape {n} x {d_curr}, not {b.shape[0]} x {b.shape[1]}."

    xW = np.dot(x, W)
    neuron_matrix = xW + b_matrix

    assert neuron_matrix.shape[0] == n and neuron_matrix.shape[1] == d_curr,     \
        "The dot product of matrices x and W should produce a sum matrix for the current layer of size" \
        f"{n} x {d_curr}. However, its shape is {neuron_matrix.shape}."

    out, neuron_activation_gradient = activation_fn(input=neuron_matrix)

    cache = (x, W, neuron_activation_gradient)

    return out, cache
