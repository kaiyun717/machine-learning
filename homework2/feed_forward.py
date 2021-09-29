import numpy as np


def sigmoid_activation(s: np.array, eps: float = 1e-15):
    """

    :param input:
    :param eps:
    :return:
        - output: output of the sigmoid function given the input, element-wise
        - gradient: d_sigmoid = sigmoid * (1 - sigmoid), element-wise
    """
    s = np.where(s >= -709, s, -709)
    output = np.reciprocal(1 + np.exp(-s))
    output = np.clip(output, eps, 1 - eps)

    gradient = np.multiply(output, 1 - output)

    return output, gradient


def logistic_loss(g: np.array, y: np.array):
    """

    :param g:
    :param y:
    :return:
    """
    assert g.shape == y.shape, "Array of the outputs of the neural network and " \
                               "array of the true labels do not have the same shape."

    g = np.where(g < 1e-324, g, 1e-323)
    loss = -np.multiply(y, np.log(g)) - np.multiply(1-y, np.log(1-g))
    g = np.array(g, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    dl_dg = -np.divide(y, g) + np.divide(1-y, 1-g)

    return loss, dl_dg


def relu_activation(input: np.array):
    """

    :param input:
    :return:
    """
    output = np.maximum(input, 0)
    gradient = (output > 0) * 1

    return output, gradient


def layer_forward_incorporated(x, W, b, activation_fn):
    """

    :param x: n x d(l-1) dimensional matrix consisting of the mini-batch neurons for layer l-1
    :param W: d(l-1) x d(l) weight matrix for layer l
    :param b: 1 x d(l) dimensional basis vector for layer l
    :param activation_fn: activation function (element-wise); Sigmoid or ReLU
    :return:
        - out
        - cache
    """
    assert x.shape[1] == W.shape[0], f"Number of columns of matrix x, {x.shape[1]}, does not match" \
                                     f"the number of rows of matrix W, {W.shape[0]}."
    assert b.shape[0] == 1, f"This bias vector b is not a row vector. It has {b.shape[0]} number of rows."
    assert W.shape[1] == b.shape[1], f"Number of columns of matrix W, {W.shape[1]}, does not match" \
                                     f"the number of columns of row vector b, {b.shape[1]}."\

    n = x.shape[0]          # Size of mini-batch, i.e., the number of datasets being trained.
    d_prev = x.shape[1]     # Dimension of previous layer l-1.
    d_curr = W.shape[1]     # Dimension of current layer l.

    new_x = np.hstack([np.ones((n, 1)), x])
    new_W = np.vstack([b, W])
    neuron_matrix = np.dot(new_x, new_W)     # Matrix of S_j (summed input to node j) (Size: n x d_curr)

    assert neuron_matrix.shape[0] == n and neuron_matrix.shape[1] == d_curr, \
        "The dot product of matrices x and W should produce a sum matrix for the current layer of size" \
        f"{n} x {d_curr}. However, its shape is {neuron_matrix.shape}."

    out, activation_gradient = activation_fn(input=neuron_matrix)

    cache = (neuron_matrix, activation_gradient)

    return out, cache


def layer_forward(x, W, b, activation_fn):
    """
    Same function as `layer_forward`, but without incorporating biases into weight matrix W.
    :param x: n x d(l-1) dimensional matrix consisting of the mini-batch neurons for layer l-1
    :param W: d(l-1) x d(l) weight matrix for layer l
    :param b: 1 x d(l) dimensional basis vector for layer l
    :param activation_fn: activation function (element-wise); Sigmoid or ReLU
    :return:
        - out
        - cache
    """
    assert x.shape[1] == W.shape[0], f"Number of columns of matrix x, {x.shape[1]}, does not match" \
                                     f"the number of rows of matrix W, {W.shape[0]}."
    assert b.shape[0] == 1, f"This bias vector b is not a row vector. It has {b.shape[0]} number of rows."
    assert W.shape[1] == b.shape[1], f"Number of columns of matrix W, {W.shape[1]}, does not match" \
                                     f"the number of columns of row vector b, {b.shape[1]}."\

    n = x.shape[0]          # Size of mini-batch, i.e., the number of datasets being trained.
    d_prev = x.shape[1]     # Dimension of previous layer l-1.
    d_curr = W.shape[1]     # Dimension of current layer l.

    # b_array = [b for _ in range(n)]
    # b_matrix = np.stack(b_array, axis=0)
    b_matrix = np.tile(b, (n, 1))
    assert b_matrix.shape[0] == n, f"Basis matrix should have shape {n} x {d_curr}, not {b.shape[0]} x {b.shape[1]}."

    xW = np.dot(x, W)
    neuron_matrix = xW + b_matrix

    assert neuron_matrix.shape[0] == n and neuron_matrix.shape[1] == d_curr, \
        "The dot product of matrices x and W should produce a sum matrix for the current layer of size" \
        f"{n} x {d_curr}. However, its shape is {neuron_matrix.shape}."

    out, neuron_activation_gradient = activation_fn(input=neuron_matrix)
    _, xW_activation_gradient = activation_fn(input=xW)
    _, b_activation_gradient = activation_fn(input=b)

    cache = (neuron_matrix, neuron_activation_gradient,
             xW, xW_activation_gradient,
             b, b_activation_gradient)

    return out, cache
