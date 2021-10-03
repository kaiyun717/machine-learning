"""
UC Berkeley CS 189 - Homework 2
Part 1.3: Deliverables

Author: Kai Yun (kaiyun799@berkeley.edu)
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from homework2.data.get_mnist_data import get_mnist_threes_nines
from homework2.finite_difference import nn_finite_difference_checker
from homework2.neural_network import FullyConnectedMLP, forward_pass, loss_fn, backward_pass
from homework2.feed_forward import sigmoid_activation, logistic_loss

if __name__ == "__main__":
    # =================== MNIST Data Processing ====================
    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
    num_train_data = X_train.shape[0]
    X_train_each_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))

    num_test_data = X_test.shape[0]
    X_test_each_size = X_test.shape[1] * X_test.shape[2]
    X_test_flat = np.reshape(X_test, (num_test_data, X_test_each_size))

    # ==========================
    # ======== Part (a) ========
    # ==========================
    print("\n ============= PART (A) =============\n")
    test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data",
                                                  "test_batch_weights_biases.pkl"))
    with open(test_data_path, "rb") as fn:
        (X_batch, y_batch, weight_matrices, biases) = pickle.load(fn)

    activations = ["relu", "sigmoid"]
    weight_finite, biases_finite = \
        nn_finite_difference_checker(X_batch, y_batch, weight_matrices, biases, activations)

    with np.printoptions(precision=2):
        print(f"Gradient of Weight[0]:\n\t{weight_finite[0]}\n"
              f"Gradient of Weight[1]:\n\t{weight_finite[1]}\n"
              f"Gradient of Bias[0]:\n\t{biases_finite[0]}\n"
              f"Gradient of Bias[1]:\n\t{biases_finite[1]}")

    # ==========================
    # ======== Part (b) ========
    # ==========================
    print("\n ============= PART (B) =============\n")

    # Part (i)
    sigmoid_output_i = sigmoid_activation(input=np.asarray([1., 0., -1.]))
    print(f"(i) Sigmoid Values for [1., 0., -1.]: {sigmoid_output_i}")

    # Part (ii)
    # If x < 0, use sig(x) = e^x / (1 + e^x),
    # otherwise, use sig(x) = 1 / (1 + e^-x).
    sigmoid_output_ii = sigmoid_activation(input=np.asarray([-1000, 1000]))
    print(f"(i) Sigmoid Values for [-1000, 1000]: {sigmoid_output_ii}")

    # Part (iii)
    # Answered in paper.

    # Part (iv)
    # Answered in paper.

    # ==========================
    # ======== Part (c) ========
    # ==========================
    print("\n ============= PART (C) =============\n")

    activations = ["relu", "sigmoid"]
    output_pickle, layer_cahces_pickle = forward_pass(X_batch=X_batch,
                                                      weight_matrices=weight_matrices,
                                                      biases=biases,
                                                      activations=activations)
    loss_pickle, dL_dg_pickle = logistic_loss(g=output_pickle, y=y_batch)
    print(f"Average logistic loss of sample test dataset: {loss_pickle.mean()}")

    # ==========================
    # ======== Part (d) ========
    # ==========================
    print("\n ============= PART (D) =============\n")
    pickle_dL_dw, pickle_dL_db = backward_pass(dL_dg_pickle, layer_cahces_pickle)
    with np.printoptions(precision=2):
        print(f"Gradient of Weight[0] after Back-Propagation:\n\t{pickle_dL_dw[0]}\n"
              f"Gradient of Weight[1] after Back-Propagation:\n\t{pickle_dL_dw[1]}\n"
              f"Gradient of Bias[0] after Back-Propagation:\n\t{pickle_dL_db[0]}\n"
              f"Gradient of Bias[1] after Back-Propagation:\n\t{pickle_dL_db[1]}")

    # ==========================
    # ======== Part (e) ========
    # ==========================
    layer_dims: list = [X_train_each_size, 200, 1]
    activations: list = ["relu", "sigmoid"]

    fc_mlp_e = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

    average_train_losses, average_test_losses, average_train_accuracy, average_test_accuracy \
        = fc_mlp_e.train(X_train=X_train_flat, y_train=y_train, X_test=X_test_flat, y_test=y_test)

    plt.plot(average_train_losses)
    plt.show()

    plt.plot(average_test_losses)
    plt.show()

    plt.plot(average_train_accuracy)
    plt.show()

    plt.plot(average_test_accuracy)
    plt.show()

    print(f"Final Test Accuracy: {average_test_accuracy[-1]}")
