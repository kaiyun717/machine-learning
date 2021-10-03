import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from homework2.data.get_mnist_data import get_mnist_threes_nines
from homework2.neural_network import FullyConnectedMLP, forward_pass, backward_pass, loss_fn

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
    num_train_data = X_train.shape[0]
    X_train_each_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))
    # print(f"X_train_each_size: {X_train_each_size}")
    layer_dims: list = [X_train_each_size, 256, 128, 1]
    activations: list = ["relu", "relu", "sigmoid"]

    fc_mlp = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

    output, layer_caches = fc_mlp.nn_forward_pass(X_batch=X_train_flat)

    y_estimate = np.reshape(output, (y_train.shape[0]))
    loss, dl_dg = fc_mlp.nn_loss_fn(y_estimate=y_estimate, y_batch=y_train)
    # print("Average Loss Across Batch:", loss.mean(), "\n")

    _, _ = fc_mlp.nn_backward_pass(dl_dg, layer_caches)

    # ================
    # === Part (e) ===
    # ================
    num_test_data = X_test.shape[0]
    X_test_each_size = X_test.shape[1] * X_test.shape[2]
    X_test_flat = np.reshape(X_test, (num_test_data, X_test_each_size))

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

    print(f"Final Train Accuracy: {average_train_accuracy[-1]}\n"
          f"Final Test Accuracy: {average_test_accuracy[-1]}")
