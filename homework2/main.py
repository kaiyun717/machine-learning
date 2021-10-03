import os
import pickle

import numpy as np

from homework2.data.get_mnist_data import get_mnist_threes_nines
from homework2.neural_network import FullyConnectedMLP, forward_pass, backward_pass, loss_fn

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
    num_train_data = X_train.shape[0]
    X_train_each_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))
    print(f"X_train_each_size: {X_train_each_size}")
    layer_dims: list = [X_train_each_size, 256, 128, 1]
    activations: list = ["relu", "relu", "sigmoid"]

    fc_mlp = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

    output, layer_caches = fc_mlp.nn_forward_pass(X_batch=X_train_flat)

    y_estimate = np.reshape(output, (y_train.shape[0]))
    loss, dl_dg = fc_mlp.nn_loss_fn(y_estimate=y_estimate, y_batch=y_train)
    print("Average Loss Across Batch:", loss.mean(), "\n")

    fc_mlp.nn_backward_pass(dl_dg, layer_caches)

    # =========================
    # === Pickled Test Data ===
    # =========================
    test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data",
                                                  "test_batch_weights_biases.pkl"))
    with open(test_data_path, "rb") as fn:
        (X_batch, y_batch, weight_matrices, biases) = pickle.load(fn)
    activations = ["relu", "sigmoid"]

    test_forward_out, layer_caches = forward_pass(X_batch, weight_matrices, biases, activations)
    y_estimate = np.reshape(test_forward_out, (y_batch.shape[0]))
    loss, dl_dg = loss_fn(y_estimate, y_batch)

    test_dL_dw, test_dL_db = backward_pass(dl_dg, layer_caches)
    print(test_dL_dw)
    print(test_dL_db)
