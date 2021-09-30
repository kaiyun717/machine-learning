import numpy as np

from homework2.data.get_mnist_data import get_mnist_threes_nines
from homework2.neural_network import FullyConnectedMLP

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
    num_train_data = X_train.shape[0]
    X_train_each_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))

    layer_dims: list = [X_train_each_size, 256, 128, 1]
    activations: list = ["relu", "relu", "sigmoid"]

    fc_mlp = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

    output, layer_caches = fc_mlp.forward_pass(X_batch=X_train_flat)
    print(output)
    print(output.shape)

    y_estimate = np.reshape(output, (y_train.shape[0]))
    loss, dl_dg = fc_mlp.loss_fn(y_estimate=y_estimate, y_batch=y_train)
    print("Average Loss Across Batch:", loss.mean())
