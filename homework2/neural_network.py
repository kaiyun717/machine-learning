import numpy as np
from finite_difference import finite_difference
from feed_forward import sigmoid_activation, relu_activation, layer_forward, logistic_loss
from data.get_mnist_data import get_mnist_threes_nines


class FullyConnectedMLP:
    def __init__(self, layer_dims: list, activations: list):
        """
        Implementation of "Fully Connected Multi-Layer Perception Neural Network".
        :param layer_dims: list of layer dimensions.
        :param activations: list of activation functions for each layer.
        """
        self.layer_dims = layer_dims
        self.activations = activations
        self.mean = 0
        self.std_dev = 0.01

    def create_weight_matrices(self, layer_dims):
        """

        :param layer_dims:
        :return: list of weights; each index has a Numpy matrix of weights.
        """
        weights = []
        for layer in range(len(layer_dims) - 1):
            dim_curr = layer_dims[layer]            # Layer l - 1
            dim_next = layer_dims[layer + 1]        # Layer l
            weight_matrix = np.random.normal(loc=self.mean, scale=self.std_dev, size=(dim_curr, dim_next))
            weights.append(weight_matrix)

        return weights

    def create_bias_vectors(self, layer_dims):
        """

        :param layer_dims:
        :return: list of biases; each index has a Numpy vector of biases.
        """
        biases = []
        for layer in range(len(layer_dims) - 1):
            dim_next = layer_dims[layer + 1]        # Layer l
            bias_vector = np.random.normal(loc=self.mean, scale=self.std_dev, size=(1, dim_next))
            biases.append(bias_vector)

        return biases

    def _forward_pass(self, X_batch, weight_matrices, biases, activations):
        layer_caches = []
        output = None

        nn_depth = len(weight_matrices)
        for layer in range(nn_depth):
            if layer == 0:
                x = X_batch
            else:
                x = output

            W = weight_matrices[layer]
            b = biases[layer]

            activation_fn_name = activations[layer]
            if activation_fn_name == "sigmoid":
                activation_fn = sigmoid_activation
            else:   #`activation_fn_name == "relu"`
                activation_fn = relu_activation

            output, caches = layer_forward(x=x, W=W, b=b, activation_fn=activation_fn)
            layer_caches.append(caches)

        return output, layer_caches

    def backward_pass(self, loss_func, layer_caches):
        pass

    def forward_pass(self, X_batch):
        weight_matrices = self.create_weight_matrices(layer_dims=self.layer_dims)
        biases = self.create_bias_vectors(layer_dims=self.layer_dims)

        output, layer_caches = self._forward_pass(X_batch=X_batch,
                                                  weight_matrices=weight_matrices,
                                                  biases=biases,
                                                  activations=self.activations)
        return output, layer_caches

    def logistic_loss(self, y_estimate, y_batch):
        loss, dl_dg = logistic_loss(g=y_estimate, y=y_batch)
        return loss, dl_dg

    def _backward_pass(self):
        pass

    def train(self):
        pass


if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
    num_train_data = X_train.shape[0]
    X_train_each_size = X_train.shape[1] * X_train.shape[2]
    X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))

    layer_dims: list = [X_train_each_size, 256, 128, 1]
    activations: list = ["relu", "relu", "relu", "sigmoid"]

    fc_mlp = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

    output, layer_caches = fc_mlp.forward_pass(X_batch=X_train_flat)
    print(output)

    y_estimate = np.reshape(output, (y_train.shape[0]))
    loss, dl_dg = fc_mlp.logistic_loss(y_estimate=y_estimate, y_batch=y_train)
    print("Average Loss Across Batch:", loss.mean())
