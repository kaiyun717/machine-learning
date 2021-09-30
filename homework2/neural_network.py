import numpy as np

from feed_forward import sigmoid_activation, relu_activation, layer_forward, logistic_loss


class FullyConnectedMLP:
    def __init__(self, layer_dims: list, activations: list):
        """
        Implementation of "Fully Connected Multi-Layer Perception Neural Network".
        :param layer_dims: list of layer dimensions.
        :param activations: list of activation functions for each layer.
        """
        self.layer_dims = layer_dims
        self.activations = activations
        self.nn_depth = len(layer_dims)

        self.mean = 0
        self.std_dev = 0.01

        self.weight_matrices = self.create_weight_matrices(layer_dims=layer_dims)
        self.bias_vectors = self.create_bias_vectors(layer_dims=layer_dims)

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

    def forward_pass(self, X_batch):
        output, layer_caches = self._forward_pass(X_batch=X_batch,
                                                  weight_matrices=self.weight_matrices,
                                                  biases=self.bias_vectors,
                                                  activations=self.activations)
        return output, layer_caches

    def backward_pass(self, loss_func, layer_caches):
        pass

    @staticmethod
    def loss_fn(y_estimate, y_batch, name="logistic"):
        if name == "logistic":
            loss_fn = logistic_loss
        loss, dl_dg = loss_fn(g=y_estimate, y=y_batch)
        return loss, dl_dg

    @staticmethod
    def _forward_pass(X_batch, weight_matrices, biases, activations):
        layer_caches = []

        nn_depth = len(weight_matrices)
        for layer in range(nn_depth):
            W = weight_matrices[layer]
            b = biases[layer]

            activation_fn_name = activations[layer]
            if activation_fn_name == "sigmoid":
                activation_fn = sigmoid_activation
            else:   #`activation_fn_name == "relu"`
                activation_fn = relu_activation

            X_batch, caches = layer_forward(x=X_batch, W=W, b=b, activation_fn=activation_fn)
            layer_caches.append(caches)
        output = X_batch
        return output, layer_caches

    def _backward_pass(self, dL_dg, layer_caches):
        pass

    def train(self):
        pass
