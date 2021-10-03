import numpy as np

from homework2.feed_forward import sigmoid_activation, relu_activation, layer_forward, logistic_loss


def loss_fn(y_estimate, y_batch, name="logistic"):
    if name == "logistic":
        loss_fn = logistic_loss
    loss, dl_dg = loss_fn(g=y_estimate, y=y_batch)
    return loss, dl_dg


def forward_pass(X_batch, weight_matrices, biases, activations):
    layer_caches = []

    nn_depth = len(weight_matrices)
    for layer in range(nn_depth):
        W = weight_matrices[layer]
        b = biases[layer]

        activation_fn_name = activations[layer]
        if activation_fn_name == "sigmoid":
            activation_fn = sigmoid_activation
        else:  # `activation_fn_name == "relu"`
            activation_fn = relu_activation

        X_batch, caches = layer_forward(x=X_batch, W=W, b=b, activation_fn=activation_fn)
        layer_caches.append(caches)
    output = X_batch
    return output, layer_caches


def backward_pass(dL_dg, layer_caches):
    """

    :param dL_dg:
    :param layer_caches: length should be `self.nn_depth`.
    :return:
    """
    dL_dg = np.array(dL_dg, dtype=np.float64).reshape((len(dL_dg), 1))
    # print(f"Shape of dL_dg: {dL_dg.shape} (This is the loss vector: len = n)")
    # print(f"Number of weight matrices: {len(self.weight_matrices)}")
    # for i in range(len(self.weight_matrices)):
    #     print(f"Shape of weight matrix {i}: {self.weight_matrices[i].shape}")
    #
    # print(f"Number of layer caches: {len(layer_caches)}")
    # for i in range(len(layer_caches)):
    #     print(f"Shape of layer {i} xW_activation: {layer_caches[i][2].shape}")
    #     print(f"Shape of layer {i} xW_activation_gradient: {layer_caches[i][3].shape}")
    #     print(f"Shape of layer {i} b_activation: {layer_caches[i][4].shape}")
    #     print(f"Shape of layer {i} b_activation_gradient: {layer_caches[i][5].shape}")

    L = len(layer_caches) - 1  # Final layer (output) index
    print(f"L: {L}")
    dL_dw = []
    delta_next_weights = np.einsum('ij,ij->ij', dL_dg, layer_caches[L][2])
    dL_dw_final = np.dot(layer_caches[L][0].T, delta_next_weights)
    dL_dw.insert(0, dL_dw_final)

    dL_db = []
    # delta_next_bias =

    for l in range(L - 1, -1, -1):
        delta_next_weights = np.einsum("ij,ij->ij", np.dot(delta_next_weights, layer_caches[l + 1][1].T),
                                       layer_caches[l][2])
        dL_dw_l = np.dot(layer_caches[l][0].T, delta_next_weights)
        dL_dw.insert(0, dL_dw_l)

    for dd in dL_dw:
        print(dd.shape)

    return dL_dw, dL_db


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

    def nn_forward_pass(self, X_batch):
        output, layer_caches = forward_pass(X_batch=X_batch,
                                            weight_matrices=self.weight_matrices,
                                            biases=self.bias_vectors,
                                            activations=self.activations)
        return output, layer_caches

    def nn_backward_pass(self, dL_dg, layer_caches):
        dL_dw, dL_db = backward_pass(dL_dg=dL_dg, layer_caches=layer_caches)
        return dL_dw, dL_db

    def nn_loss_fn(self, y_estimate, y_batch, name="logistic"):
        if name == "logistic":
            loss_fn = logistic_loss
        loss, dl_dg = loss_fn(g=y_estimate, y=y_batch)
        return loss, dl_dg

    def train(self):
        pass
