import copy

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
            (x, W, neuron_matrix_gradient)
    :return:
    """
    dL_dg = np.array(dL_dg, dtype=np.float64).reshape((len(dL_dg), 1))
    L = len(layer_caches) - 1  # Final layer (output) index

    dL_dw = []
    delta_next_weights = np.einsum('ij,ij->ij', dL_dg, layer_caches[L][2])
    dL_dw_final = np.dot(layer_caches[L][0].T, delta_next_weights) / layer_caches[L][0].shape[0]
    dL_dw.insert(0, dL_dw_final)

    dL_db = []
    dL_db.insert(0, delta_next_weights.mean(axis=0))

    for l in range(L - 1, -1, -1):
        delta_next_weights = np.einsum("ij,ij->ij",
                                       np.dot(delta_next_weights, layer_caches[l + 1][1].T),
                                       layer_caches[l][2])
        dL_dw_l = np.dot(layer_caches[l][0].T, delta_next_weights) / layer_caches[L][0].shape[0]
        dL_dw.insert(0, dL_dw_l)
        dL_db.insert(0, delta_next_weights.mean(axis=0))

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

    def return_weight_matrices(self):
        return self.weight_matrices

    def return_bias_vectors(self):
        return self.bias_vectors

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

    def train(self, X_train, y_train, X_test, y_test, mini_batch_size=100, learning_rate=0.1, epoch=5):
        average_train_losses = []
        average_train_accuracy = []
        average_test_losses = []
        average_test_accuracy = []

        for i in range(epoch + 1):
            X_train_pre_shuffle = copy.deepcopy(X_train)
            y_train_pre_shuffle = copy.deepcopy(y_train)
            assert X_train_pre_shuffle.shape[0] == y_train_pre_shuffle.shape[0]

            shuffle_seq = np.random.permutation(len(X_train_pre_shuffle))
            X_train_shuffle = X_train_pre_shuffle[shuffle_seq]
            y_train_shuffle = y_train_pre_shuffle[shuffle_seq]

            X_train_shuffle_list = [X_train_shuffle[i: i + mini_batch_size] for i in range(0, X_train_shuffle.shape[0], 100)]
            y_train_shuffle_list = [y_train_shuffle[i: i + mini_batch_size] for i in range(0, y_train_shuffle.shape[0], 100)]

            for i in range(len(X_train_shuffle_list)):
                # === Train ===
                output, layer_caches = self.nn_forward_pass(X_batch=X_train_shuffle_list[i])
                y_train_estimate = np.reshape(output, (y_train_shuffle_list[i].shape[0]))

                loss, dl_dg = self.nn_loss_fn(y_estimate=y_train_estimate, y_batch=y_train_shuffle_list[i])
                average_train_losses.append(loss.mean())

                y_train_estimate_logit = (y_train_estimate > 0.5).astype(int)
                average_train_acc \
                    = np.sum(np.equal(y_train_estimate_logit, y_train_shuffle_list[i])) / y_train_shuffle_list[i].shape[0]
                average_train_accuracy.append(average_train_acc)

                # === Test ===
                output_test, _ = self.nn_forward_pass(X_batch=X_test)
                y_test_estimate = np.reshape(output_test, (y_test.shape[0]))

                loss_test, _ = self.nn_loss_fn(y_estimate=y_test_estimate, y_batch=y_test)
                average_test_losses.append(loss_test.mean())

                y_test_estimate_logit = (y_test_estimate > 0.5).astype(int)
                average_test_acc = np.sum(np.equal(y_test_estimate_logit, y_test)) / y_test.shape[0] * 100
                average_test_accuracy.append(average_test_acc)

                # === Train ===
                dL_dw, dL_db = self.nn_backward_pass(dL_dg=dl_dg, layer_caches=layer_caches)
                self.weight_matrices = [self.weight_matrices[i] - learning_rate * dL_dw[i]
                                        for i in range(len(self.weight_matrices))]
                self.bias_vectors = [self.bias_vectors[i] - learning_rate * dL_db[i]
                                     for i in range(len(self.bias_vectors))]

        return average_train_losses, average_test_losses, average_train_accuracy, average_test_accuracy



