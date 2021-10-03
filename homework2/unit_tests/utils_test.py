import pickle
import unittest
import os
import numpy as np

from homework2.data.get_mnist_data import get_mnist_threes_nines
from homework2.feed_forward import *
from homework2.finite_difference import finite_difference, nn_finite_difference_checker
from homework2.neural_network import forward_pass, loss_fn, backward_pass, FullyConnectedMLP


class NNFiniteDifferenceChecker(unittest.TestCase):
    @staticmethod
    def dummy_func(x):
        y = 0.1 * x[0] ** 5 - 0.2 * x[1] ** 3 + 13 * x[2] + 6
        return y, None

    @staticmethod
    def dummy_func_dev(x):
        y_dev_x0 = 0.5 * x[0] ** 4
        y_dev_x1 = -0.6 * x[1] ** 2
        y_dev_x2 = 13
        return np.array([y_dev_x0, y_dev_x1, y_dev_x2])

    def test_finite_difference(self, test_array=np.array([5, 3, 1], dtype=np.float64).T):
        all_close = np.allclose(self.dummy_func_dev(test_array), finite_difference(func=self.dummy_func, a=test_array))
        self.assertEqual(True, all_close)

    def test_nn_finite_difference_checker_pickle(self):
        test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data",
                                                      "test_batch_weights_biases.pkl"))
        with open(test_data_path, "rb") as fn:
            (X_batch, y_batch, weight_matrices, biases) = pickle.load(fn)

        print(f"X_batch:\n{X_batch}\n"
              f"y_batch:\n{y_batch}\n"
              f"Weights:\n{weight_matrices}\n"
              f"biases:\n{biases}\n")

        activations = ["relu", "sigmoid"]
        weight_finite, biases_finite = \
            nn_finite_difference_checker(X_batch, y_batch, weight_matrices, biases, activations)
        print(f"Weight Finite Difference:\n\t{weight_finite}\n"
              f"Biases Finite Difference:\n\t{biases_finite}\n")

        test_forward_out, layer_caches = forward_pass(X_batch, weight_matrices, biases, activations)
        y_estimate = np.reshape(test_forward_out, (y_batch.shape[0]))
        loss, dl_dg = loss_fn(y_estimate, y_batch)

        pickle_dL_dw, pickle_dL_db = backward_pass(dl_dg, layer_caches)
        print(f"dL_dW:\n\t{pickle_dL_dw}\n"
              f"dL_db:\n\t{pickle_dL_db}\n")

        for i in range(len(weight_finite)):
            all_close_weight = np.allclose(weight_finite[i], pickle_dL_dw[i])
            all_close_biases = np.allclose(biases_finite[i], pickle_dL_db[i])
            self.assertEqual(True, all_close_weight)
            self.assertEqual(True, all_close_biases)

    def test_nn_finite_difference_checker_mnist(self):
        # === Retrieve Data ===
        (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()
        num_train_data = X_train.shape[0]
        X_train_each_size = X_train.shape[1] * X_train.shape[2]
        X_train_flat = np.reshape(X_train, (num_train_data, X_train_each_size))

        # === Neural Network settings ===
        layer_dims: list = [X_train_each_size, 200, 1]
        activations: list = ["relu", "sigmoid"]

        # ===========================================
        # === Neural Network forward and backward ===
        # ===========================================
        fc_mlp = FullyConnectedMLP(layer_dims=layer_dims, activations=activations)

        output, layer_caches = fc_mlp.nn_forward_pass(X_batch=X_train_flat)

        y_estimate = np.reshape(output, (y_train.shape[0]))
        loss, dl_dg = fc_mlp.nn_loss_fn(y_estimate=y_estimate, y_batch=y_train)

        dL_dw, dL_db = fc_mlp.nn_backward_pass(dl_dg, layer_caches)

        print(len(dL_dw))
        print(dL_dw[0].shape, dL_dw[1].shape)
        print(len(dL_db))
        print(dL_db[0].shape, dL_db[1].shape)

        # ================================
        # === Finite Difference Values ===
        # ================================
        # weight_matrices = fc_mlp.return_weight_matrices()
        # bias_vectors = fc_mlp.return_bias_vectors()
        # weight_finite, biases_finite = \
        #     nn_finite_difference_checker(X_train_flat, y_train, weight_matrices, bias_vectors, activations)
        #
        # for i in range(len(weight_finite)):
        #     all_close_weight = np.allclose(weight_finite[i], dL_dw[i])
        #     all_close_biases = np.allclose(biases_finite[i], dL_db[i])
        #     self.assertEqual(True, all_close_weight)
        #     self.assertEqual(True, all_close_biases)
