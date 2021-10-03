import pickle
import unittest
import os
import numpy as np

from homework2.feed_forward import *
from homework2.finite_difference import finite_difference, nn_finite_difference_checker


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

    def test_nn_finite_difference_checker(self):
        test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data",
                                                      "test_batch_weights_biases.pkl"))
        with open(test_data_path, "rb") as fn:
            (X_batch, y_batch, weight_matrices, biases) = pickle.load(fn)

        print(X_batch.shape)
        print(y_batch.shape)
        print(len(weight_matrices))
        for l in weight_matrices:
            print(l.shape)

        activations = ["relu", "sigmoid"]
        nn_finite_difference_checker(X_batch, y_batch, weight_matrices, biases, activations)