import unittest

import numpy as np

from homework2.feed_forward import *
from homework2.finite_difference import finite_difference


class TestFiniteDifference(unittest.TestCase):
    @staticmethod
    def dummy_func(x):
        y = 0.1 * x ** 5 - 0.2 * x ** 3 + 1
        return y, None

    @staticmethod
    def dummy_func_dev(x):
        y_dev = 0.5 * x ** 4 - 0.6 * x ** 2
        return y_dev

    def test_finite_difference(self, test_array=np.array([5, 3, 1, 7, 87, 10], dtype=np.float64).T):
        all_close = np.allclose(self.dummy_func_dev(test_array), finite_difference(func=self.dummy_func, a=test_array))
        self.assertEqual(True, all_close)


class TestFeedForwardFunctions(unittest.TestCase):

    def test_sigmoid_activation(self):

        pass

    def test_logistic_loss(self):
        pass

    def test_relu_activation(self):
        pass

    def test_layer_forward(self):
        pass
