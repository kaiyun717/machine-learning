"""
UC Berkeley CS 189 - Homework 2
Part 1.2(a)
Finite Difference Checker

Author: Kai Yun (kaiyun799@berkeley.edu)
"""

import numpy as np


def partial_derivative_approx(func, a, index, eps=1e-5):
    a_1 = np.copy(a)
    a_2 = np.copy(a)
    a_1[index] = a[index] + eps
    a_2[index] = a[index] - eps
    a_1_out, _ = func(a_1)
    a_2_out, _ = func(a_2)

    partial_approx = (a_1_out - a_2_out) / (2*eps)

    return partial_approx[index]


# Should implement this with Numpy and get rid of the `for-loop`.
def finite_difference(func, a, eps=1e-5):
    a_length = a.shape[0]       # `a` is a vector
    a_diff = np.zeros(a_length)

    for i in range(a_length):
        a_diff[i] = partial_derivative_approx(func=func, a=a, index=i, eps=eps)

    return a_diff
