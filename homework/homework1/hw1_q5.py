"""
UC Berkeley CS 189 Fall 2021
Homework 1, Question 5
Isocontours of Normal Distributions

Author: Kai Yun

* Citations:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class PlotIsocontour:
    def __init__(self):
        self.fig = plt.figure()

    def plot_isocontour(self, pos, name, x, y, mu, sigma):
        """
        Plots the isocontour of a probability density function (pdf).
        :param pos: position of the subplot
        :param name: name of the subplot
        :param x: x-axis
        :param y: y-axis
        :param mu: mean vector
        :param sigma: covariance matrix
        :return: None
        """
        position = np.dstack((x, y))

        random_variable = multivariate_normal(mean=mu, cov=sigma)

        sub_plot = self.fig.add_subplot(pos)
        sub_plot.contourf(x, y, random_variable.pdf(position))
        sub_plot.set_aspect('equal')
        sub_plot.title.set_text(name)

    def plot_diff_isocontour(self, pos, name, x, y, mu_1, mu_2, sigma_1, sigma_2):
        """
        Plots the isocontour of the difference between two pdf's.
        :param pos: position of the subplot
        :param name: name of the subplot
        :param x: x-axis
        :param y: y-axis
        :param mu_1: mean vector of first pdf
        :param mu_2: mean vector of second pdf
        :param sigma_1: covariance matrix of first pdf
        :param sigma_2: covariance matrix of second pdf
        :return: None
        """
        position = np.dstack((x, y))

        random_variable_1 = multivariate_normal(mean=mu_1, cov=sigma_1)
        random_variable_2 = multivariate_normal(mean=mu_2, cov=sigma_2)

        rv_1_pdf = random_variable_1.pdf(position)
        rv_2_pdf = random_variable_2.pdf(position)
        diff_pdf = rv_1_pdf - rv_2_pdf

        sub_plot = self.fig.add_subplot(pos)
        sub_plot.contourf(x, y, diff_pdf)
        sub_plot.set_aspect('equal')
        sub_plot.title.set_text(name)

    def show_isocontours(self):
        """
        Shows all the isocontours.
        :return: None
        """
        self.fig.subplots_adjust(left=0.1,
                                 bottom=0.1,
                                 right=0.9,
                                 top=0.9,
                                 wspace=0.4,
                                 hspace=0.4)
        self.fig.show()


if __name__ == "__main__":
    isocontours = PlotIsocontour()

    # Part (a)
    mu_a = np.array([1, 1]).T
    sigma_a = np.array([[1, 0], [0, 2]])
    x_a, y_a = np.mgrid[-5:5:0.01, -5:5:0.01]

    isocontours.plot_isocontour(pos=231, name="Part(a)", x=x_a, y=y_a, mu=mu_a, sigma=sigma_a)

    # Part (b)
    mu_b = np.array([-1, 2]).T
    sigma_b = np.array([[2, 1], [1, 4]])
    x_b, y_b = np.mgrid[-5:5:0.01, -3:7:0.01]

    isocontours.plot_isocontour(pos=232, name="Part(b)", x=x_b, y=y_b, mu=mu_b, sigma=sigma_b)

    # Part (c)
    x_c, y_c = np.mgrid[-5:5:0.01, -5:5:0.01]
    mu_c_1 = np.array([0, 2]).T
    mu_c_2 = np.array([2, 0]).T
    sigma_c_1 = np.array([[2, 1], [1, 1]])
    sigma_c_2 = sigma_c_1

    isocontours.plot_diff_isocontour(pos=233, name="Part(c)", x=x_c, y=y_c, mu_1=mu_c_1, mu_2=mu_c_2, sigma_1=sigma_c_1, sigma_2=sigma_c_2)

    # Part (d)
    x_d, y_d = np.mgrid[-5:5:0.01, -5:5:0.01]
    mu_d_1 = np.array([0, 2]).T
    mu_d_2 = np.array([2, 0]).T
    sigma_d_1 = np.array([[2, 1], [1, 1]])
    sigma_d_2 = np.array([[2, 1], [1, 4]])

    isocontours.plot_diff_isocontour(pos=234, name="Part(d)", x=x_d, y=y_d, mu_1=mu_d_1, mu_2=mu_d_2, sigma_1=sigma_d_1, sigma_2=sigma_d_2)

    # Part (e)
    x_e, y_e = np.mgrid[-5:5:0.01, -5:5:0.01]
    mu_e_1 = np.array([1, 1]).T
    mu_e_2 = np.array([-1, -1]).T
    sigma_e_1 = np.array([[2, 0], [0, 1]])
    sigma_e_2 = np.array([[2, 1], [1, 2]])

    isocontours.plot_diff_isocontour(pos=235, name="Part(e)", x=x_e, y=y_e, mu_1=mu_e_1, mu_2=mu_e_2, sigma_1=sigma_e_1, sigma_2=sigma_e_2)

    isocontours.show_isocontours()
