import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from hw2_data_code.hw1_prob5 import assemble_feature


def ridge_regression(X, y, degree, penalty=0.1):
    feat_mat = assemble_feature(x=X, D=degree)      # Feature Matrix
    feat_mat_size = feat_mat.T.dot(feat_mat).shape[0]
    identity = np.identity(feat_mat_size)

    w_l2 = np.linalg.inv(feat_mat.T.dot(feat_mat) + penalty * identity).dot(feat_mat.T).dot(y)
    y_estimate = np.dot(feat_mat, w_l2)

    return w_l2, y_estimate


class KFoldCrossValidation:
    def __init__(self, k, X, y, degree, penalty=0.1):
        """
        K-Fold Cross-Validation
        :param k: k-value
        :param X: entirety of feature values
        :param y: entirety of y values
        :param degree: degree of polynomial f_D(X)
        """
        self.k = k
        self.X = X
        self.y = y
        self.degree = degree
        self.penalty = penalty
        self.n = X.shape[0]         # number of training sets

    # def test_and_training_data(self):
    #     test_frac = self.n / self.k
    #
    #     test_X = self.X[-test_frac:, :]               # the last `test_frac` number of rows
    #     test_y = self.y[-test_frac:, :]               # the last `test_frac` number of rows
    #
    #     training_X = self.X[:self.n - test_frac, :]   # the first `test_frac` number of rows
    #     training_y = self.y[:self.n - test_frac, :]   # the first `test_frac` number of rows
    #
    #     return test_X, test_y, training_X, training_y

    def mean_squared_error(self, train_fold_X, train_fold_y, test_fold_X, test_fold_y):
        """
        First compute the polynomial coefficients with `ridge_regression'.
        Then, compute the mean-squared-error of the estimated y with `test_fold_y`.
        :param train_fold_X: training data X for current split
        :param train_fold_y: training data y for current split
        :param test_fold_X: testing data X for current split
        :param test_fold_y: testing data y for current split
        :return: Mean-squared-error between estimated y for `test_fold_X` and `test_fold_y`.
        """
        w_l2, _ = ridge_regression(X=train_fold_X, y=train_fold_y, degree=self.degree, penalty=self.penalty)
        test_feat_mat = assemble_feature(x=test_fold_X, D=self.degree)
        y_estimate = np.dot(test_feat_mat, w_l2)

        mse = (np.square(test_fold_y - y_estimate)).mean()
        return mse

    def k_folds_cross_mse_avg(self, training_X, training_y):
        """
        First, split the entirety of training data X and training data y into k-folds.
        Then, calculate the mean-squared-error for each split.
        Finally, find the average of those mean-squared-errors.
        :param training_X
        :param training_y
        :return: average of mean-squared-error of each split
        """
        splits_X = np.split(training_X, self.k, axis=0)
        splits_y = np.split(training_y, self.k, axis=0)

        cross_mse_sum = 0

        # For each split
        for i in range(self.k):
            train_fold_X = []
            train_fold_y = []
            test_fold_X = None
            test_fold_y = None

            # For each fold, make training and testing sets
            for fold in range(self.k):
                if fold != i:
                    train_fold_X.append(splits_X[fold])
                    train_fold_y.append(splits_y[fold])
                else:
                    test_fold_X = splits_X[fold]
                    test_fold_y = splits_y[fold]

            train_fold_X = np.vstack(tuple(train_fold_X))
            train_fold_y = np.vstack(tuple(train_fold_y))

            mean_squared_error = self.mean_squared_error(train_fold_X, train_fold_y, test_fold_X, test_fold_y)
            cross_mse_sum += mean_squared_error

        cross_mse_avg = cross_mse_sum / self.k
        return cross_mse_avg

    def k_fold_cross_mse_avg(self):
        # test_X, test_y, training_X, training_y = self.test_and_training_data()
        cross_mse_avg = self.k_folds_cross_mse_avg(training_X=self.X, training_y=self.y)
        return cross_mse_avg


if __name__ == "__main__":
    samples_dir = os.path.join("hw2_data_code", "polynomial_regression_samples.mat")
    samples = loadmat(samples_dir)

    y_data = samples["y"]
    X_data = samples["x"]

    k = 4

    # == Part (f) ==
    mse_list = []
    min_cross_mse_avg = sys.float_info.max
    best_degree = 0
    D = range(6)
    for degree in D:
        k_fold = KFoldCrossValidation(k=k, X=X_data, y=y_data, degree=degree)
        k_fold_cross_mse_avg = k_fold.k_fold_cross_mse_avg()
        mse_list.append(k_fold_cross_mse_avg)
        if min_cross_mse_avg > k_fold_cross_mse_avg:
            min_cross_mse_avg = k_fold_cross_mse_avg
            best_degree = degree

    print(f"(f) Best Degree: {best_degree}")
    print(f"(f) Mean Squared Error: {min_cross_mse_avg}\n\n")

    # == Part (g) ==
    min_cross_mse_avg = sys.float_info.max
    best_degree = 0
    best_penalty = 0
    D = range(6)
    penalty = [0.05, 0.1, 0.15, 0.2]
    for degree in D:
        for p in penalty:
            k_fold = KFoldCrossValidation(k=k, X=X_data, y=y_data, degree=degree, penalty=p)
            k_fold_cross_mse_avg = k_fold.k_fold_cross_mse_avg()
            if min_cross_mse_avg > k_fold_cross_mse_avg:
                min_cross_mse_avg = k_fold_cross_mse_avg
                best_degree = degree
                best_penalty = p

            print(f"Degree {degree} & Penalty {p}: {k_fold_cross_mse_avg}")

    print(f"(g) Best Degree: {best_degree}")
    print(f"(g) Best Penalty: {best_penalty}")
    print(f"(f) Mean Squared Error: {min_cross_mse_avg}")
