import os
import sys

import numpy as np
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
        w_l2, y_train_estimate = ridge_regression(X=train_fold_X,
                                                  y=train_fold_y,
                                                  degree=self.degree,
                                                  penalty=self.penalty)
        test_feat_mat = assemble_feature(x=test_fold_X, D=self.degree)
        y_test_estimate = np.dot(test_feat_mat, w_l2)

        train_mse = (np.square(train_fold_y - y_train_estimate)).mean()
        test_mse = (np.square(test_fold_y - y_test_estimate)).mean()
        return train_mse, test_mse

    def k_folds_cross_mse_avg(self, training_X, training_y):
        """
        First, split the entirety of training data X and training data y into k-folds.
        Then, calculate the mean-squared-error for each split.
        Finally, find the average of those mean-squared-errors.
        :param training_X
        :param training_y
        :return: average of mean-squared-training/validation-error of each split
        """
        splits_X = np.split(training_X, self.k, axis=0)
        splits_y = np.split(training_y, self.k, axis=0)

        cross_train_mse_sum = 0
        cross_test_mse_sum = 0

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

            train_mse, test_mse = self.mean_squared_error(train_fold_X, train_fold_y, test_fold_X, test_fold_y)
            cross_train_mse_sum += train_mse
            cross_test_mse_sum += test_mse

        train_mse_avg = cross_train_mse_sum / self.k
        test_mse_avg = cross_test_mse_sum / self.k
        return train_mse_avg, test_mse_avg

    def k_fold_cross_mse_avg(self):
        train_mse_avg, test_mse_avg = self.k_folds_cross_mse_avg(training_X=self.X, training_y=self.y)
        return train_mse_avg, test_mse_avg


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
        _, k_fold_cross_mse_avg = k_fold.k_fold_cross_mse_avg()
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
    D = range(1, 6)
    penalty = [0.05, 0.1, 0.15, 0.2]
    for degree in D:
        for p in penalty:
            k_fold = KFoldCrossValidation(k=k, X=X_data, y=y_data, degree=degree, penalty=p)
            k_fold_train_mse_avg, k_fold_test_mse_avg = k_fold.k_fold_cross_mse_avg()
            if min_cross_mse_avg > k_fold_test_mse_avg:
                min_cross_mse_avg = k_fold_test_mse_avg
                best_degree = degree
                best_penalty = p

            print(f"Degree {degree} & Penalty {p} \n"
                  f"==> Train: {k_fold_train_mse_avg}\tValidation: {k_fold_test_mse_avg}")

    print(f"\n(g) Best Degree: {best_degree}")
    print(f"(g) Best Penalty: {best_penalty}")
    print(f"(g) Mean Squared Error: {min_cross_mse_avg}")
