"""
UC Berkeley CS 189 - Homework 2
Original Name: `hw2.py`
"""

import numpy as np


def display_image(image):
    """
    Displays an image from the mnist dataset

    Make sure you have the matplotlib library installed

    If using Jupyter, you may need to add %matplotlib inline to the top
    of your notebook
    """
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="gray")
    plt.show()


def get_mnist_threes_nines():
    """
    Creates MNIST train / test datasets
    """
    import mnist
    Y0 = 3
    Y1 = 9

    y_train = mnist.train_labels()
    y_test = mnist.test_labels()
    X_train = (mnist.train_images()/255.0)
    X_test = (mnist.test_images()/255.0)
    train_idxs = np.logical_or(y_train == Y0, y_train == Y1)
    test_idxs = np.logical_or(y_test == Y0, y_test == Y1)
    y_train = y_train[train_idxs].astype('int')
    y_test = y_test[test_idxs].astype('int')
    X_train = X_train[train_idxs]
    X_test = X_test[test_idxs]
    y_train = (y_train == Y1).astype('int')
    y_test = (y_test == Y1).astype('int')
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()

    import mnist
    images = mnist.train_images()
    # image = images[1]
    image = X_train[0]
    display_image(image=image)

    print(y_train)
    print(X_train.shape)
    print(y_train.shape)
    # 1 is 9, 0 is 3
