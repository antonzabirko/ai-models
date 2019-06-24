import numpy as np
import h5py


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def relu_backwards(dA, cache):
    Z = cache
    r = relu(Z)
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ


def sigmoid(Z):
    """
    Sigmoid function

    :param Z: Numpy array of any shape
    :return A: the activity
    :return cache: saved Z for backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backwards(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)

    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
