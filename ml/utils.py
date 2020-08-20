"""
The :mod:`ml.utils` module includes classes and
functions to split the data based on a preset strategy.
"""
# -*- coding: utf-8 -*-
import random
import numpy as np


def train_test_split(x, y, shuffle=True, test_rate=0.2, engine="numpy"):
    """
    X = [[x, 2 * x] for x in range(10)]
    y = [x for x in range(10)]
    train_X, train_y, test_X, test_y = ml.train_test_split(X, y, engine='numpy')
    :param x: an iterable object
    :param y: an iterable object
    :param shuffle: bool
    :param test_rate: default = 0.2
    :param engine: either numpy or python
    :return: train_x, train_y, test_x, test_y
    """
    assert iter(x), "Sample X should be iterable."
    assert iter(y), "Target y should be iterable."
    assert len(x) == len(y), "The length of X does not equal to that of y."
    assert engine in ['numpy', 'python'], r'Engine should be either "numpy" or "python".'

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    idx = list(range(len(x)))
    if shuffle:
        random.shuffle(idx)
    split_pos = int((1 - test_rate) * len(x))
    assert split_pos > 0, "Not enough samples for split."
    train_idx, test_idx = idx[:split_pos], idx[split_pos:]

    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]

    if engine == "python":
        train_x, train_y = train_x.tolist(), train_y.tolist()
        test_x, test_y = test_x.tolist(), test_y.tolist()
    return train_x, train_y, test_x, test_y


class KFold:
    """
    X = [[x, 2 * x] for x in range(100)]
    y = [x for x in range(100)]
    kf = ml.KFold(X, y, True, "numpy")
    for train_x, train_y, test_x, test_y in kf.split(17):
    print(test_x, test_y)
    """
    def __init__(self, x, y, shuffle=True, engine="numpy"):
        assert iter(x), "Sample X should be iterable."
        assert iter(y), "Target y should be iterable."
        assert len(x) == len(y), "The length of X does not equal to that of y."
        assert engine in ['numpy', 'python'], r'Engine should be either "numpy" or "python".'

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self._x, self._y = x, y

        self.idx = list(range(len(x)))
        if shuffle:
            random.shuffle(self.idx)

        self._engine = engine

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def engine(self):
        return self._engine

    def split(self, k=5):
        """
        :param k: number of folds
        :return: return None, but yield train_x, train_y, test_x, test_y
        """
        assert k > 1, "The folds number k should be more than 1."
        fold_size = len(self.x) // k
        if fold_size == 0:
            raise ValueError("Cannot have number of splits k={0} greater"
                             " than the number of samples: n_samples={1}."
                             .format(k, len(self.x)))
        for i in range(k):
            test_idx = self.idx[i * fold_size:(i + 1) * fold_size]
            train_idx = np.array(list(set(self.idx) - set(test_idx)))
            train_x, train_y = self.x[train_idx], self.y[train_idx]
            test_x, test_y = self.x[test_idx], self.y[test_idx]
            if self.engine == 'python':
                train_x, train_y = train_x.tolist(), train_y.tolist()
                test_x, test_y = test_x.tolist(), test_y.tolist()
            yield train_x, train_y, test_x, test_y


def convert2array(arr):
    """
    :param arr: iterable object
    :return: np.ndarray
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr


def normalization(arr):
    arr = convert2array(arr)
    mean = arr.mean()
    std = arr.std()
    arr = (arr - mean) / std
    return arr


def one_hot(arr):
    arr = convert2array(arr)
    one_hot_targets = np.eye(len(set(arr)))[np.array(list(map(int, arr)))]
    return one_hot_targets
