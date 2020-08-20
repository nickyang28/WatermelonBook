"""
The :mod:`ml.measurements` module includes classes and
functions to measure the performance of a model.
"""
# -*- coding: utf-8 -*-
import numpy as np
from .utils import *


def mse(y_true, y_pred):
    """
    Calculate the mean squared error.
    :return: ((y_true - y_pre) ** 2).mean()
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)
    return ((y_true - y_pred) ** 2).mean()


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy.
    :return: (y_true == y_pre).mean()
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)
    return (y_true == y_pred).mean()


def error(y_true, y_pred):
    """
    Calculate the error rate.
    :return: 1 - accuracy
    """
    return 1 - accuracy(y_true, y_pred)


def multilabel_confusion_matrix(y_true, y_pred, labels=None):
    """
    One-vs-rest confusion matrix.
    :param y_true: Pass
    :param y_pred: Pass
    :param labels: Pass
    :return: np.ndarray confusion matrix
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)

    if not labels:
        labels = sorted(list(set(y_true)))
    matrix = np.zeros((len(labels), 2, 2))
    for i, label in enumerate(labels):
        tp = (y_true[y_pred == label] == label).sum()
        fp = (y_true[y_pred == label] != label).sum()
        tn = (y_true[y_pred != label] != label).sum()
        fn = (y_true[y_pred != label] == label).sum()
        matrix[i, 0, 0] = tp
        matrix[i, 1, 1] = tn
        matrix[i, 0, 1] = fn
        matrix[i, 1, 0] = fp
    return matrix


def confusion_matrix(y_true, y_pred, labels=None):
    """
    From wiki: https://en.wikipedia.org/wiki/Confusion_matrix
            Predicted
    Actual
            pos     neg
      pos   TP      FN
      neg   FP      TN
    :param y_true: 1-d iterable object
    :param y_pred: 1-d iterable object
    :param labels: Pass
    :return: np.ndarray confusion matrix
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)

    if not labels:
        labels = sorted(list(set(y_true)))
    matrix = np.zeros((len(labels), len(labels)))
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            matrix[i, j] = (y_true[y_pred == label_i] == label_j).sum()
    return matrix


def recall_score(y_true, y_pred, average='samples', labels=None, zero_division=0):
    """
    :param zero_division: [0, 1], default: 0
    :param labels: iterable object
    :param y_true: 1-d iterable object.
    :param y_pred: 1-d iterable object.
    :param average: string, [None, ‘micro’, ‘macro’, ‘samples’ (default)]
    :return: recall score.
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)
    if not labels:
        labels = sorted(list(set(y_true)))

    if average == "samples" or average is None:
        matrix = multilabel_confusion_matrix(y_true, y_pred)
        scores = dict()
        for i, label in enumerate(labels):
            tp = int(matrix[i, 0, 0])
            fn = int(matrix[i, 0, 1])
            try:
                scores[label] = tp / (tp + fn)
            except ZeroDivisionError:
                scores[label] = zero_division
        return scores

    if average == "macro":
        return sum(recall_score(y_true, y_pred, labels=labels
                                , zero_division=zero_division).values()) / len(labels)

    if average == "micro":
        matrix = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tp = fn = 0
        for i, label in enumerate(labels):
            tp += int(matrix[i, 0, 0])
            fn += int(matrix[i, 0, 1])
        avg_tp = tp / len(labels)
        avg_fn = fn / len(labels)
        try:
            return avg_tp / (avg_tp + avg_fn)
        except ZeroDivisionError:
            return zero_division


def precision_score(y_true, y_pred, average='samples', labels=None, zero_division=0):
    """
    :param zero_division: [0, 1], default: 0
    :param labels: iterable object
    :param y_true: 1-d iterable object.
    :param y_pred: 1-d iterable object.
    :param average: string, [None, ‘micro’, ‘macro’, ‘samples’ (default)]
    :return: precision score.
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)
    if not labels:
        labels = sorted(list(set(y_true)))

    if average == "samples" or average is None:
        matrix = multilabel_confusion_matrix(y_true, y_pred)
        scores = dict()
        for i, label in enumerate(labels):
            tp = int(matrix[i, 0, 0])
            fp = int(matrix[i, 1, 0])
            try:
                scores[label] = tp / (tp + fp)
            except ZeroDivisionError:
                scores[label] = zero_division
        return scores

    if average == "macro":
        return sum(precision_score(y_true, y_pred, labels=labels
                                   , zero_division=zero_division).values()) / len(labels)

    if average == "micro":
        matrix = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tp = fp = 0
        for i, label in enumerate(labels):
            tp += int(matrix[i, 0, 0])
            fp += int(matrix[i, 1, 0])
        avg_tp = tp / len(labels)
        avg_fp = fp / len(labels)
        try:
            return avg_tp / (avg_tp + avg_fp)
        except ZeroDivisionError:
            return zero_division


def f1_score(y_true, y_pred, average='samples', labels=None, zero_division=0):
    """
    :param zero_division: [0, 1], default: 0
    :param labels: iterable object
    :param y_true: 1-d iterable object.
    :param y_pred: 1-d iterable object.
    :param average: string, [None, ‘micro’, ‘macro’, ‘samples’ (default)]
    :return: f1 score.
    """
    y_true = convert2array(y_true)
    y_pred = convert2array(y_pred)
    if not labels:
        labels = sorted(list(set(y_true)))

    if average == "samples" or average is None:
        scores = dict()
        p_dict = precision_score(y_true, y_pred, labels=labels, zero_division=zero_division)
        r_dict = recall_score(y_true, y_pred, labels=labels, zero_division=zero_division)
        for label in labels:
            try:
                scores[label] = (2 * p_dict[label] * r_dict[label]) / (p_dict[label] + r_dict[label])
            except ZeroDivisionError:
                scores[label] = zero_division
        return scores

    if average == "macro":
        return sum(f1_score(y_true, y_pred, labels=labels,
                            zero_division=zero_division).values()) / len(labels)

    if average == "micro":
        micro_p = precision_score(y_true, y_pred, labels=labels,
                                  average="micro", zero_division=zero_division)
        micro_r = recall_score(y_true, y_pred, labels=labels,
                               average="micro", zero_division=zero_division)
        try:
            return (2 * micro_p * micro_r) / (micro_p + micro_r)
        except ZeroDivisionError:
            return zero_division
