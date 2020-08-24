# -*- coding: utf-8 -*-
import numpy as np
from ..utils import *


class LinearDiscriminantAnalysis:

    def __init__(self):
        pass

    def fit(self, X, y):
        # X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        X0 = X[y == 0]
        X1 = X[y == 1]
        mu0 = X0.mean(axis=0).reshape((1, -1))
        mu1 = X1.mean(axis=0).reshape((1, -1))
        Sb = (mu0 - mu1).T @ (mu0 - mu1)
        sigma0 = 0
        for i in range(len(X0)):
            x = X0[i].reshape(1, -1)
            sigma0 += (x - mu0).T @ (x - mu0)

        sigma1 = 0
        for i in range(len(X1)):
            x = X1[i].reshape(1, -1)
            sigma1 += (x - mu1).T @ (x - mu1)

        Sw = np.mat(sigma0 + sigma1)
        u, s, vt = np.linalg.svd(Sw)
        v = vt.T

        Sw_1 = v @ np.linalg.inv(np.diag(s)) @ u.T
        w = (Sw_1 @ (mu0 - mu1).T).T
        return self
