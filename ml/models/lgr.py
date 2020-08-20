# -*- coding: utf-8 -*-
from abc import ABC

import numpy as np
from ..utils import *
import torch
from torch import optim, nn
import random


class LogisticRegression:
    """
    X = np.array([[i, j] for i in range(10) for j in reversed(range(10))])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LogisticRegression().fit(X, y)
    """
    def __init__(self, engine: str = "python", method: str = 'newton',
                 device: str = 'cpu', lr: float = 0.5, max_iter: int = 500):
        """
        Logistic Regression Mode: Support multiple engine.
        :param engine: Select from [None, 'python' (default), 'torch']
        :param method: Select from [None, 'newton' (default), 'SGD']
        :param device: Select from [None, 'cpu' (default), 'cuda']
        :param lr: learning rate
        """
        self.max_iter = max_iter
        self.lr = lr
        self.device = device
        self.engine = engine
        self.method = method
        self.coef_ = self.intercept_ = 0
        self.beta = np.array([0])

    def fit(self, X: list or np.ndarray, y: list or np.ndarray):
        X = convert2array(X)
        y = convert2array(y)
        if self.engine == 'python':
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        if self.method == 'newton':
            self.beta = self._newton(X, y)
        else:
            self.beta = self._sgd(X, y)
        self.intercept_ = self.beta[:, -1]
        self.coef_ = self.beta[:, :-1]
        return self

    def _newton(self, X, y):
        if self.engine == 'python':
            beta = np.array([random.normalvariate(0, 1) for _ in range(X.shape[1])]).reshape((1, -1))
            for _ in range(self.max_iter):
                beta -= self._likelihood_1d(X, y, beta) @  np.linalg.inv(self._likelihood_2d(X, beta))
        else:
            raise ValueError('Does not support Newton with pytorch now.')
        return beta

    def _sgd(self, X, y):
        if self.engine == 'python':
            beta = np.array([random.normalvariate(0, 1) for _ in range(X.shape[1])]).reshape((1, -1))
            for _ in range(self.max_iter):
                beta -= self.lr * self._likelihood_1d(X, y, beta)
        else:
            X = torch.tensor(X, dtype=torch.float).to(self.device)
            y = torch.tensor(y, dtype=torch.float).to(self.device)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = LogisticNet(X.shape[1]).to(self.device)
            loss = nn.BCELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.lr)

            for _ in range(self.max_iter):
                y_hat = net(X)
                criterion = loss(y_hat, y.view(y_hat.shape)).sum()
                optimizer.zero_grad()
                criterion.backward()
                optimizer.step()
            beta = np.concatenate(list(param.detach().numpy().reshape(1, -1) for param in net.parameters()), axis=1)
        return beta

    def _likelihood_1d(self, X, y, beta):
        p1 = np.array(list(map(self._p1, X @ beta.T)))
        return - (X * (y - p1).reshape(-1, 1)).sum(axis=0)

    def _likelihood_2d(self, X, beta):
        s = []
        for i in range(X.shape[0]):
            xi = X[i].reshape((1, -1))
            p1 = self._p1(xi @ beta.T)
            s.append(xi.T @ xi * p1 * (1 - p1))
        s = np.array(s)
        return s.sum(axis=0)

    def predict(self, X):
        X = convert2array(X)
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        p1 = np.array(list(map(self._p1, X @ self.beta.T)))
        p0 = np.array(list(map(self._p0, X @ self.beta.T)))
        return np.array(list(map(float, p1 > p0)))

    def score(self, X, y):
        X = convert2array(X)
        y = convert2array(y)
        y_hat = self.predict(X)
        return ((y_hat - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    @staticmethod
    def _p1(x):
        return np.math.exp(x) / (1 + np.math.exp(x))

    @staticmethod
    def _p0(x):
        return 1 / (1 + np.math.exp(x))


class LogisticNet(nn.Module, ABC):

    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


