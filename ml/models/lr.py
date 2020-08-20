# -*- coding: utf-8 -*-
import numpy as np
from ..utils import *
from scipy import linalg


class LinearRegression:

    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.coef_ = self._residues = self.rank_ = self.singular_ = self.intercept_ = 0

    def fit(self, X, y):
        X = convert2array(X)
        y = convert2array(y)
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        if self.normalize:
            X = normalization(X)
            y = normalization(y)
        self.coef_, self._residues, self.rank_, self.singular_ = linalg.lstsq(X, y)
        self.coef_ = self.coef_.T
        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        return self

    def predict(self, X):
        X = convert2array(X)
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_hat = X @ self.coef_
        return y_hat

    def score(self, X, y):
        X = convert2array(X)
        y = convert2array(y)
        y_hat = self.predict(X)
        return ((y_hat - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
