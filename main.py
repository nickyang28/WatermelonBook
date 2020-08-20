# -*- coding: utf-8 -*-
import ml
import numpy as np
import random
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
from ml.models import LinearRegression, LogisticRegression
from sklearn import linear_model
from ml.measurements import *
import torch
from torch import nn, optim

data = np.loadtxt('./data/watermelon_3a.csv', delimiter=',')
X, y = data[:, 1:-1], data[:, -1]
'''
lgr = linear_model.LogisticRegression(solver="newton-cg", max_iter=500)
lgr.fit(X, y)
y_pred = lgr.predict(X)
print(classification_report(y, y_pred))
print(lgr.coef_)
print(lgr.intercept_)
'''

lgr = LogisticRegression(engine='torch', method='SGD', max_iter=10000)
lgr.fit(X, y)
y_pred = lgr.predict(X)
print(classification_report(y, y_pred))
print(lgr.coef_)
print(lgr.intercept_)

# X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
# print(torch.sigmoid(torch.tensor(X @ lgr.beta.T)))

