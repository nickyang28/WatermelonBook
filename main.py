# -*- coding: utf-8 -*-
import ml
import numpy as np
import random
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
from ml.models import LinearRegression, LogisticRegression, LinearDiscriminantAnalysis
from sklearn import linear_model
from ml.measurements import *
import torch
from torch import nn, optim

data = np.loadtxt('./data/watermelon_3a.csv', delimiter=',')
X, y = data[:, 1:-1], data[:, -1]
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

