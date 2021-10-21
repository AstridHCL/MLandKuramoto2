# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:15:08 2021

@author: AstCor
"""

import os
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# https://scikit-learn.org/0.15/auto_examples

classifiers = {
        "KNN": KNeighborsClassifier(20),
        "L_SVM": SVC(kernel="linear", C=0.025, probability=True),
        "RBF_SVM": SVC(gamma=1, C=1, probability=True),
        "GP": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "DT": DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 30, min_samples_split= 10, splitter= 'random'),
        "RF": RandomForestClassifier(n_estimators=100),
        "NN_1l": MLPClassifier(early_stopping=True, hidden_layer_sizes=100,learning_rate_init=0.1),
        "NN_ml": MLPClassifier(early_stopping=True, hidden_layer_sizes=(25,50,15),learning_rate_init=0.1),
        "AB": AdaBoostClassifier(n_estimators=100),
        "NB": GaussianNB(),
        "LR": LogisticRegression(),
        "LDA": LDA(),
        "QDA": QDA(),
    }
