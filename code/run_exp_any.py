import cvxpy as cp
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

from LR2NB import LR2NB
import pickle

import utils
sys.setrecursionlimit(20000)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="if its in data/x, put x")
parser.add_argument("--label", help="x = the label between train-x-labels.txt")
args = parser.parse_args()

print ("Choosing dataset: {}".format(args.folder))
print ("   label        : {}".format(args.label))

# Data Loading

X_train, y_train, X_test, y_test = utils.load_dataset(args.folder, args.label)

# Train LR
clf = LogisticRegression(solver='lbfgs', max_iter = 1000).fit(X_train, y_train)
W = np.hstack((clf.intercept_[:,None], clf.coef_)).flatten()
print("Logistic Regression Accuracy: {}".format(np.average(clf.predict(X_test) == y_test)))
# Train NB Directly
NB = BernoulliNB().fit(X_train, y_train)
print("Naive Bayes Accuracy: {}".format(np.average(NB.predict(X_test) == y_test)))
# Train LR->NB
a = LR2NB(W)
a.setObj(X_train, y_train, divider = 100, c = 1)
a.solve()
print("Relaxed bounds are tight (eps = {}): {}".format(1e-6, a.__validate_relaxed_sums__(1e-6)))
print("LR ->NB Accuracy: {}".format(np.average(a.classify(X_test) == y_test)))


utils.save("trained_lr2nb_{}.pickle".format(args.folder), (a, clf, NB) )

print("Solution Value: {}".format(a.problem.value) )




