import cvxpy as cp
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
from sklearn.naive_bayes import BernoulliNB
import keras

from LR2NBK import LR2NBK
import pickle

import utils
sys.setrecursionlimit(20000000)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--remove", help="remove #")

args = parser.parse_args()

print ("remove: {}".format(args.remove))

# Data Loading
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

Z  = X_train < 128
nZ = X_train >= 128

tZ  = X_test < 128
tnZ = X_test >= 128

X_train[Z] = 0
X_train[nZ] = 1
X_test[tZ] = 0
X_test[tnZ] = 1

X_train_raw = X_train.reshape(-1, 28*28)
X_test_raw  = X_test.reshape(-1, 28*28)

remove = int(args.remove)  ##### Change this for less or more remving features
choose = np.array([ 1 if (28*remove-1<i<28*(28-remove)) and (remove <= i%28 < 28-remove) else 0 for i in range(X_train_raw.shape[1]) ])
inds = np.array(np.where(choose ==1)).flatten()
X_train = X_train_raw[:, inds]
X_test  = X_test_raw[:, inds]

# Train LR
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, verbose = True, n_jobs=6).fit(X_train, y_train)
W = np.hstack((clf.intercept_[:,None], clf.coef_))
print(W.shape)
print("Logistic Regression Accuracy: {}".format(np.average(clf.predict(X_test) == y_test)))
# Train NB Directly
NB = BernoulliNB().fit(X_train, y_train)
print("Naive Bayes Accuracy: {}".format(np.average(NB.predict(X_test) == y_test)))

# Train LR->NB
a = LR2NBK(W)
a.setObj(X_train, y_train, divider = 100, c = 1)
a.solve()


utils.save("trained_lr2nn_k_{}mnist.pickle".format(X_train.shape[1]), (clf, NB,a))
print("Solution Value: {}".format(a.problem.value) )




