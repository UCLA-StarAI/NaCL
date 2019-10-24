import math
from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import utils
import keras
import LR2NBK_GP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--remove", help="remove #")

sys.setrecursionlimit(2000000)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = np.copy(X_train)
X_test  = np.copy(X_test)

### Binarizing
train_mean = np.mean(X_train, axis=0)
train_std  = np.std(X_train, axis=0)
trainZ = X_train <= train_mean + 0.05 * train_std

test_mean = np.mean(X_test, axis=0)
test_std  = np.std(X_test, axis=0)
testZ = X_test <= test_mean + 0.05 * test_std

nZ  = np.logical_not(trainZ)
tnZ = np.logical_not(testZ)

X_train[trainZ] = 0
X_train[nZ] = 1
X_test[testZ] = 0
X_test[tnZ] = 1

##
X_train_raw = X_train.reshape(-1, 28*28)
X_test_raw  = X_test.reshape(-1, 28*28)

##
args = parser.parse_args()
print ("Remove: {}".format(args.remove)) # used to remove some features mostly for benchmarking the gp solver
remove = int(args.remove)
choose = np.array([ 1 if (28*remove-1<i<28*(28-remove)) and (remove <= i%28 < 28-remove) else 0 for i in range(X_train_raw.shape[1]) ])

inds = np.array(np.where(choose ==1)).flatten()

X_train = X_train_raw[:, inds]
X_test  = X_test_raw[:, inds]

print("X_train.shape = {}".format(X_train.shape))
print("X_test.shape = {}".format(X_test.shape))

clf = LogisticRegression(solver='lbfgs', 
    multi_class='multinomial',
    verbose=True, 
    max_iter=1000, n_jobs=6).fit(X_train, y_train)

print np.average(clf.predict(X_test) == y_test)
W = np.hstack((clf.intercept_[:,None], clf.coef_))
print W.shape

import time
start = time.time()
print "Starting to solve using gpkit"
a = LR2NBK_GP.LR2NBK(W)
a.setObj(X_train, y_train)
a.solve(solver = 'mosek_cli', verbose=1)
end = time.time()
print "Done solving! " + str( end - start ) 

utils.save("mnistk_meanbinarized_{}.pickle".format(W.shape[1]), (clf, a.save()) )
print np.average(np.mean(a.classify(X_test) == y_test)), np.average(clf.predict(X_test) == y_test)