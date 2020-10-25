import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms
import sys
sys.path.append("src")

import utils
from NaCL import NaCLK

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--remove", help="remove #", default=0)

sys.setrecursionlimit(2000000)

train = datasets.MNIST('data', train=True, download=True)
test = datasets.MNIST('data', train=False, download=True)

X_train = train.data.numpy().reshape(-1, 28*28)
y_train = train.targets.numpy()

train_idx35 = (y_train == 9) | (y_train == 5) | (y_train == 3) | (y_train == 1)
X_train = X_train[train_idx35]
y_train = y_train[train_idx35]

X_test = test.data.numpy().reshape(-1, 28*28)
y_test = test.targets.numpy()

test_idx35 = (y_test == 9) | (y_test == 5) | (y_test == 3) | (y_test == 1)
X_test = X_test[test_idx35]
y_test = y_test[test_idx35]

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
    verbose=False, 
    max_iter=400, n_jobs=6).fit(X_train, y_train)

print ("test accuracy", np.average(clf.predict(X_test) == y_test))
W = np.hstack((clf.intercept_[:,None], clf.coef_))
print (W.shape)

import time
start = time.time()
print ("Starting to solve using gpkit")
nacl = NaCLK().setup(clf, X_train, y_train, c=0.01, divider=10)
nacl.solve(solver = 'mosek_conif', verbose=1)
end = time.time()
print ("Done solving! " + str( end - start ) )

# This should print 1, i.e. NaCL and the LogisticRegression agree down to probabliity level
print( np.mean(  np.abs(nacl.predict_proba(X_test) - clf.predict_proba(X_test)) < 1e-5 ) )

print("Saving to file")
utils.save("mnistk_{}.pickle".format(W.shape[1]), (clf, nacl.save()) )


print("Loading from file")
clf, nacl_data = utils.load("mnistk_{}.pickle".format(W.shape[1]))
nacl2 = NaCLK().load(nacl_data)
print( np.mean(  np.abs(nacl2.predict_proba(X_test) - clf.predict_proba(X_test)) < 1e-5 ) )
