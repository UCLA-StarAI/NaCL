import math
from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.append("src")

import utils
from NaCL import NaCLK

sys.setrecursionlimit(2000000)

X_train, y_train, X_test, y_test = utils.load_dataset("splice", "splice")
print("X_train.shape = {}".format(X_train.shape))
print("X_test.shape = {}".format(X_test.shape))

from sklearn.naive_bayes import BernoulliNB
NB = BernoulliNB().fit(X_train, y_train)
print(np.average(NB.predict(X_train) == y_train))
print(np.average(NB.predict(X_test) == y_test))

clf = LogisticRegression(solver='lbfgs', 
    multi_class='multinomial',
    verbose=True, 
    max_iter=1000, n_jobs=6).fit(X_train, y_train)

print (np.average(clf.predict(X_test) == y_test))

import time
start = time.time()
print ("Starting to solve using gpkit")
nacl = NaCLK().setup(clf, X_train, y_train, c=0.01, divider=10)\
    .solve(solver = 'mosek_conif', verbose=1)

# nacl.solve(solver = 'mosek_conif', verbose=1)
end = time.time()
print ("Done solving! " + str( end - start ) )


# This should print 1, i.e. NaCL and the LogisticRegression agree down to probabliity level
print( np.mean(  np.abs(nacl.predict_proba(X_test) - clf.predict_proba(X_test)) < 1e-5 ) )

# save to file
print("Saving to file")
utils.save("splice.pickle", (clf, nacl.save()) )


# load from file and initlalize
print("Loading from file")
clf, nacl_data = utils.load("splice.pickle")
nacl2 = NaCLK().load(nacl_data)
print( np.mean(  np.abs(nacl2.predict_proba(X_test) - clf.predict_proba(X_test)) < 1e-5 ) )