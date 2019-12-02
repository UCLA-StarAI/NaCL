import gpkit as gp
import math
import numpy as np
import utils

class LR2NB(object):

    """
    W = weights of a Logistic Regression model
    """
    def __init__(self, W):
        self.W = W
        self.N = len(W) - 1


    def __init_vars__(self):

        self.p  = gp.Variable("p")
        self.p_ = gp.Variable("p_")

        self.A = []
        self.A_ = []
        self.B = []
        self.B_ = []

        for i in range(self.N):
            self.A. append(gp.Variable("a_" + str(i+1)))
            self.A_.append(gp.Variable("a'_" + str(i+1)))
            self.B. append(gp.Variable("b_" + str(i+1)))
            self.B_.append(gp.Variable("b'_" + str(i+1)))

    def __init_constraints__(self, EPS=1e-9):
        self.constraints = [
            self.p + self.p_ <= 1
        ]

        for i in range(self.N):
            self.constraints.append(self.A[i] + self.A_[i] <= 1 )
            self.constraints.append(self.B[i] + self.B_[i] <= 1 )   
            
            # should use W[i+1] since A[0] coressponds to W[1]
            self.constraints.append( math.exp(self.W[i+1]) * self.A[i]**(-1) * \
                self.A_[i] * self.B[i] * self.B_[i]**(-1) == 1 )
            
        C = (math.exp(EPS + self.W[0])) * self.p**(-1) * self.p_
        for i in range(self.N):
            C *= self.A_[i]**(-1) * self.B_[i]
    
        self.constraints.append( C == 1 )  
    

    def setObj(self, X, Y, divider = 1.0, c = 1.0):
        self.__init_vars__()
        self.__init_constraints__()

        NN = len(Y)
        divider *= NN

        n_t = -1.0 * (np.sum(Y == 1) + c) / divider 
        n_f = -1.0 * (np.sum(Y == 0) + c) / divider
    
        
        self.obj = self.p**(n_t) * self.p_**(n_f)

        for i in range(self.N):
            na  = -1.0 * float(np.sum( (X[:,i] == 1) & (Y == 1)) + c) / divider 
            na_ = -1.0 * float(np.sum( (X[:,i] == 0) & (Y == 1)) + c) / divider 
            nb  = -1.0 * float(np.sum( (X[:,i] == 1) & (Y == 0)) + c) / divider 
            nb_ = -1.0 * float(np.sum( (X[:,i] == 0) & (Y == 0)) + c) / divider 

            self.obj *= self.A[i]**(na)
            self.obj *= self.A_[i]**(na_) 
            self.obj *= self.B[i]**(nb)
            self.obj *= self.B_[i]**(nb_) 
                

    def solve(self, solver ='cvxopt', verbose = False):
            self.model = gp.Model(self.obj, self.constraints)    
            self.solution = self.model.solve(verbosity = verbose, solver=solver)
            
            sol = self.solution['freevariables']

            self._p = sol["p"]
            self._p_ = sol["p_"]

            self._A = np.zeros(self.N, dtype="float")
            self._A_ = np.zeros(self.N, dtype="float")
            self._B = np.zeros(self.N, dtype="float")
            self._B_ = np.zeros(self.N, dtype="float")

            for i in range(self.N):
                self._A[i]  = sol["a_" + str(i+1)]
                self._A_[i] = sol["a'_" + str(i+1)]
                self._B[i]  = sol["b_" + str(i+1)]
                self._B_[i] = sol["b'_" + str(i+1)]
            
            self._ready_()


    def save(self):
        return (self.W, self.N, self._p, self._p_, self._A, self._A_, self._B, self._B_)

    def load(self, myTuple):
        self.W, self.N, self._p, self._p_, self._A, self._A_, self._B, self._B_ = myTuple
        self._ready_()


    def _ready_(self, EPS = 1e-200):
        self.mP  = np.log(EPS + np.matrix( (self._p_, self._p)))
        
        self.mA  = np.log(EPS + np.matrix((self._B, self._A))).T
        self.mA_  = np.log(EPS + np.matrix((self._B_, self._A_))).T


    def classify(self, X, missing = None, prob = False):
        mX = X
        mX_ = 1 - X
        if not missing is None:
            mX = mX  * (1 - missing)
            mX_ = mX_ * (1 - missing)

        mX = np.matrix(mX)

        Z1 = np.exp(mX * self.mA + mX_ * self.mA_ + self.mP)
        Z2 = Z1 / np.sum(Z1, axis = 1)

        if prob:
            return np.array(Z2)
        else:
            yHatz = np.argmax(Z2, axis = 1)
            yHat = np.array([int(yHatz[i][0]) for i in range(yHatz.shape[0])]).reshape(1, -1)
            return yHat

    def predict_proba(self, X, missing = None):
        return self.classify(X, missing = missing, prob = True)

    def prob_x_given_c(self, X, eps = 1e-8):
        mX = X
        mX_ = 1 - X
        mX = np.matrix(mX)
        Z1 = np.exp(mX * self.mA + mX_ * self.mA_ + self.mP)
        Z2 = Z1 / np.sum(Z1, axis = 1)
        return np.array(Z2)
