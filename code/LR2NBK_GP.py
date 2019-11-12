import gpkit as gp
import math
import numpy as np
import utils

class LR2NBK(object):

    """
    W = weights of a Logistic Regression model
    """
    def __init__(self, W):
        self.W = W

        if W.shape[0] == 1:
            return Exception("Not supported for binary case")

        self.N = self.W.shape[1] - 1
        self.K = self.W.shape[0]

    def __init_vars__(self):

        self.p = []
        for k in range(self.K):
            self.p.append(gp.Variable("p_" + str(k)))
        
        self.A = []
        self.A_ = []


        for k in range(self.K):
            Ak = []
            Ak_ = []
            for i in range(self.N):    
                Ak.append (gp.Variable("a_"  + str(i+1) + "," + str(k) ))
                Ak_.append(gp.Variable("a'_" + str(i+1) + "," + str(k) ))
            
            self.A.append(Ak)
            self.A_.append(Ak_)
        


    def __init_constraints__(self, EM = False, EPS=1e-9):
        self.constraints = []

        P = self.p[0]
        for k in range(1, self.K):
            P += self.p[k]
        self.constraints.append(P <= 1)
    
        for k in range(self.K):
            for i in range(self.N):
                self.constraints.append(self.A[k][i] + self.A_[k][i] <= 1.0 )
    
        
        for k in range(1, self.K):
            for i in range(self.N):       
                # should use W[.][i+1] since A[0] coressponds to W[.][1]
                temp = math.exp(self.W[k][i+1] - self.W[0][i+1] + EPS)
                temp *= self.A[k][i]**(-1)
                temp *= self.A_[k][i] * self.A[0][i] * self.A_[0][i]**(-1)

                self.constraints.append( temp  == 1 )


        for k in range(1, self.K):  
            Ck = math.exp(self.W[k][0] - self.W[0][0] + EPS) * self.p[k]**(-1) * self.p[0]    
            for i in range(self.N):
                Ck *= self.A_[k][i]**(-1) * self.A_[0][i]
    
            self.constraints.append( Ck == 1 )
    

    def setObj(self, X, Y, divider = 1.0, c = 1.0):
        self.__init_vars__()
        self.__init_constraints__()

        NN = len(Y)
        divider *= NN
        
        self.obj = 1         
        for k in range(self.K):
            n_p_k = -1 * (np.sum(Y == k) + c) / divider
            self.obj *= self.p[k]**(n_p_k)

        
        for k in range(self.K):
            for i in range(self.N):
                na  = -1 * float(np.sum( (X[:,i] == 1) & (Y == k)) + c) / divider 
                na_ = -1 * float(np.sum( (X[:,i] == 0) & (Y == k)) + c) / divider 

                self.obj *= self.A[k][i]**(na)
                self.obj *= self.A_[k][i]**(na_) 
                
    # Assuming everything is already set-up
    def solve(self, solver = 'mosek_cli', verbose = False):
        self.model = gp.Model(self.obj, self.constraints)    
        self.solution = self.model.solve(verbosity = verbose, solver=solver)

        sol = self.solution['freevariables']

        self._P = np.array([ sol["p_" + str(k)] for k in range(self.K)], dtype="float")
        self._AKI = np.zeros((self.N, self.K), dtype="float")
        self._A_KI = np.zeros((self.N, self.K), dtype="float")

        for i in range(self.N):
            for k in range(self.K):
                self._AKI [i][k] = sol[ "a_"  + str(i+1) + "," + str(k) ]
                self._A_KI[i][k] = sol[ "a'_"  + str(i+1) + "," + str(k)  ]

        self._ready_()

    def save(self):
        return (self.W, self.N, self.K, self._P, self._AKI, self._A_KI)

    def load(self, myTuple):
        self.W, self.N, self.K, self._P, self._AKI, self._A_KI = myTuple
        self._ready_()

    def PK(self, k):
        return self._P[k]
    
    def AKI(self, k, i):
        return self._AKI[i][k]

    def A_KI(self, k, i):
        return self._A_KI[i][k]

    def classify(self, X, missing = None, prob = False, EPS = 1e-14):
        return self.classify_fast(X, missing, prob=prob)
        result = np.zeros( (len(X), self.K) )
        sum = np.zeros( len(X) )

        for j in range(len(X)):
            for k in range(self.K):
                val_k = np.log( self.PK(k) + EPS) - np.log( self.PK(0) + EPS)
                for i in range(self.N):
                    if (not missing is None) and missing[j][i] == True:
                        continue
                    
                    if X[j][i] == 1:
                        val_k += np.log(self.AKI(k, i) + EPS) - np.log(self.AKI(0, i) + EPS)                        
                    else:
                        val_k += np.log(self.A_KI(k, i) + EPS) - np.log(self.A_KI(0, i) + EPS)
            
                result[j][k] = np.exp(val_k)
                sum[j] += np.exp(val_k)

            for k in range(self.K):
                result[j][k] = result[j][k] / (sum[j] + EPS)
        if prob:
            return result
        else:
            return np.argmax(result, axis=1)

    def classify_fast(self, X, missing = None, prob = False):
        mX = X
        mX_ = 1 - X
        if not missing is None:
            mX = mX * (1 - missing)
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

    def _ready_(self, EPS = 1e-200):
        self.mP  = np.log(EPS + np.matrix(self._P))
        self.mA  = np.log(EPS + np.matrix(self._AKI))
        self.mA_ = np.log(EPS + np.matrix(self._A_KI))

        for i in range(1, self.K):
            self.mA[:,i]  -= self.mA[:,0]
            self.mA_[:,i] -= self.mA_[:,0]
            self.mP[0,i]  -= self.mP[0,0]

        self.mA [:,0] = 0
        self.mA_[:,0] = 0
        self.mP [0,0] = 0



    def prob_x_given_c(self, X, eps = 1e-8):
        mX = X
        mX_ = 1 - X
        mX = np.matrix(mX)
        Z1 = np.exp(mX * self.mA + mX_ * self.mA_ + self.mP)
        Z2 = Z1 / np.sum(Z1, axis = 1)
        return np.array(Z2)