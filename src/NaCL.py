import gpkit as gp
import math
import numpy as np
from sklearn.linear_model import LogisticRegression

class NaCLK(object):
    """
    This class should be used when we have more than 2 classes.
    clf = LogisticRegression trained model
    X = training data (binary)
    Y = labels (discrete) should be integers from 0:k
    divider = float. Divide all the powers in the objective. Does not change the optimal. For a more stable optimization.
    c = float. pseudocount, add c to all n_theta 
    """

    def __init__(self):
        pass

    def setup(self, clf: LogisticRegression, X, Y, divider=1.0, c = 1.0, EPS=1e-9):

        self.W = np.hstack((clf.intercept_[:,None], clf.coef_))
        assert self.W.shape[0] > 1 # need bigger than 2 classes. use NaCL2 for when there is exactly two classes
        self.N = self.W.shape[1] - 1
        self.K = self.W.shape[0]


        # init variables
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

        ## init constraints
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

        ## Objective
        divider *= len(Y) # for more stable optimization
        
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

        return self

    def solve(self, solver = 'cvxopt', verbose= False):
        """
        Other options for solver = "mosek_cli", "mosek_conif". 
        For python3 it seems 'mosek_config' is best
        """
        
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
        return self

    def _ready_(self, EPS = 1e-200):
        """
        Puts variables in matrix format
        """
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

    def __classify_fast__(self, X, missing = None, prob = False):
        """
        Calculate p(c | x^o) using matrix multipication. x^o is the observed features.
        """
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

    def predict_proba(self, X, missing = None):
        return self.__classify_fast__(X, missing = missing, prob = True)

    def predict(self, X, missing = None):
        return self.__classify_fast__(X, missing, prob=False)

    def save(self):
        return (self.W, self.N, self.K, self._P, self._AKI, self._A_KI)

    def load(self, myTuple):
        self.W, self.N, self.K, self._P, self._AKI, self._A_KI = myTuple
        self._ready_()
        return self


class NaCL2(object):
    """
    This class should be used when we have exactly 2 classes.
    clf = LogisticRegression trained model
    X = training data (numpy array binary)
    Y = labels (bool) numpy array of 0 or 1s
    divider = float. Divide all the powers in the objective. Does not change the optimal. For a more stable optimization.
    c = float. pseudocount, add c to all n_theta 
    """

    def __init__(self):
        pass

    def setup(self, clf: LogisticRegression, X, Y, divider=1.0, c = 1.0, EPS=1e-9):
        W = np.hstack((clf.intercept_[:,None], clf.coef_))
            

        assert W.shape[0] == 1 # need exactly 2 classes, use NaCLK for when there is more than two classes
        assert clf.multi_class != "multinomial" # should not use mutlinomil with 2 classes, nacl does not support multinomial clf's for 2 classes

        self.W = W[0]
        self.N = len(self.W) - 1


        # init vars
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

        # init constraints
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


        # objective
        divider *= len(Y)

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

        return self

    def solve(self, solver = 'cvxopt', verbose=False):
        """
        Other options for solver = "mosek_cli", "mosek_conif". 
        For python3 it seems 'mosek_config' is best
        """

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
        return self

    def __check_solution__(self, EPS=1e-6):
        good = True
        good &= ( abs(self._p + self._p_ - 1) < EPS )
        for i in range(self.N):
            good &= ( abs(self._A[i] + self._A_[i] - 1) < EPS )   
            good &= ( abs(self._B[i] + self._B_[i] - 1) < EPS  ) 
            good &= ( abs(math.exp(self.W[i+1]) * self._A[i]**(-1) * self._A_[i] * self._B[i] * self._B_[i]**(-1) - 1 ) < EPS )
            
        C = (math.exp(EPS + self.W[0])) * self._p**(-1) * self._p_
        for i in range(self.N):
            C *= self._A_[i]**(-1) * self._B_[i]
        good &= (abs( C  - 1 )  < EPS)
        return good

    def _ready_(self, EPS = 1e-200):
        """
        Put solution thetas in matrix format
        """
        self.mP  = np.log(EPS + np.matrix( (self._p_, self._p)))
        self.mA  = np.log(EPS + np.matrix((self._B, self._A))).T
        self.mA_  = np.log(EPS + np.matrix((self._B_, self._A_))).T

    
    def save(self):
        return (self.W, self.N, self._p, self._p_, self._A, self._A_, self._B, self._B_)

    def load(self, myTuple):
        self.W, self.N, self._p, self._p_, self._A, self._A_, self._B, self._B_ = myTuple
        self._ready_()
        return self

    def __classify_fast__(self, X, missing = None, prob = False):
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
        return self.__classify_fast__(X, missing = missing, prob = True)

    def predict(self, X, missing = None):
        return self.__classify_fast__(X, missing = missing, prob=False)