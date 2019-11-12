import cvxpy as cp
import math
import numpy as np
import utils

class LR2NBK(object):

    """
    W = weights of a Logistic Regression model
    """
    def __init__(self, W):
        self.W = W
        self.N = W.shape[1] - 1
        self.K = W.shape[0]

        self.__init_vars__()
        self.__init_constraints__()


    def __init_vars__(self):

        self.p = []
        for k in range(self.K):
            self.p.append(cp.Variable(pos=True, name="p_" + str(k)))
        
        self.A = []
        self.A_ = []


        for k in range(self.K):
            Ak = []
            Ak_ = []
            for i in range(self.N):    
                Ak.append (cp.Variable(pos=True, name="a_"  + str(i+1) + ", " + str(k) ))
                Ak_.append(cp.Variable(pos=True, name="a'_" + str(i+1) + ", " + str(k) ))
            
            self.A.append(Ak)
            self.A_.append(Ak_)
        


    def __init_constraints__(self, EPS=1e-9):
        self.constraints = []

        P = self.p[0]
        for k in range(1, self.K):
            P += self.p[k]
        self.constraints.append(P <= 1)

        
        for k in range(self.K):
            for i in range(self.N):        
                self.constraints.append(self.A[k][i] + self.A_[k][i] <= 1 )
        
        for k in range(1, self.K):
            for i in range(self.N):       
                # should use W[i+1] since A[0] coressponds to W[1]..
                self.constraints.append( math.exp(self.W[k][i+1] - self.W[0][i+1] + EPS) * self.A[k][i]**(-1) * self.A_[k][i] * self.A[0][i] * self.A_[0][i]**(-1) == 1 )

        for k in range(1, self.K):  
            Ck = math.exp(self.W[k][0] - self.W[0][0] + EPS) * self.p[k]**(-1) * self.p[0]    
            for i in range(self.N):
                Ck *= self.A_[k][i]**(-1) * self.A_[0][i]
    
            self.constraints.append( Ck == 1 )  
    

    def setObj(self, X, Y, divider = 1.0, c = 1.0):

        self.obj = 1         
        for k in range(self.K):
            n_p_k = (np.sum(Y == k) + c) / divider
            self.obj *= self.p[k]**(n_p_k)

        
        for k in range(self.K):
            for i in range(self.N):
                na  = float(np.sum( (X[:,i] == 1) & (Y == k)) + c) / divider 
                na_ = float(np.sum( (X[:,i] == 0) & (Y == k)) + c) / divider 

                self.obj *= self.A[k][i]**(na)
                self.obj *= self.A_[k][i]**(na_) 

                
    # Assuming everything is already set-up
    def solve(self, solver = cp.ECOS, verbose = False):
        self.problem = cp.Problem(cp.Maximize(self.obj), self.constraints)    

        #try:
        self.problem.solve(solver=solver, gp = True, verbose = verbose, max_iters=200)
        # except cp.SolverError as a:            
        #     print(self.problem.status)
        #     print("Trying a different solver")
        #     self.problem.solve(solver=cp.SCS, gp = True, verbose = verbose, parallel = parallel)


    def classify(self, X, missing = None, prob = False, EPS = 1e-14):
        result = np.zeros( (len(X), self.K) )
        sum = np.zeros( len(X) )

        for j in range(len(X)):
            for k in range(self.K):
                val_k = np.log( self.p[k].value + EPS) - np.log( self.p[0].value + EPS)
                for i in range(self.N):
                    if (not missing is None) and missing[j][i] == True:
                        continue
                    
                    if X[j][i] == 1:
                        val_k += np.log(self.A[k][i].value + EPS) - np.log(self.A[0][i].value + EPS)                        
                    else:
                        val_k += np.log(self.A_[k][i].value + EPS) - np.log(self.A_[0][i].value + EPS)
            
                result[j][k] = np.exp(val_k)
                sum[j] += np.exp(val_k)

            for k in range(self.K):
                result[j][k] = result[j][k] / (sum[j] + EPS)
        if prob:
            return result
        else:
            return np.argmax(result, axis=1)