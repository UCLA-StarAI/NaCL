import cvxpy as cp
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

        self.__init_vars__()
        self.__init_constraints__()


    def __init_vars__(self):

        self.p = cp.Variable(pos=True, name="p")
        self.p_ = cp.Variable(pos=True, name="p'")

        self.A = []
        self.A_ = []
        self.B = []
        self.B_ = []

        for i in range(self.N):
            self.A.append(cp.Variable(pos=True, name="a_" + str(i+1)))
            self.A_.append(cp.Variable(pos=True, name="a'_" + str(i+1)))
            self.B.append(cp.Variable(pos=True, name="b_" + str(i+1)))
            self.B_.append(cp.Variable(pos=True, name="b'_" + str(i+1)))

    def __init_constraints__(self):
        self.constraints = [
            self.p + self.p_ <= 1
        ]

        for i in range(self.N):
            self.constraints.append(self.A[i] + self.A_[i] <= 1 )
            self.constraints.append(self.B[i] + self.B_[i] <= 1 )   
            
            # should use W[i+1] since A[0] coressponds to W[1]..
            self.constraints.append( math.exp(self.W[i+1]) * self.A[i]**(-1) * \
                self.A_[i] * self.B[i] * self.B_[i]**(-1) == 1 )
            
        C = (math.exp(self.W[0]) + 1e-7 ) * self.p**(-1) * self.p_
        for i in range(self.N):
            C *= self.A_[i]**(-1) * self.B_[i]
    
        self.constraints.append( C == 1 )  


    """
    P_X = array of float, P'(X_i)
    """
    def add_marginal_constraints(self, P_X, eps = 1e-7):
        assert P_X.shape[0] == self.N
        for i in range(self.N):
            C = self.A[i] * self.p / (P_X[i] + eps) + self.B[i] * self.p_ / (P_X[i] + eps) 
            self.constraints.append( C <= 1 ) 
    
    def setBadObj(self, X, Y, divider = 1, c = 1):
        n_t = 1 
        n_f = 1 
    
        self.obj = self.p * self.p_
        for i in range(self.N):
            na = 1 
            na_ = 1 
            nb = 1 
            nb_ = 1 
            
            self.obj *= self.A[i] #**(na)
            self.obj *= self.A_[i]#**(na_) 
            self.obj *= self.B[i] #**(nb)
            self.obj *= self.B_[i]#**(nb_) 
    


    def setObj(self, X, Y, divider = 1.0, c = 1.0):
        n_t = (np.sum(Y) + c)/ divider 
        n_f = (len(Y) - np.sum(Y) + c) / divider
    
        self.obj = self.p**(n_t ) * self.p_**(n_f)
            
        for i in range(self.N):
            na  = float(np.sum( (X[:,i] == 1) & (Y == 1)) + c) / divider 
            na_ = float(np.sum( (X[:,i] == 0) & (Y == 1)) + c) / divider 
            nb  = float(np.sum( (X[:,i] == 1) & (Y == 0)) + c) / divider 
            nb_ = float(np.sum( (X[:,i] == 0) & (Y == 0)) + c) / divider 

            self.obj *= self.A[i]**(na)
            self.obj *= self.A_[i]**(na_) 
            self.obj *= self.B[i]**(nb)
            self.obj *= self.B_[i]**(nb_) 

    def setObjWithoutP(self, X, Y, divider = 1.0, c = 1.0):
        self.obj = self.p**(c/ divider) * self.p_**(c/divider)
            
        for i in range(self.N):
            na  = float(np.sum( (X[:,i] == 1) & (Y == 1)) + c) / divider 
            na_ = float(np.sum( (X[:,i] == 0) & (Y == 1)) + c) / divider 
            nb  = float(np.sum( (X[:,i] == 1) & (Y == 0)) + c) / divider 
            nb_ = float(np.sum( (X[:,i] == 0) & (Y == 0)) + c) / divider 

            self.obj *= self.A[i]**(na)
            self.obj *= self.A_[i]**(na_) 
            self.obj *= self.B[i]**(nb)
            self.obj *= self.B_[i]**(nb_)


    """
    Assuming data points are repeated, so X[0] == X[1]
    But Y[0] = P(Y= 0 | X[0]), Y[1] = P(Y=1 | X[1])
    """
    def setObjEM(self, X, Y, divider = 1.0, c = 1.0):
        N = len(Y) // 2
        sum_1 = np.sum([Y[i] for i in range(1, 2*N, 2)])
        sum_2 = np.sum([Y[i] for i in range(0, 2*N, 2)])
        n_t = (sum_1 + c) / divider 
        n_f = (sum_2 + c) / divider   

        self.obj = self.p**(n_t ) * self.p_**(n_f)
        #self.obj = self.A[0]

        for i in range(self.N):
            na  = (c + np.sum( [Y[j] if X[j][i] == 1 else 0  for j in range(1, 2*N, 2)] )) /divider
            na_ = (c + np.sum( [Y[j] if X[j][i] == 0 else 0  for j in range(1, 2*N, 2)] )) /divider
            nb  = (c + np.sum( [Y[j] if X[j][i] == 1 else 0  for j in range(0, 2*N, 2)] )) /divider 
            nb_ = (c + np.sum( [Y[j] if X[j][i] == 0 else 0  for j in range(0, 2*N, 2)] )) /divider 

            if i == 25:
                print( (na, na_, nb, nb_) )

            self.obj *= self.A[i]**(na)
            self.obj *= self.A_[i]**(na_) 
            self.obj *= self.B[i]**(nb)
            self.obj *= self.B_[i]**(nb_) 
                

    # Assuming everything is already set-up
    def solve(self, verbose = False, parallel=False):
        self.problem = cp.Problem(cp.Maximize(self.obj), self.constraints)
        
        # print(self.problem)
        
        assert not self.problem.is_dcp()
        assert self.problem.is_dgp()

        try:
            self.problem.solve(gp = True, verbose = verbose, parallel = parallel)
        except cp.SolverError as a:
            #print(a)
            print(self.problem.status)
            print("Trying a different solver")
            self.problem.solve(solver=cp.SCS, gp = True, verbose = verbose, parallel = parallel)


    def __validate_relaxed_sums__(self, EPS = 1e-6):
        # TODO add relaxed sum for marginals if they are added
        if math.fabs(self.p.value + self.p_.value - 1) > EPS:
            return False
        for i in range(self.N):
            if math.fabs(self.A[i].value + self.A_[i].value - 1) > EPS:
                return False
            if math.fabs(self.B[i].value + self.B_[i].value - 1) > EPS:
                return False

        return True


    def print_relaxed_sums(self):
        # TODO add relaxed sum for marginals if they are added
        print ( self.p.value + self.p_.value )
        for i in range(self.N):
            print( (self.A[i].value + self.A_[i].value, self.B[i].value + self.B_[i].value) ) 


    def solved_logistic_weights(self):
        w = [0 for i in range(self.N + 1)]

        w[0] = np.log( self.p.value / self.p_.value ) + np.sum ( [ np.log(self.A_[i].value / self.B_[i].value)  for i in range(self.N)] )

        for i in range(self.N):
            w[i+1] = np.log( (self.A[i].value * self.B_[i].value) /(self.B[i].value * self.A_[i].value) )

        return w

    def classify(self, X, missing = None, EPS = 1e-8):
        result = []

        for j in range(len(X)):
            val = np.log( self.p.value / self.p_.value + EPS)

            for i in range(self.N):
                if (not missing is None) and missing[j][i] == True:
                    continue

                if X[j][i] == 1:
                    val += np.log(self.A[i].value + EPS)
                    val -= np.log(self.B[i].value + EPS)
                else:
                    val += np.log(self.A_[i].value + EPS)
                    val -= np.log(self.B_[i].value + EPS)

            if val >= 0:
                result.append(1)
            else:
                result.append(0)

        return result

    def prob_x_given_c(self, X, c, eps = 1e-8):
        datas = X.shape[0]
        result = np.zeros(datas)
        if c == 1:
            for d in range(datas):
                for i in range(self.N):
                    if X[d][i] == 1:
                        result[d] += np.log(self.A[i].value + eps)
                    else:
                        result[d] += np.log(self.A_[i].value + eps)

        else:
            for d in range(datas):
                for i in range(self.N):
                    if X[d][i] == 1:
                        result[d] += np.log(self.B[i].value + eps)
                    else:
                        result[d] += np.log(self.B_[i].value + eps)


        return np.exp(result)


    # def classify_fast(self, X, missing = None, EPS = 1e-8):
    #     result = np.zeros(X.shape[0], dtype='float')

    #     val = np.log( self.p.value / self.p_.value + EPS)

    #     for i in range(self.N):
    #         if (not missing is None) and missing[j][i] == True:
    #              continue
    #         if X[j][i] == 1:
    #              val += np.log(self.A[i].value + EPS)
    #              val -= np.log(self.B[i].value + EPS)
    #         else:
    #              val += np.log(self.A_[i].value + EPS)
    #              val -= np.log(self.B_[i].value + EPS)
        
   
    #     result = int(result > 0)
    #     return result

    def odd(self, X, missing = None, EPS = 1e-8, prob = False):
        result = []

        for j in range(len(X)):
            val = np.log( self.p.value / self.p_.value + EPS)

            for i in range(self.N):
                if (not missing is None) and missing[j][i] == True:
                    continue

                if X[j][i] == 1:
                    val += np.log(self.A[i].value + EPS)
                    val -= np.log(self.B[i].value + EPS)
                else:
                    val += np.log(self.A_[i].value + EPS)
                    val -= np.log(self.B_[i].value + EPS)

            result.append(val)
        
        if not prob:
            return result
        else:
            return utils.sigmoid(np.array(result))

    def predict_proba(self, X, missing = None, EPS = 1e-8):
        return self.odd(X, missing = missing, EPS = EPS, prob = True)