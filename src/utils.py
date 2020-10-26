import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import pickle
from sklearn.metrics import f1_score 


def save(file, obj):
    with open(file,'wb') as outfile: 
         pickle.dump(obj, outfile)

def load(file):
    with open(file,'rb') as infile: 
        return pickle.load(infile)


def cross_entropy_conditional(p, q, eps = 1e-8):
    return 0 - ( np.sum( p * np.log(q + eps) + (1-p) * np.log(1-q +eps) ) )

def cross_entropy(p, q, eps = 1e-8):
    return 0 - np.sum( p * np.log(q + eps) )

def kl_divergence(p, q, eps = 1e-8):
    return 0 - np.sum( p * (np.log(q + eps) - np.log(p + eps + eps) ))

'''
should have predict_proba method

conditional log likelihood
'''
def log_likelihood(probs, Y, eps = 1e-9):
    return np.sum(np.log(probs + eps)*Y +  np.log(1-probs + eps) * (1-Y))

'''
p_c = p(c)
p_x_c[i] = p(x|c)
p_x_nc[i] = p(x| not c)
'''
def marginal_log_likelihood(p_c, p_x_c, p_x_nc, Y, eps = 1e-20):
    result = np.log(p_c * p_x_c + (1-p_c) * p_x_nc)
    return np.sum(result)

def marginal_log_likelihood_em(p_c, p_x_c, eps = 1e-20):
    result = np.log( np.sum(p_c * p_x_c + eps, axis = 1) ) 
    return np.sum(result)

def marginal_log_likelihood_k(P, PX, eps = 1e-20):
    return np.sum( np.log( P * PX + eps))

def conditional_likelihood_k(P, Q, eps = 1e-14):
    return (0.0 - np.sum(P * np.log(Q + eps))) / (1.0 * P.shape[0])

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def predict_nbk_with_missing(X, NB, missing, prob = False):
    mX = X
    mX_ = 1 - X
    if not missing is None:
        mX = mX * (1 - missing)
        mX_ = mX_ * (1 - missing)
    
    mX = np.matrix(mX)
    mX_ = np.matrix(mX_)

    mA = NB.feature_log_prob_
    mA_ =  np.log(1 - np.exp(mA) )

    mA = np.matrix(mA).T
    mA_ = np.matrix(mA_).T
    
    mP = NB.class_log_prior_

    Z1 = np.exp(mX * mA + mX_ * mA_ + mP)
    Z2 = Z1 / np.sum(Z1, axis = 1)
    if prob:
        return np.array(Z2)
    else:
        yHatz = np.argmax(Z2, axis = 1)
        yHat = np.array([int(yHatz[i][0]) for i in range(yHatz.shape[0])]).reshape(1, -1)
        return yHat

def predict_with_missing(X2_test, NB, missing, prob = False, eps=1e-8):
    """
    X2_test is X with of missing values imputed
    missing is same shape as X2_test, bool
    NB is sklearn Naive Bayes
    """
    Yhat2 = np.array([0 for i in range(X2_test.shape[0])])

    Yhat3 = np.array([0.0 for i in range(X2_test.shape[0])])
    
    for i in range(X2_test.shape[0]):
        num =   np.log( (NB.class_count_[0] + 1+eps)/(NB.class_count_[1] + 1+eps)  )   # NB.class_log_prior_[0] - NB.class_log_prior_[1]

        for j in range(X2_test.shape[1]):
            if missing[i][j]:
                continue

            if X2_test[i][j]:
                num += np.log( eps+ (NB.feature_count_[0][j]+eps + 1) / (NB.class_count_[0] + 1+eps)) 
                num -= np.log( eps+(NB.feature_count_[1][j]+1+eps) / (NB.class_count_[1] + 1+eps))
            else:
                num += np.log( eps+(NB.class_count_[0] - NB.feature_count_[0][j] + 1+eps ) / (NB.class_count_[0] + 1+eps )) 
                num -= np.log( eps+(NB.class_count_[1] - NB.feature_count_[1][j] + 1+eps) / (NB.class_count_[1] + 1+eps ))
        
        Yhat3[i] = num

        if num >= 0:
            Yhat2[i] = 0.0
        else:
            Yhat2[i] = 1.0

    if not prob:
        return Yhat2
    else:
        return sigmoid(-Yhat3)

def run_experiment(X_impute, X_test, y_test, clf, NB, a, repeat = 1, k_jump=10, K = None, FEATURES = None, return_all = False, use_clasffier_as_truth = False, use_cross_entropy = False, use_f1 = False):
    """
    X_impute: What value to impute for each missing features
    X_test: 
    y_test:
    clf: Logistic Regression directly trained
    NB: Naive Bayes directly trained 
    a: LR -> NB trained model using GP (see LR2NB.py)
    K: arrays of number of missing variables to have
    repeat: how many times sample input with k missing values
    FEATURES: features that might be removed
    """
    #X_mean = np.mean(X_train, axis=0)
    
    assert(not use_cross_entropy or not use_f1)

    YTrue = y_test
    if use_clasffier_as_truth:
        YTrue = clf.predict(X_test)

    features = X_test.shape[1]
    
    k_all = []
    missing_err_nb_all = []
    missing_err_lr_all = []
    missing_err_ours_all = []

    missing_err_nb = []
    missing_err_lr = []
    missing_err_ours = []
    
    if FEATURES is None:
        FEATURES = np.array( [i for i in range(features)] )
    else:
         FEATURES = np.array( FEATURES )

    print("Possible features to remove: {}".format(FEATURES.shape[0]))

    if K is None:
        K = [i for i in range(0, features, k_jump)]

    for k in K:
        print("K = {}".format(k))

        if k > FEATURES.shape[0]:
            print("Early stop: Only had {} features possible to remove".format(FEATURES.shape[0]))
            break

        err_nb = 0
        err_lr = 0
        err_ours = 0

        for R in range(repeat):
            X2_test = np.array(X_test, dtype = 'float')
        
            missing = np.zeros(X_test.shape, dtype=bool)
            
            for i in range(X2_test.shape[0]):
                #miss = np.random.choice(X2_test.shape[1], k, replace=False)
                miss = np.random.choice(FEATURES, k, replace=False)
                
                missing[i][miss] = True
                X2_test[i][miss] = X_impute[miss]

            Yhat2 = predict_with_missing(X2_test, NB, missing)
            A_nb = np.average(Yhat2 == YTrue)
            A_clf = np.average(clf.predict(X2_test) == YTrue)
            A_a = np.average(a.classify(X2_test, missing) == YTrue)
            
            if use_f1:
                A_nb = f1_score(YTrue,  predict_with_missing(X2_test, NB, missing))
                A_clf = f1_score(YTrue, clf.predict(X2_test))
                A_a = f1_score(YTrue, a.classify(X2_test, missing))

            if use_cross_entropy:
                Y_clf_prob = clf.predict_proba(X_test)[:,1]

                A_nb  = cross_entropy(Y_clf_prob, predict_with_missing(X2_test, NB, missing, prob= True))
                A_clf = cross_entropy(Y_clf_prob, clf.predict_proba(X2_test)[:,1])
                A_a   = cross_entropy(Y_clf_prob, a.odd(X2_test, missing, prob = True))

            k_all.append(k)
            missing_err_nb_all.append(A_nb)
            missing_err_lr_all.append(A_clf)
            missing_err_ours_all.append(A_a)
            
            err_nb +=  A_nb
            err_lr += A_clf
            err_ours += A_a

        missing_err_nb.append( err_nb/repeat )
        missing_err_lr.append( err_lr/repeat )
        missing_err_ours.append( err_ours/repeat )

    if not return_all:
        return missing_err_nb, missing_err_lr, missing_err_ours
    else:
        return k_all, missing_err_nb_all, missing_err_lr_all, missing_err_ours_all


def run_experiment_k_paper(X_test, y_test, clf, NB, nacl, setting):
    import impyute
    
    X_impute_mean   = np.mean(X_test, axis = 0)
    X_impute_median = np.median(X_test, axis = 0)
    X_impute_max    = np.max(X_test, axis = 0)
    X_impute_min    = np.min(X_test, axis = 0)
    X_impute_flip   = np.copy(1 - X_test)


    k_all = []
    missing_err_nb_all = []
    missing_err_lr_mean_all = []
    missing_err_lr_median_all = []
    missing_err_lr_max_all = []
    missing_err_lr_min_all = []
    missing_err_lr_flip_all = []
    missing_err_lr_em_impute_all = []
    missing_err_lr_mice_impute_all = []
    missing_err_lr_knn_impute_all = []

    discreteFeatures = setting["discreteFeatures"] if "discreteFeatures" in setting else 1
    featureEncoding = setting["feature_encoding"] if "feature_encoding" in setting else None

    do_emImpute = setting["emImpute"] if "emImpute" in setting else False
    do_miceImpute = setting["miceImpute"] if "miceImpute" in setting else False
    do_knnImpute = setting["knnImpute"] if "knnImpute" in setting else False

    verbose = setting["verbose"] if "verbose" in setting else True

    missing_err_ours_all = []

    useProb = setting["prob"] if "prob" in setting else True
    function = setting["function"] if "function" in setting else None
    if function is None:
        if useProb:
            function = conditional_likelihood_k
        else:
            function = f1_score

    if verbose:
        print("Using following function: ")
        print(function)
    
    repeat = setting["repeat"] if "repeat" in setting else 1

    FEATURES = setting["features"] if "features" in setting else None
    if FEATURES is None:
        NNN = X_test.shape[1]
        if not featureEncoding is None:
            NNN = len(featureEncoding)
        FEATURES = np.array( [i for i in range(int(NNN / discreteFeatures))] )
    else:
        FEATURES = np.array( FEATURES )

    if verbose:
        print("Possible features to remove: {}".format(FEATURES.shape[0]))

    K = setting["k"]

    for k in K:
        if verbose:
            print("K = {}".format(k))

        if k > FEATURES.shape[0]:
            print("Early stop: Only had {} features possible to remove".format(FEATURES.shape[0]))
            break

        cur_nb = []
        cur_lr_mean = []
        cur_lr_median = []
        cur_lr_max = []
        cur_lr_min = []
        cur_flip = []
        cur_em_impute = []
        cur_mice_impute = []
        cur_knn_impute = []

      
        cur_ours = []
        
        for R in range(repeat):
            if R % 10 == 0:
                if verbose:
                    print("\t R = {}".format(R))
            X_test_mean   = np.array(X_test, dtype = 'float')
            X_test_median = np.array(X_test, dtype = 'float')
            X_test_max    = np.array(X_test, dtype = 'float')
            X_test_min    = np.array(X_test, dtype = 'float')
            X_test_flip   = np.array(X_test, dtype = 'float')
            X_test_em_impute = np.array(X_test, dtype = 'float')
            X_test_mice_impute = np.array(X_test, dtype = 'float')
            X_test_knn_impute = np.array(X_test, dtype = 'float')
            missing = np.zeros(X_test.shape, dtype=bool)

            for i in range(X_test.shape[0]):
                miss = np.random.choice(FEATURES, k, replace=False)

                if not featureEncoding is None and k > 0:
                    missK = []
                    for m in miss:
                        for z in featureEncoding[m]:
                            missK.append(z)
                    miss = np.copy(np.array(missK))

                elif discreteFeatures != 1 and k > 0:
                    missK = []
                    for m in miss:
                        for z in range(discreteFeatures):
                            missK.append(m * discreteFeatures + z)
                    miss = np.copy(np.array(missK))

                   
                missing[i][miss] = True
                X_test_mean[i][miss]   = X_impute_mean[miss]
                X_test_median[i][miss] = X_impute_median[miss]
                X_test_max[i][miss]    = X_impute_max[miss]
                X_test_min[i][miss]    = X_impute_min[miss]
                X_test_flip[i][miss]   = X_impute_flip[i][miss]
                X_test_em_impute[i][miss] = np.nan
                X_test_mice_impute[i][miss] = np.nan
                X_test_knn_impute[i][miss] = np.nan

            if do_emImpute:
                import time
                start = time.time()
                loops = 6
                print ("\tStarting to em impute with loops = {}".format(loops))
                X_test_em_impute = impyute.em(X_test_em_impute, loops = loops)
                end = time.time()
                print ("\tDone imputing! " + str( end - start ) )
            else:
                X_test_em_impute = np.zeros(X_test.shape)

            if do_miceImpute:
                import time
                start = time.time()
                print ("\tStarting to mice impute")
                X_test_mice_impute = impyute.mice(X_test_mice_impute)
                end = time.time()
                print ("\tDone imputing! " + str( end - start ) )
            else:
                 X_test_mice_impute = np.zeros(X_test.shape)


            if do_knnImpute:
                import time
                start = time.time()
                print ("\tStarting to knn impute")
                X_test_knn_impute = impyute.fast_knn(X_test_knn_impute)
                end = time.time()
                print ("\tDone imputing! " + str( end - start ) )
            else:
                 X_test_knn_impute = np.zeros(X_test.shape)

            lr_prob = clf.predict_proba(X_test)
            
            if useProb:
                cur_nb.append         ( function(lr_prob, predict_nbk_with_missing(X_test_mean, NB, missing, prob = True)) )
                cur_lr_mean.append    ( function(lr_prob, clf.predict_proba(X_test_mean)) )
                cur_lr_median.append  ( function(lr_prob, clf.predict_proba(X_test_median)))
                cur_lr_max.append     ( function(lr_prob, clf.predict_proba(X_test_max)))
                cur_lr_min.append     ( function(lr_prob, clf.predict_proba(X_test_min)))
                cur_em_impute.append  ( function(lr_prob, clf.predict_proba(X_test_em_impute)))
                cur_mice_impute.append( function(lr_prob, clf.predict_proba(X_test_mice_impute)))
                cur_knn_impute.append ( function(lr_prob, clf.predict_proba(X_test_knn_impute)))    
                cur_ours.append       ( function(lr_prob, nacl.predict_proba(X_test, missing)))        

            else:
                cur_nb.append         ( function(y_test, predict_nbk_with_missing(X_test_mean, NB, missing)) )
                cur_lr_mean.append    ( function(y_test, clf.predict(X_test_mean)) )
                cur_lr_median.append  ( function(y_test, clf.predict(X_test_median)))
                cur_lr_max.append     ( function(y_test, clf.predict(X_test_max)))
                cur_lr_min.append     ( function(y_test, clf.predict(X_test_min)))
                cur_em_impute.append  ( function(y_test, clf.predict(X_test_em_impute)))
                cur_mice_impute.append( function(y_test, clf.predict(X_test_mice_impute)))
                cur_knn_impute.append( function(y_test, clf.predict(X_test_knn_impute)))
                cur_ours.append   ( function(y_test, nacl.predict(X_test_mean, missing)))
                
        
        k_all.append(k)
        missing_err_nb_all.append       (cur_nb)
        missing_err_lr_mean_all.append  (cur_lr_mean)
        missing_err_lr_median_all.append(cur_lr_median)
        missing_err_lr_max_all.append   (cur_lr_max)
        missing_err_lr_min_all.append   (cur_lr_min)
        missing_err_lr_flip_all.append  (cur_flip)
        missing_err_lr_em_impute_all.append(cur_em_impute)
        missing_err_lr_mice_impute_all.append(cur_mice_impute)
        missing_err_lr_knn_impute_all.append(cur_knn_impute)
        missing_err_ours_all.append  (cur_ours)

    
    #end of for loops
    
    missing_err_ours_all = np.array(missing_err_ours_all)

    data = {
        "features_count": FEATURES.shape[0],
        "k" :     np.array(k_all),
        "nb":     np.array(missing_err_nb_all),
        "mean":   np.array(missing_err_lr_mean_all),
        "median": np.array(missing_err_lr_median_all),
        "max":    np.array(missing_err_lr_max_all),
        "min":    np.array(missing_err_lr_min_all),
        "ours":   missing_err_ours_all,
        "flip":   np.array(missing_err_lr_flip_all),
        "em_impute": np.array(missing_err_lr_em_impute_all),
        "mice_impute": np.array(missing_err_lr_mice_impute_all),
        "knn_impute": np.array(missing_err_lr_knn_impute_all),
    }            

    return data

def plot_side_by_side_paper(data, data2, setting, setting2):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'errorbar.capsize': 2})
    matplotlib.rcParams.update({'figure.autolayout':True})
    
    font = {'size'   : 22}
    plt.rc('font', **font)
   
    SIZE = setting["size"] if "size" in setting else (8,6)
    saveAs = setting["saveAs"] if "saveAs" in setting else "plot.eps"

    fig, (ax1, ax2) = plt.subplots(1,2)
    #plt.figure(figsize=SIZE)
    fig.set_figheight(SIZE[1])
    fig.set_figwidth(SIZE[0]*2)

    
    plot_results_one_side(data, ax1, setting)
    plot_results_one_side(data2, ax2, setting2)

    plt.savefig(saveAs)
    plt.show()
    return plt
    
    
def plot_results_one_side(data, ax, setting, show_nb = False):
    K = data["k"]
    Ylabel = setting["Ylabel"] if "Ylabel" in setting else "Accuracy"
    Xlabel = setting["Xlabel"] if "Xlabel" in setting else "Percentage of Features Missing"
    title = setting["title"] if "title" in setting else "MNIST"
    multiply = setting["mult"] if "mult" in setting else 1.0

    legendInclude = setting["legend"] if "legend" in setting else True
    features_count = data["features_count"] if "features_count" in data else 1.0
    ax.set_title(title)

    choices = [
        "flip",
        "nb",
        "mean", 
        "median", 
        "max", 
        "min", 
        "ours",
        "em_impute",
        "mice_impute",
        "knn_impute",
    ]
    labels  = [
        "Flip",
        "Naive Bayes",
        "Mean Imp", 
        "Median Imp", 
        "Max Imp", 
        "Min Imp", 
        "NaCL",
        "EM Impute",
        "Mice Impute",
        "KNN Impute"
    ]
    fmts = [
        "y^-.",
        "y^-.",
        "bo--",
        "m+-.",
        "g^-.",
        "gv-.",
        "rx-",
        "y^-.",
        "y^-.",
        "y^-.",
    ]

    K = K / (0.01 * features_count)

    show = set(setting["show"]) if "show" in setting else set(["mean", "median", "min", "ours"])

    if show_nb and "nb" in data:
        ax.errorbar(K, multiply*np.mean(data["nb"], axis=1), yerr = multiply*np.std(data["nb"], axis=1) )    

    for i,c in enumerate(choices):
        if c in data and c in show:
            ax.errorbar(K, multiply*np.mean(data[c], axis=1), yerr = multiply*np.std(data[c], axis=1), label=labels[i], fmt=fmts[i] )

    if Ylabel != "":
        ax.set_ylabel(Ylabel)
    if Xlabel != "":
        ax.set_xlabel(Xlabel)
    if legendInclude:
        ax.legend(loc='best', fontsize = 'x-small')


def plot_results_paper(data, setting = {}, show_nb = False):
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.rcParams.update({'errorbar.capsize': 2})
    matplotlib.rcParams.update({'figure.autolayout':True})
    
    K = data["k"]
    font = {'size'   : 22}
    plt.rc('font', **font)

    SIZE = setting["size"] if "size" in setting else (8,6)
    plt.figure(figsize=SIZE)

    useEM = setting["em"] if "em" in setting else False
    saveAs = setting["saveAs"] if "saveAs" in setting else "plot.eps"
    Ylabel = setting["Ylabel"] if "Ylabel" in setting else "Accuracy"
    Xlabel = setting["Xlabel"] if "Xlabel" in setting else "Percentage of Features Missing"
    title = setting["title"] if "title" in setting else "MNIST"
    multiply = setting["mult"] if "mult" in setting else 1.0

    legendInclude = setting["legend"] if "legend" in setting else True
    features_count = data["features_count"] if "features_count" in data else 1.0
    plt.title(title)

    choices = [
        "flip",
        "nb",
        "mean", 
        "median", 
        "max", 
        "min", 
        "ours",
        "em_impute",
        "mice_impute",
    ]
    labels  = [
        "Flip",
        "Naive Bayes",
        "Mean Imp", 
        "Median Imp", 
        "Max Imp", 
        "Min Imp", 
        "NaCL",
        "EM Impute",
        "Mice Impute"
    ]
    fmts = [
        "y^-.",
        "y^-.",
        "bo--",
        "m+-.",
        "g^-.",
        "gv-.",
        "rx-",
        "y^-.",
        "y^-.",
    ]

    K = K / (0.01 * features_count)

    show = set(setting["show"]) if "show" in setting else set(["mean", "median", "min", "ours"])

    if show_nb and "nb" in data:
        plt.errorbar(K, multiply*np.mean(data["nb"], axis=1), yerr = multiply*np.std(data["nb"], axis=1) )    

    for i,c in enumerate(choices):
        if c in data and c in show:
            if useEM and c == "ours":
                for x in data[c]:
                    plt.errorbar(K, multiply*np.mean(data[c][x], axis=1), yerr = multiply*np.std(data[c][x], axis=1), label=x, fmt=fmts[i] )
            else:
                plt.errorbar(K, multiply*np.mean(data[c], axis=1), yerr = multiply*np.std(data[c], axis=1), label=labels[i], fmt=fmts[i] )

    if Ylabel != "":
        plt.ylabel(Ylabel)
    if Xlabel != "":
        plt.xlabel(Xlabel)
    if legendInclude:
        plt.legend(loc='best', fontsize = 'x-small')
    plt.savefig(saveAs)
    return plt

def load_mnist_5v3():
    folder = "../data/binaryizedMNIST-3-5/"

    X_train = pd.read_csv(folder + "train-3-5-images.txt").values
    y_train = pd.read_csv(folder + "train-3-5-labels.txt").values.ravel()

    X_test = pd.read_csv(folder + "test-3-5-images.txt").values
    y_test = pd.read_csv(folder + "test-3-5-labels.txt").values.ravel()


    return X_train, y_train, X_test, y_test

def load_fashion_binarized():
    
    folder = "../data/Fashion-0-1/"
    
    X_train = pd.read_csv(folder + "train-0-1-images.txt").values
    y_train = pd.read_csv(folder + "train-0-1-labels.txt").values.ravel()

    X_test = pd.read_csv(folder + "test-0-1-images.txt").values
    y_test = pd.read_csv(folder + "test-0-1-labels.txt").values.ravel()
    
    return X_train, y_train, X_test, y_test

def load_dataset(folder, label):
        
    X_train = pd.read_csv(folder + "/train-" + label + "-samples.txt").values
    y_train = pd.read_csv(folder + "/train-" + label + "-labels.txt").values.ravel()

    X_test = pd.read_csv(folder + "/test-" + label + "-samples.txt").values
    y_test = pd.read_csv(folder + "/test-" + label + "-labels.txt").values.ravel()
    
    return X_train, y_train, X_test, y_test

def load_digits_binarized():
    from sklearn.datasets import load_digits
    digits = load_digits()

    # Binarize
    X = digits.images
    X[ X < 8 ] = 0
    X[ X >=8 ] = 1

    # only to detect 0 vs non-zero for now
    Y = digits.target
    zeros = (Y == 0)
    non_zeros = (Y != 0)
    Y[ zeros] = 1
    Y[ non_zeros ] = 0
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    return X_train, y_train, X_test, y_test
