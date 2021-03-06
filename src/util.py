'''
author: Xiaowu He. 2017.9.19
'''
import scipy
import os
import numpy as np
from numpy.random import normal # generate transforming matrix


def precision_at_k(truth, vote, k=1, sparse=True):
    '''
    evaluate precision at k for a vote vector
    p@k = num of correct prediction in topk / k
    
    truth: scipy sparse matrix: shape = [sample, label]
    vote: kNN voted matrix, list of sample * scipy sparse matrix [1,label]
    '''
    #import pdb; pdb.set_trace()
    if sparse:
        success = 0
        for i in range(truth.shape[0]):
            # find the k-largest index using partition selet
            # topk are not sorted, np.argsort(vote[topk]) can do that but not needed here
            if vote[i].data.shape[0] < k:
                topk = vote[i].indices # in case the number of non-zero elements too small
            else:
                topk = vote[i].data.argpartition(-k)[-k:] # k largest's index in data
                topk = vote[i].indices[topk] # col index of data
                success += truth[i, topk].sum()
        
    else:
        sucess = 0
        for i in range(truth.shape[0]):
            topk = np.argpartition(vote[i], -k)[-k:]
            sucess += truth[i, topk].sum()
    return success / ((float(truth.shape[0]))*k)
        

def map_2_z(Y, L_hat):
    np.random.seed(0)
    M = normal(size=(L_hat, Y.shape[1]))
    Z = sign(Y.dot(M.T)) # z = n*\hat L
    return Z

def sign(Z): 
    return np.apply_along_axis(lambda x: [0 if elem < 0 else 1 for elem in x], 0, Z) #sign

def group_test(Y, L_hat, sparsity):
    np.random.seed(0)
    M = np.random.binomial(1, p=1./(sparsity+1), size=(L_hat, Y.shape[1]))
    Z = Y.dot(M.T) # z = n*\hat L
    return np.apply_along_axis(lambda x: [1 if elem > 0 else 0 for elem in x], 0, Z)

def hamming(y_true, y_pred):
    '''
    return an np array: hamming distance of every sample between y_true and y_pred
    '''
    hamming = []
    for i in range(y_pred.shape[0]):
        hamming.append((y_pred[i]!=y_true[i]).sum())
    return np.array(hamming) / float(y_true.shape[1])

def num_to_bin(x, length=None):
    '''
    convert a single label to corresponding binary vector
    '''
    import math
    if length == None:
        length=int(math.log(x)/math.log(2)) + 1
    bits = [0]*length
    i = 0
    while x:
        bits[i] = x%2
        x >>= 1
        i+=1
    return bits


class Predictor(object):
    '''
    This is the regular predictor
    Predict directly from the model using the classifier
    
    Attribute:
        clf: randomforest classifier clf.predcit(X)==>Z_pred
        index: nn search indexer
        Y_tr: the training data of Y in order to help kNN search
        
    Method:
        predict_z: clf.predict(X)
        predict: use predict_z and kNN to take a weighted version of Y
    '''
    def __init__(self, classifier, indexer, Y_tr):
        self.clf = classifier
        self.index=indexer
        self.Y_tr=Y_tr #useful when recover Y from kNN search
        
    def predict_z(self, X):
        return self.clf.predict(X)
    
    def predict(self, X, voter=30):
        '''
        voter=30: the number of NN we want to find
        
        return:
            list of sample * scipy sparse matrix [1,label]
        '''
        Z_pred = self.predict_z(X)
        dist, ind = self.index.search(Z_pred.astype('float32'), voter)
        return [np.sum([self.Y_tr[indij]/float(distij*distij+0.01)\
                        for indij, distij in zip(indi, disti)])\
                        for indi, disti in zip(ind,dist)]
    
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
def fit_bit(method, X, y):
    return method().fit(X, y)

def predict_bit(clf, X):
    return clf.predict(X)


class OvsA():
    '''
    use OvsA technic to predict one bit in y by a base classifier
    '''
    def __init__(self, method=LogisticRegression, n_jobs=-1):
        '''
        method: 
            the function to generate the base classifiers.
        '''
        self.method = method
        self.n_jobs = n_jobs
        
    def predict(self, X):
        bits = Parallel(n_jobs=self.n_jobs)(delayed(predict_bit)(clf, X)
                                           for clf in self.clfs)
#         bits = [clf.predict(X) for clf in self.clfs]
        return np.stack(bits, axis=1)
    
    def fit(self, X, y):
        self.clfs = Parallel(n_jobs=self.n_jobs)(delayed(fit_bit)(self.method, X, y[:,i]) 
                                     for i in range(y.shape[1]))
#         self.clfs = [self.method().fit(X, y[:, i]) for i in range(y.shape[1])]


class CombinedClf():
    '''
    use two different multilabel classifier classifier for
        X -> y[:, :x] and X -> y[:, x:]
    '''
    def __init__(self, clf1, clf2, seperate):
        '''
        input:
            method1,2: 
                method that return a new classifier needed
            sperate:
                the part that seperates which classifier to use
            n_jobs:
                number of treads to use
        '''
        self.clf1 = clf1
        self.clf2 = clf2
        self.seperate = seperate
        
    def predict(self, X):
        y1 = self.clf1.predict(X)
        y2 = self.clf2.predict(X)
        return np.append(y1, y2, axis=1)
    
    def fit(self, X, y):
        self.clf1.fit(X, y[:, :self.seperate])
        self.clf2.fit(X, y[:, self.seperate:])