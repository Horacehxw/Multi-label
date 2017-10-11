'''
author: Xiaowu He. 2017.9.19
'''

import numpy as np
from numpy.random import normal # generate transforming matrix

def precision_at_k(truth, vote, k=1):
    '''
    evaluate precision at k for a vote vector
    p@k = num of correct prediction in topk / k
    '''
    success = 0
    for i in range(truth.shape[0]):
        # find the k-largest index using partition selet
        # topk are not sorted, np.argsort(vote[topk]) can do that but not needed here
        topk = vote[i].data.argpartition(-k)[-k:]
        topk = vote[i].indices[topk]
        success += truth[i, topk].sum()
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


