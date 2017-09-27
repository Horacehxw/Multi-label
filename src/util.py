'''
author: Xiaowu He. 2017.9.19
'''

import numpy as np
from numpy.random import normal # generate transforming matrix

def precision_at_k(truth, vote, k):
    '''
    evaluate precision at k for a vote vector
    p@k = num of correct prediction in topk / k
    '''
    success = 0
    for i in range(len(truth)):
        # find the k-largest index using partition selet
        # topk are not sorted, np.argsort(vote[topk]) can do that but not needed here
        topk = np.argpartition(vote[i], -k)[-k:] 
        success += np.sum(truth[i, topk])
    return success / ((float(len(truth)))*k)

def map_2_z(Y, L_hat):
    np.random.seed(0)
    M = normal(size=(L_hat, Y.shape[1]))
    Z = sign(M.dot(Y.T).T) # z = n*\hat L
    return Z

def sign(Z):
    return np.apply_along_axis(lambda x: [0 if elem < 0 else 1 for elem in x], 0, Z) #sign

def group_test(Y, L_hat, sparsity):
    np.random.seed(0)
    M = np.random.binomial(1, p=1./(sparsity+1), size=(L_hat, Y.shape[1]))
    Z = M.dot(Y.T).T
    return np.apply_along_axis(lambda x: [1 if elem > 0 else 0 for elem in x], 0, Z)

