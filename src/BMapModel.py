import joblib
from sklearn.externals import joblib
import os
import numpy as np
from numpy.random import normal
import util

class BM_Predictor():
    '''
    v1.0: update the BIHT prediction method
    Binary Map multilabel prediction model
    
    Attribute:
        Mat: M, which maps y into z space
        L: dim of y
        L_hat: dim of z
        index(optional): kNN search index, faiss.index object
        Y_tr(optional): kNN recover index, training data
        model_path(optional): dir of classifiers
    '''
    def __init__(self, L, L_hat, Y_tr=None, index=None, model_path=None, Mat=None):
        self.Y_tr = Y_tr
        self.index = index
        self.clfs = []
        self.L_hat = L_hat
        self.L = L
        self.Mat=Mat
        if model_path != None:
            self.load_clf(model_path)
                    
    def load_clf(self, model_path):
        self.clfs = []
        for bit in range(self.L_hat): # load the binary classifiers
            self.clfs.append(joblib.load(os.path.join(model_path , 'label{}.pkl'.format(bit))))

    def predict_z(self, X):
        '''
        predict the subspace L_hat dim z vector with the data
        '''
        z_bits = []
        for clf in self.clfs:
            z_bit = clf.predict(X)
            z_bits.append(z_bit)
        return np.column_stack(z_bits)
    
    def vote_y(self, Z_pred, vote, weighted=True):
        dist, ind = self.index.search(Z_pred.astype('float32'), vote)
        if weighted: # 1/dist^2 as weight
            Y_pred = np.array([np.sum([self.Y_tr[ind[i][j]]/float(dist[i][j]*dist[i][j]+0.01) for j in range(len(ind[i]))], axis=0) for i in range(len(ind))])
        else:
            #Y_pred = np.array([np.sum([self.Y_tr[j[i]] for i in range(len(j))], axis=0) for j in ind])
            Y_pred = np.array([np.sum([self.Y_tr[ind[i][j]] for j in range(len(ind[i]))], axis=0) for i in range(len(ind))])
        return Y_pred
    
    def BIHT_y(self, Z_pred, sparsity, tau=0.5, iterate=50):
        '''
        Iterating to recover y from z.
        '''
        def threshold_k(a):
            '''keep top k element in a, leave zero other wise'''
            for i in range(a.shape[0]):
                topk = np.argpartition(a[i], -sparsity)[-sparsity:] #topk index
                temp = np.zeros(a.shape[1])
                temp[topk] = a[i][topk]
                a[i] = temp
            return a
        
        # create the Mat which is same everywhere with seed = 0
        if self.Mat == None:
            np.random.seed(0)
            self.Mat = normal(size=(self.L_hat, self.L))
        y = np.zeros((self.L,Z_pred.shape[0]))
        z_sign = np.apply_along_axis(lambda x: [-1 if elem <= 0 else 1 for elem in x], 0, Z_pred) # -1, 1 mapping
        #import pdb; pdb.set_trace()
        for _ in range(iterate):
            #time.tic()
            a = y + 0.5*tau*self.Mat.T.dot(z_sign.T-np.sign(self.Mat.dot(y)))
            #time.toc('multiply at {} iter'.format(_))
            y = threshold_k(a)
        return y.T # (sample, num of label)
        
            
    
    def predict_y(self, X, sparsity=1, vote=1, recover='kNN', weighted=True):
        '''
        predict y based on test data, can choose 'kNN', 'BIHT' method to recover
        Don't use default value!!!
        input:
            X: test data set
            sparsity: sparse estimate of labels, use in BIHT
            k: kNN's k nearest neighbor
            weighted: unweight or weight in kNN search, default is good here
        '''
        Z_pred = self.predict_z(X)
        if recover == 'kNN':
            y_pred = self.vote_y(Z_pred, vote, weighted=weighted)
        elif recover == 'BIHT':
            y_pred = self.BIHT_y(Z_pred, sparsity)
        else:
            y_pred = None
        return y_pred