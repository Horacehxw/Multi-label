from sklearn.externals import joblib
import os
import numpy as np
from numpy.random import normal
import util
from joblib import Parallel, delayed # Multitread

def _vote(Y_tr, indi, disti, weighted):
    if weighted:
        return np.sum([Y_tr[indij]/float(distij*distij+0.01) for indij, distij in zip(indi, disti)])
    else:
        return np.sum([Y_tr[indij] for indij in indi])

class BM_Predictor():
    '''
    Binary Map multilabel prediction model
    v1.0: update the BIHT prediction method
    v2.0: switch to scipy.sparse matrices
    
    Attribute:
        Mat: M, which maps y into z space
        L: dim of y
        L_hat: dim of z
        index(optional): kNN search index, faiss.index object
        Y_tr(optional): kNN recover index, training data
        model_path(optional): dir of classifiers
    '''
    def __init__(self, L, L_hat, Y_tr=None, index=None, model_path=None, Mat=None, num_core=-1):
        self.Y_tr = Y_tr
        self.index = index
        self.clfs = []
        self.L_hat = L_hat
        self.L = L
        self.Mat=Mat
        self.num_core = num_core
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
        Z_pred = []
        for clf in self.clfs:
            Z_pred.append(clf.predict(X))
        return np.column_stack(Z_pred)
        #return np.column_stack(Parallel(n_jobs=self.num_core)(delayed(util.predict_bit)(X, clf) for clf in self.clfs))
    

    
    def vote_y(self, Z_pred, vote, weighted=True):
        dist, ind = self.index.search(Z_pred.astype('float32'), vote)
        if weighted:
            return [np.sum([self.Y_tr[indij]/float(distij*distij+0.01) for indij, distij in zip(indi, disti)]) for indi, disti in zip(ind,dist)]
        else:
            return [np.sum([self.Y_tr[indij] for indij in indi]) for indi, disti in zip(ind,dist)]
        # issue: seems no speed up here, may be the bottle neck is swap io?
        #return Parallel(n_jobs=self.num_core)\
        #        (delayed(_vote)(self.Y_tr, indi, disti, True) for indi, disti in zip(ind, dist))
    
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
    
    def predict_prob_z(self, X):
        '''
        Use random forest to predict the probability of 1 for each label
        '''
        classifier = self.clfs[0] #load the RandomForest classifier
        z_pred_prob = classifier.predict_proba(X)
        return np.ascontiguousarray(np.array([prob[:, 1] for prob in z_pred_prob]).T) # kNN must use c-continuous array
        
            
    
    def predict_y(self, X, sparsity=1, vote=30, recover='kNN', classifier='OvsA', weighted=True, predict_prob=False):
        '''
        predict y based on test data, can choose 'kNN', 'BIHT' method to recover
        Don't use default value!!!
        input:
            X: test data set
            sparsity: sparse estimate of labels, use in BIHT
            k: kNN's k nearest neighbor
            weighted: unweight or weight in kNN search, default is good here
            classifier: the classifier to predict z, 'OvsA' or 'RandomForest'
            preditc_prob: use predict_prob in random forest instead of directly predict the classification result.
        '''
        
        if classifier == 'RandomForest':
            if predict_prob==True:
                Z_pred = self.predict_prob_z(X)
            else:
                Z_pred = self.predict_z(X)
        elif classifier == 'OvsA':
            Z_pred = self.predict_z(X)   
        else:
            Z_pred = None
        
        if recover == 'kNN':
            y_pred = self.vote_y(Z_pred, vote, weighted=weighted)
        elif recover == 'BIHT':
            y_pred = self.BIHT_y(Z_pred, sparsity)
        else:
            y_pred = None
        return y_pred