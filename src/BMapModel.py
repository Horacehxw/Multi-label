from sklearn.externals import joblib # store classifiers
import os
import numpy as np

class BM_Predictor():
    '''
    v1.0: update the BIHT prediction method
    Binary Map multilabel prediction model
    predict_z: output the predicted Z_pred value, where is the projected subspace
    output: the vote on Y space, higher value represents higher weight
    '''
    def __init__(self, length, Mat=None, Y_tr=None, index=None, model_path=None):
        self.Y_tr = Y_tr
        self.Mat=Mat
        self.index = index
        self.clfs = []
        self.length = length
        if model_path != None:
            self.load_clf(model_path)
                    
    def load_clf(self, model_path):
        self.clfs = []
        for bit in range(self.length): # load the binary classifiers
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
    
    def vote_y(self, Z_pred, k, weighted=True):
        dist, ind = self.index.search(Z_pred.astype('float32'), k)
        if weighted: # 1/dist^2 as weight
            Y_pred = np.array([np.sum([self.Y_tr[ind[i][j]]/float(dist[i][j]*dist[i][j]+0.01) for j in range(len(ind[i]))], axis=0) for i in range(len(ind))])
        else:
            #Y_pred = np.array([np.sum([self.Y_tr[j[i]] for i in range(len(j))], axis=0) for j in ind])
            Y_pred = np.array([np.sum([self.Y_tr[ind[i][j]] for j in range(len(ind[i]))], axis=0) for i in range(len(ind))])
        return Y_pred
    
    def BIHT_y(self, Z_pred, tau=0.5, iterate=50):
        pass # use BIHT recovery
    
    def predict_y(self, X, k=10, recover='kNN', weighted=True):
        Z_pred = self.predict_z(X)
        if recover == 'kNN':
            y_pred = self.vote_y(Z_pred, k, weighted=weighted)
        elif recover == 'BIHT':
            y_pred = BIHT_y()
        else:
            y_pred = None
        return y_pred
    