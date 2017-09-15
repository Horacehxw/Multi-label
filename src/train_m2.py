%matplotlib inline
import math
import os
import data_util
#from data_util import DataPoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer # convert y to {0,1}^L
from sklearn.preprocessing import StandardScaler # normalize features 
from sklearn.feature_extraction import DictVectorizer # extract feature vector to x
from numpy.random import normal # generate transforming matrix
from sklearn.neighbors import KDTree #KDTree for fast kNN search
from sklearn.externals import joblib # store classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed # Multitread

# setting path
data_dir = "../data"
model_dir = "../.model2"
train_filename = "/Eurlex/eurlex_train.txt"
test_filename = "/Eurlex/eurlex_test.txt"
#tr_split_file = "/Delicious/delicious_trSplit.txt"
#te_split_file = "/Delicious/delicious_tstSplit.txt"

path = os.path.dirname(train_filename)
model_path = model_dir + path

# data prepossesing
tr_data, num_point, num_feature, num_label = data_util.read_file(data_dir+train_filename)
te_data, _, _, _ = data_util.read_file(data_dir+test_filename)
X_tr, Y_tr, X_te, Y_te = data_util.data_transform(tr_data, te_data, num_label)

# normalize features
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)
X_tr.shape
# sparsity
k = sorted([Y.sum() for Y in Y_tr], reverse=True)[int(num_point*0.001)]

# constructing model
np.random.seed(0)
L_hat = k * math.log(Y_tr.shape[1], 2)
M = normal(size=(int(math.ceil(L_hat)), Y_tr.shape[1]))
Z_tr = M.dot(Y_tr.T).T # z = n*\hat L
Z_tr = np.apply_along_axis(lambda x: [0 if elem < 0 else 1 for elem in x], 0, Z_tr) # sign

# traing function
def train_bit(bit):
    print "Trianning model for the {}th bit\n... ... ... \n".format(bit)
    #clf = LogisticRegression(solver='sag')
    clf = LinearSVC(dual=False)
    clf.fit(y=Z_tr[:, bit], X=X_tr)
    joblib.dump(clf, os.path.join(model_path , 'label{}.pkl'.format(bit)))
    print "{}th bit's model successfully stored in {}/label{}.pkl\n".format(bit, model_path, bit)

Parallel(n_jobs=8)(delayed(train_bit)(i) for i in range(Z_tr.shape[1]))

print "all done! success!\n"