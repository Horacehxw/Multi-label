from sklearn.preprocessing import MultiLabelBinarizer # convert y to {0,1}^L
from sklearn.feature_extraction import DictVectorizer # extract feature vector to x
import numpy as np
from sklearn.preprocessing import StandardScaler # normalize features 

class DataPoint():
    def __init__(self):
        self.labels = []
        self.features = {}

    def create_label(self, classes):
        self.labels = classes

    def add_feature(self, key, val):
        self.features[key] = val

    def __str__(self):
        string = ""
        for label in self.labels[:-1]:
            string += str(label)
            string += ","
        string += str(self.labels[-1]) + " "
        for feature in self.features:
            string += str(feature)+":"+str(self.features[feature])
            string += " "
        string += "\n"
        return string

def _read_a_point(line, counter = None, prominent = None):
    def read_feature(__tokens): # data closure
        for token in __tokens:
            #print(token)
            pair = token.split(":")
            try:
                feature, value = int(pair[0]), float(pair[1])
            except ValueError:
                print("Error when adding features, Invalid Data Point!\n")
                return
            except IndexError:
                print("can't split this pair\n")
                print(token)
                print(__tokens)
                return
            if counter is not None:
                counter[feature] = counter.get(feature, 0) + 1
            if prominent is None:
                d_point.add_feature(feature, value)
            elif feature in prominent:
                d_point.add_feature(feature, value)
        return d_point

    d_point = DataPoint()
    tokens = line.split() #default by space
    #print (tokens[0])
    classes = tokens[0].split(",")
    try:
        classes = [int(x) for x in classes]
    except ValueError:
        #print ("No labels in the Data Point!\n")
        return read_feature(tokens)
    d_point.create_label(classes)
    return read_feature(tokens[1:])

def read_file(filename):
    '''
    return: data, num_point, num_features, num_labels
        data: list of DataPoint read in the files
    '''
    data = []
    num_point, num_features, num_labels = 0,0,0
    with open(filename, "r") as file:
        first_line = file.readline().split()
        first_line = [int(x) for x in first_line]
        num_point, num_features, num_labels = first_line[0], first_line[1], first_line[2]
        d_points = []
        feature_counter = {}
        for line in file:
            data.append(_read_a_point(line, counter=feature_counter))
    return data, num_point, num_features, num_labels

def data_transform(tr, te, num_label):
    '''
    return: X_tr, Y_tr, X_te, Y_te
    X is sparse matrix for memory reserve.
    '''
    # transform train and test data into sparse matrix
    lb = MultiLabelBinarizer(classes=range(num_label), sparse_output=True)
    Y_tr = np.array([np.array(data_point.labels) for data_point in tr])
    Y_te = np.array([np.array(data_point.labels) for data_point in te])
    lb.fit(Y_tr)
    Y_tr = lb.transform(Y_tr) #scipy sparse matrix
    Y_te = lb.transform(Y_te)
    lb = None

    fv = DictVectorizer(sparse=True)
    X_tr = [data_point.features for data_point in tr] #scipy sparse matrix
    X_te = [data_point.features for data_point in te]
    fv.fit(X_tr)
    X_tr = fv.transform(X_tr)
    X_te = fv.transform(X_te)
    
    scaler = StandardScaler(with_mean=False) # X is sparse matrix, maintain sparsity by not centering the data
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, Y_tr, X_te, Y_te

def split_data(data, split_file, index=0):
    '''
    input: 
        data: list of DataPoint
        split_file: split file path, each column indicate a split
        index: which split column to choose, default 0
    return:
        the split result, list of DataPoint
    '''
    result = []
    with open(split_file, "r") as file:
        for line in file:
            #result.extend([data[int(x)-1] for x in line.split()])
            # each column in the split file is a split
            result.append(data[int(line.split()[index]) - 1])
    return result