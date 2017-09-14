from sklearn.preprocessing import MultiLabelBinarizer # convert y to {0,1}^L
from sklearn.feature_extraction import DictVectorizer # extract feature vector to x
import numpy as np

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

def read_a_point(line, counter = None, prominent = None):
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
            data.append(read_a_point(line, counter=feature_counter))
    return data, num_point, num_features, num_labels

def data_transform(tr, te, num_label):
    '''
    return: X_tr, Y_tr, X_te, Y_te
    '''
    # transform train and test data into sparse matrix
    lb = MultiLabelBinarizer(classes=range(num_label))
    Y_tr_raw = np.array([np.array(data_point.labels) for data_point in tr])
    Y_te_raw = np.array([np.array(data_point.labels) for data_point in te])
    lb.fit(Y_tr_raw)
    Y_tr = lb.transform(Y_tr_raw)
    Y_te = lb.transform(Y_te_raw)

    fv = DictVectorizer(sparse=False)
    X_tr_raw = [data_point.features for data_point in tr]
    X_te_raw = [data_point.features for data_point in te]
    fv.fit(X_tr_raw)
    X_tr = fv.transform(X_tr_raw)
    X_te = fv.transform(X_te_raw)
    return X_tr, Y_tr, X_te, Y_te