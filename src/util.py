import numpy as np

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