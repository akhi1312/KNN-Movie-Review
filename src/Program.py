import re 
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise






trainDf = pd.read_csv(
    filepath_or_buffer='./train.dat', 
    header=None, 
    sep='\t')

testDf = pd.read_csv(
    filepath_or_buffer='./test.dat', 
    header=None, 
    sep='\t')

trainvalues = trainDf.iloc[:,:].values

#print trainvalues

trainRatings = []
trainReviews = []

for value in trainvalues:
    trainRatings.append(value[0])
    trainReviews.append(re.sub('[^A-z -]', '', re.sub('<[^>]*>','',value[1])).lower())
    

# TestSet 

testvalues = testDf.iloc[:,:].values
testReviews = []

for value in testvalues:
    testReviews.append(re.sub('[^A-z -]', '', re.sub('<[^>]*>','',value[0])).lower())


# Training Set 




trainDocs = [l.split() for l in trainReviews]
testDocs = [l.split() for l in testReviews]

def filterLen(docs, minlen):
    return[ [t for t in d if len(t) >= minlen ] for d in docs ]

trainDocs1 = filterLen(trainDocs,5)
testDocs1 =  filterLen(testDocs ,5)

combinedList = trainDocs1 + testDocs1

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

combinedMatrix = build_matrix(combinedList)
combinedMatrix.shape
trainMatrix = combinedMatrix[0:25000]
testMatrix = combinedMatrix[25000:50000]



def cosine_similarity_n_space(m1, m2, batch_size=10):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break # cause I'm too lazy to elegantly handle edge cases
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
    return ret

cosineSimilarityValue = cosine_similarity_n_space(testMatrix,trainMatrix)



f = open('./format.dat', 'w')

for r in cosineSimilarityValue:

    k=89
    partitioned = np.argpartition(-r, k)  
    topIndex = partitioned[:k]
    
    ReviewTypeNegative = 0
    ReviewTypePositive = 0

    for index in topIndex:

        if trainvalues[index][0] == -1:
               ReviewTypeNegative+=1
        elif trainvalues[index][0] == 1:
               ReviewTypePositive+=1
            
    
    if ReviewTypeNegative > ReviewTypePositive:
        f.write('-1\n')
    else:
        f.write('+1\n')
        
    