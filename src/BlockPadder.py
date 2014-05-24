from __future__ import division
from math import ceil

import numpy as np

def blockPad(array, blockSizes):
    def getNewSize(i, blockSize):
        return int(ceil(i/blockSize))*blockSize
    
    slices = list()
    dim = list()
    for i in range(len(array.shape)):
        s = getNewSize(array.shape[i], blockSizes[i])
        dim.append(s)
        slices.append(slice(0, array.shape[i], 1))
        
    A = np.zeros(dim, dtype = array.dtype, order='F')
    A[slices] = array
    return A