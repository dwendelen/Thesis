import numpy as np
import pyopencl as cl

from Kernel import Kernel
from BlockPadder import getNewSize

class Sum16(Kernel):
    size = None
    n = None
    Array = None
    Sum = None
    localSum = None
    
    def getNbOperaties(self, I, R, n):
        return 0
    
    def getNbWorkGroups(self, I, R, n):
        return 0
        
    def getDataTransferZonderCache(self, I, R, n):
        return 0
        
    def getDataTransferMetCache(self, I, R, n):
        return 0
    
    def getBasicElements(self, I, R, n):
        return 0
    
    def getName(self):
        return 'Sum16'
    
    def init(self, Array, Sum):
        self.Sum = Sum
        self.Array = Array
        
        self.n = Array.size/np.dtype(np.float32).itemsize
        self.size = getNewSize(self.n, 16)
        self.localSum = np.zeros(1, dtype = np.float32)
        
    def getGlobalSize(self):
        return self.size
    
    def getLocalSize(self):
        return (16,)
    
    def getSum(self):
        cl.enqueue_copy(self.contextQueue.queue, self.localSum, self.Sum)
        return self.localSum[0]