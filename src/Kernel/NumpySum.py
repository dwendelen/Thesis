import pyopencl as cl
import numpy as np

class NumpySum:
    
    array = None
    Sum = None
    queue = None
    
    def __init__(self, queue):
        self.queue = queue
        
    def init(self, Sum):
        self.Sum = Sum
        self.array = np.zeros((Sum.size/np.dtype(np.float32).itemsize), dtype = np.float32)
        
    def getF(self):
        cl.enqueue_copy(self.queue, self.array, self.Sum)
        return np.sum(self.array)/2
