import pyopencl as cl
import numpy as np

class NumpySum:
    
    array = None
    Sum = None
    queue = None
    
    def __init__(self, queue):
        self.queue = queue
        
    def init(self, Sum):
        self.sum = Sum
        self.array = np.zeros(Sum.size/np.float32.nbytes, dtype = np.float32)
        
    def getSum(self):
        cl.enqueue_copy(self.queue, self.array, self.Sum)
        return np.sum(self.array)