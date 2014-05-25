import pyopencl as cl
from BlockPadder import blockPad
import numpy as np

class TBuffer:

    def __init__(self, context):
        self.context = context
        
    def setT(self, T):
        if(len(T.shape) != 3):
            raise Exception("Illegal shape.")
        
        T1 = blockPad(T, [16,16,16])
        
        #Try to avoid garbage collection
        self.Ibuffers = [np.int32(T1.shape[0]/4),
                         np.int32(T1.shape[1]/4),
                         np.int32(T1.shape[2]/4)]
                         
        
        
        self.I = T1.shape
        
        mf = cl.mem_flags

        self.T = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T1)
