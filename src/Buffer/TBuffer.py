import pyopencl as cl
from BlockPadder import blockPad
import numpy as np

class TBuffer:
    def a(self):
        pass
    def setT(self, T):
        if(len(T.shape) != 3):
            raise Exception("Illegal shape.")
        
        T1 = blockPad(T, [16,16,16])
        self.I = [np.int32(T1.shape[0]),
                  np.int32(T1.shape[1]),
                  np.int32(T1.shape[0])]
        
        mf = cl.mem_flags
        self.T = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T1)