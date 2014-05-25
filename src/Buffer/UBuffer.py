import numpy as np
from code import getRank
from BlockPadder import blockPad
import pyopencl as cl
from Buffer import Buffer

class UBuffer(Buffer):
        
    def setU(self, U):
        if(len(U) != 3):
            raise Exception("Illegal shape.")
        
        self.R = np.int32(getRank(U))
        
        mf = cl.mem_flags
        
        U0 = blockPad(U[0], [16, 1])
        U1 = blockPad(U[1], [16, 1])
        U2 = blockPad(U[2], [16, 1])
        
        buf0 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U0)
        buf1 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U1)
        buf2 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U2)
        
        self.U = (buf0, buf1, buf2)
        
        for kernel in self.kernels:
            kernel.setUBuffer(self)
