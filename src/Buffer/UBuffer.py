import numpy as np
from code import getRank
from BlockPadder import blockPad
import pyopencl as cl

class UBuffer:
    def __init__(self, context):
        self.context = context
        
    def setU(self, U):
        if(len(U) != 3):
            raise Exception("Illegal shape.")
        
        self.R = np.int32(getRank(U))
        
        mf = cl.mem_flags
        
        U0 = blockPad(U[0], [16, 1])
        U1 = blockPad(U[1], [16, 1])
        U2 = blockPad(U[2], [16, 1])
        
        #Avoid garbage collection
        self.__U0 = U0
        self.__U1 = U1
        self.__U2 = U2
        print "ubuf"
        print self.context
        buf0 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__U0)
        buf1 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__U1)
        buf2 = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__U2)
        
        self.U = (buf0, buf1, buf2)
