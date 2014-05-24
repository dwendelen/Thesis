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
        
        self.U = list()
        mf = cl.mem_flags
        for i in range(3):
            Ui = blockPad(U[i], [16, 1])
            Uibuf = (cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Ui))
            self.U.append(Uibuf)
