
import math
import numpy as np

from BlockPadder import blockPad
from Kernel.BlockKernel import Block3DKernel


class FCommon(Block3DKernel):
    
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    
    def getNbWorkGroups(self, I, R, n):
        return pow(math.ceil(I/16.0), 3)
    
    def getBasicElements(self, I, R, n):
        return (I*I*I*R)
    
    
    
    def getLocalSize(self):
        return (4, 4, 4)
    
    def getNbWgs(self):
        return (self.I[0]*self.I[1]*self.I[2])/(16*16*16)
    
    def init(self, R, U, I, Sum):
        self.R = R
        self.U = U
        self.I = I
        self.Sum = Sum
        self._setBuffers()
        
    def _setBuffers(self):
        self.kernel.set_arg(1, self.U[0])
        self.kernel.set_arg(2, self.U[1])
        self.kernel.set_arg(3, self.U[2])
        self.kernel.set_arg(4, self.R)
        self.kernel.set_arg(5, self.Sum)
    

