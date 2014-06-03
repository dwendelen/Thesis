
import math
import numpy as np

from ..Kernel import Kernel
from BlockPadder import blockPad


class FCommon(Kernel):
    
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    
    def getNbWorkGroups(self, I, R, n):
        return pow(math.ceil(I/16.0), 3)
    
    def getBasicElements(self, I, R, n):
        return (I*I*I*R)
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getLocalSize()[0],
                 self.I[1]/self.getLocalSize()[1],
                 self.I[2]/self.getLocalSize()[2])
    
    def getLocalSize(self):
        return (4, 4, 4)
    
    def getNbWgs(self):
        return (self.I[0]*self.I[1]*self.I[2])/(16*16*16)
    
    def init(self, R, U, I, Sum):
        self.R = R
        self.U = U
        self.I = I
        self.Sum = Sum
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(1, self.U[0])
        self.kernel.set_arg(2, self.U[1])
        self.kernel.set_arg(3, self.U[2])
        self.kernel.set_arg(4, self.R)
        self.kernel.set_arg(5, self.Sum)
    

