
import math
import numpy as np

from ..Kernel import Kernel
from BlockPadder import blockPad


class FCommon(Kernel):
    
    R = None
    U = (None, None, None)
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
    
    def getI(self):
        raise NotImplementedError()
        
    def init(self, U):
        '''
        Precondition: self.I must be set
        '''
        self.R = np.int32(U[0].shape[1])
        
        U0 = blockPad(U[0], [16, 1])
        U1 = blockPad(U[1], [16, 1])
        U2 = blockPad(U[2], [16, 1])
        
        buf0 = self._createInitBuf(U0)
        buf1 = self._createInitBuf(U1)
        buf2 = self._createInitBuf(U2)
        self.U = (buf0, buf1, buf2)
        
        self.Sum = self._createReadWriteBuf(self.getNbWgs() * 4)
        
        self.__setBuffers()
    
    def initFromFCommon(self, kernel):
        self.R = kernel.R
        self.U = kernel.U
        self.Sum = kernel.Sum
        self.__setBuffers()    
        
    def __setBuffers(self):
        self.kernel.set_arg(1, self.U[0])
        self.kernel.set_arg(2, self.U[1])
        self.kernel.set_arg(3, self.U[2])
        self.kernel.set_arg(4, self.R)
        self.kernel.set_arg(5, self.IBuffer[0])
        self.kernel.set_arg(6, self.IBuffer[1])
        self.kernel.set_arg(7, self.IBuffer[2])
        self.kernel.set_arg(8, self.Sum)
    

