from ..Kernel import Kernel
from BlockPadder import blockPad
from IProvider import IProvider

class TInput(IProvider, Kernel):
    
    T = None
    
    def initT(self, T):
        self.T = T
        self.__setBuffers()
    
    def init(self, T):
        T1 = blockPad(T, [16,16,16])
        self.T = self._createInitBuf(T1)
        
        IProvider.init(self, T1.shape)
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(0, self.T)
