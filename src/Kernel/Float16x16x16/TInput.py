from ..Kernel import Kernel
from BlockPadder import blockPad
from IProvider import IProvider

class TInput(Kernel, IProvider):
    
    T = None
    
    def initFromTInput(self, tInput):
        IProvider.initFromIProvider(tInput)
        self.T = TInput.T
    
    def init(self, T):
        T1 = blockPad(T, [16,16,16])
        self.T = self._createInitBuf(T1)
        
        IProvider.init(T1.shape)
        self.setBuffers()
        
    def setBuffers(self):
        self.kernel.set_arg(0, self.T)