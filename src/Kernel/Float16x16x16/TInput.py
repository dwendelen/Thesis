from ..Kernel import Kernel

class TInput(Kernel):
    
    T = None
    
    def init(self, T):
        self.T = T
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(0, self.T)
