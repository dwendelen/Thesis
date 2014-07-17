from Kernel.Kernel import Kernel


class TMapper(Kernel):

    TMapped = None
    I = None
    T = None
    
    def getLocalSize(self):
        return (4,4,4)
    
    def getGlobalSize(self):
        return ( self.I[0]/2,
                 self.I[1]/2,
                 self.I[2]/2)
    
    def init(self, T, tMapped, I):
        self.T = T
        self.TMapped = tMapped
        self.I = I
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(0, self.T)
        self.kernel.set_arg(1, self.TMapped)
        
    def getName(self):
        return 'float8x8x8Mapper'

