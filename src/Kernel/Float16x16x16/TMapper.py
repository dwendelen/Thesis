from TInput import TInput


class TMapper(TInput):

    TMapped = None
    I = None
    
    def getLocalSize(self):
        return (4,4,4)
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getLocalSize()[0],
                 self.I[1]/self.getLocalSize()[1],
                 self.I[2]/self.getLocalSize()[2])
    
    def init(self, T, tMapped, I):
        TInput.init(self, T)
        self.TMapped = tMapped
        self.I = I
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(1, self.TMapped)
        
    def getName(self):
        return 'float16x16x16Mapper'

