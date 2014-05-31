from TInput import TInput


class TMapper(TInput):

    TMapped = None
    
    def init(self, T, tMapped):
        TInput.init(self, T)
        self.TMapped = tMapped
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(1, self.TMapped)

