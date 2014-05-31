from TInput import TInput


class TMapper(TInput):

    TMapped = None

    def init(self, T):
        TInput.init(self, T)
        self.TMapped = self._createReadWriteBuf(self.T.size)
        
        self.setBuffers()
    
    def initFromTMapper(self, tMapper):
        TInput.initFromTInput(self, tMapper)
        self.TMapped = tMapper.TMapped
        
    def setBuffers(self):
        self.kernel.set_arg(1, self.TMapped)
        self.kernel.set_arg(2, self.I[0])
        self.kernel.set_arg(3, self.I[1])
        self.kernel.set_arg(4, self.I[2])
