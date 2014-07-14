


from Kernel import Kernel


class FloatSingle3D(Kernel):
    
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    T = None
    
    def getLocalSize(self):
        return (64, 1, 1)
    
    def getGlobalSize(self):
        return ( self.I[0],
                 self.I[1],
                 self.I[2])
    
    def init(self, T, R, U, I, Sum):
        self.T = T
        self.R = R
        self.U = U
        self.I = I
        self.Sum = Sum
        self._setBuffers()
        
    def _setBuffers(self):
        self.kernel.set_arg(0, self.T)
        self.kernel.set_arg(1, self.U[0])
        self.kernel.set_arg(2, self.U[1])
        self.kernel.set_arg(3, self.U[2])
        self.kernel.set_arg(4, self.R)
        self.kernel.set_arg(5, self.Sum)
    
    def getName(self):
        return 'floatTSingle3D'

