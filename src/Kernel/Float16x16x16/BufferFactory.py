class BufferFactory:
    
    T = None
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    
    def createInitBuf(self, array):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
    
    def createReadWriteBuf(self, nbBytes):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_WRITE, size=nbBytes)

    def init(self, T, U):
        T1 = blockPad(T, [16,16,16])
        self.T = self._createInitBuf(T1)
        
        self.I = T1.shape
        
        U0 = blockPad(U[0], [16, 1])
        U1 = blockPad(U[1], [16, 1])
        U2 = blockPad(U[2], [16, 1])
        
        buf0 = self.createInitBuf(U0)
        buf1 = self.createInitBuf(U1)
        buf2 = self.createInitBuf(U2)
        self.U = (buf0, buf1, buf2)
        
        self.R = np.int32(U[0].shape[1])
        
        self.Sum = self.createReadWriteBuf(4*I[0]*I[1]*I[2]/(16*16*16))
        
class MappedBufferFactory(BufferFactory):

    TMapped = None
    
    def init(self, T, U):
        BufferFactory(self, T, U)
        self.TMapped = self.createReadWriteBuf(self.T.size)
        
        

