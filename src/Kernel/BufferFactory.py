import numpy as np
import pyopencl as cl

from BlockPadder import blockPad

class BufferFactory:
    
    contextQueue = None
    T = None
    R = None
    U = (None, None, None)
    I = None
    I0 = None
    I1 = None
    I2 = None
    SumArray = None
    TMapped = None
    Sum = None
    
    def __init__(self, contextQueue):
        self.contextQueue = contextQueue
    
    def _createInitBuf(self, array):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
    
    def _createReadWriteBuf(self, nbBytes):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_WRITE, size=nbBytes)
    
    def init(self, T, U):
        self.I0 = np.int32(T.shape[0])
        self.I1 = np.int32(T.shape[1])
        self.I2 = np.int32(T.shape[2])
    
        T1 = blockPad(T.flatten(order='F'), [64])
        self.T = self._createInitBuf(T1)
        
        self.I = T1.shape[0]
        
        U0 = blockPad(U[0], [64, 1])
        U1 = blockPad(U[1], [1, 1])
        U2 = blockPad(U[2], [1, 1])
        
        buf0 = self._createInitBuf(U0)
        buf1 = self._createInitBuf(U1)
        buf2 = self._createInitBuf(U2)
        self.U = (buf0, buf1, buf2)
        
        self.R = np.int32(U[0].shape[1])
        
        self.SumArray = self._createReadWriteBuf((4*self.I)/(4*4*4))
        #self.TMapped = self._createReadWriteBuf(self.T.size)
        self.Sum = self._createReadWriteBuf(4)
        

