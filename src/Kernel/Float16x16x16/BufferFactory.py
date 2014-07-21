import numpy as np
import pyopencl as cl

from BlockPadder import blockPad

class BufferFactory:
    
    contextQueue = None
    T = None
    R = None
    U = (None, None, None)
    I = (None, None, None)
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
    
    def release(self):
        if(self.T != None):
            self.T.release()
        
        if(self.TMapped != None):
            self.TMapped.release()
        
        if(self.U[0] != None):
            self.U[0].release()
            
        if(self.U[1] != None):
            self.U[1].release()
            
        if(self.U[2] != None):
            self.U[2].release()
            
        if(self.SumArray != None):
            self.SumArray.release()
            
        if(self.Sum != None):
            self.Sum.release()
    
        self.T = None
        self.R = None
        self.U = (None, None, None)
        self.I = (None, None, None)
        self.SumArray = None
        self.TMapped = None
        self.Sum = None
    
    def init(self, T, U):
        self.release()
        
        T1 = blockPad(T, [16,16,16])
        self.T = self._createInitBuf(T1)
        
        self.I = T1.shape
        
        U0 = blockPad(U[0], [16, 1])
        U1 = blockPad(U[1], [16, 1])
        U2 = blockPad(U[2], [16, 1])
        
        buf0 = self._createInitBuf(U0)
        buf1 = self._createInitBuf(U1)
        buf2 = self._createInitBuf(U2)
        self.U = (buf0, buf1, buf2)
        
        self.R = np.int32(U[0].shape[1])
        
        self.SumArray = self._createReadWriteBuf(4*self.I[0]*self.I[1]*self.I[2]/(16*16*16))
        self.TMapped = self._createReadWriteBuf(self.T.size)
        self.Sum = self._createReadWriteBuf(4)
        

