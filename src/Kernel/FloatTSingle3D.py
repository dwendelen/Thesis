import pyopencl as cl
import numpy as np

from Common import Kernel
from BlockPadder import blockPad

class FloatTSingle3D(Kernel):
    
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    T = None
    
    def getLocalSize(self):
        return (64, 1, 1)
    
    def getGlobalSize(self):
        return ( self.I,1,1)
    
    def init(self, T, R, U, I, Sum, I0, I1, I2):
        self.T = T
        self.R = R
        self.U = U
        self.I = I
        self.I0 = I0
        self.I1 = I1
        self.I2 = I2
        self.Sum = Sum
        self._setBuffers()
        
    def _setBuffers(self):
        self.kernel.set_arg(0, self.T)
        self.kernel.set_arg(1, self.U[0])
        self.kernel.set_arg(2, self.U[1])
        self.kernel.set_arg(3, self.U[2])
        self.kernel.set_arg(4, self.R)
        self.kernel.set_arg(5, self.Sum)
        self.kernel.set_arg(6, self.I0)
        self.kernel.set_arg(7, self.I1)
        self.kernel.set_arg(8, self.I2)
    
    def getName(self):
        return 'floatTSingle3D'

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
    
        self.T = None
        self.R = None
        self.U = (None, None, None)
        self.I = (None, None, None)
        self.SumArray = None
        self.TMapped = None
        self.Sum = None
    
    def init(self, T, U):
        self.release()
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
        
