import pyopencl as cl
import numpy as np

from BlockPadder import blockPad

class AbstractBufferFactory: 
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
    
    def blockSize(self):
        raise NotImplementedError()
    
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
        
        T1 = blockPad(T, [self.blockSize(),self.blockSize(),self.blockSize()])
        self.T = self._createInitBuf(T1)
        
        self.I = T1.shape
        
        U0 = blockPad(U[0], [self.blockSize(), 1])
        U1 = blockPad(U[1], [self.blockSize(), 1])
        U2 = blockPad(U[2], [self.blockSize(), 1])
        
        buf0 = self._createInitBuf(U0)
        buf1 = self._createInitBuf(U1)
        buf2 = self._createInitBuf(U2)
        self.U = (buf0, buf1, buf2)
        
        self.R = np.int32(U[0].shape[1])
        
        self.SumArray = self._createReadWriteBuf(4*self.I[0]*self.I[1]*self.I[2]/(self.blockSize()*self.blockSize()*self.blockSize()))
        self.TMapped = self._createReadWriteBuf(self.T.size)
        self.Sum = self._createReadWriteBuf(4)

class Kernel:

    contextQueue = None
    kernel = None
    time = None
    
    def __init__(self, contextQueue):
        self.contextQueue = contextQueue
    
    def getName(self):
        raise NotImplementedError()
    
    def compile(self):
        file = open('../opencl/' + self.getName() + '.cl', 'r')
        self.program = cl.Program(self.contextQueue.context, file.read()).build()
        self.kernel = cl.Kernel(self.program, self.getName())
    
    def run(self):        
        e = cl.enqueue_nd_range_kernel(self.contextQueue.queue, self.kernel, self.getGlobalSize(), self.getLocalSize())
        
        if(self.contextQueue.profile):
            e.wait()
            self.time = (e.profile.end - e.profile.start)/ 1000000.0
    
    def getGlobalSize(self):
        raise NotImplementedError()
    
    def getLocalSize(self):
        raise NotImplementedError()
    
class BlockKernel(Kernel):
    R = None
    U = (None, None, None)
    I = (None, None, None)
    Sum = None
    T = None
    
    def getLocalSize(self):
        return (4, 4, 4)
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getNbFloatsPerWorkitem(),
                 self.I[1]/self.getNbFloatsPerWorkitem(),
                 self.I[2]/self.getNbFloatsPerWorkitem())
    
    def getNbFloatsPerWorkitem(self):
        raise NotImplementedError()
    
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
    
class AbstractTMapper(Kernel):
    TMapped = None
    I = None
    T = None
    
    def getLocalSize(self):
        return (4,4,4)
    
    def getName(self):
        raise NotImplementedError()
    
    def getNbFloatsPerWorkitem(self):
        raise NotImplementedError()
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getNbFloatsPerWorkitem(),
                 self.I[1]/self.getNbFloatsPerWorkitem(),
                 self.I[2]/self.getNbFloatsPerWorkitem())
    
    def init(self, T, tMapped, I):
        self.T = T
        self.TMapped = tMapped
        self.I = I
        self.__setBuffers()
        
    def __setBuffers(self):
        self.kernel.set_arg(0, self.T)
        self.kernel.set_arg(1, self.TMapped)
            
        