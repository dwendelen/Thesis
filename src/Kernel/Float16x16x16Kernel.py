
import math
import pyopencl as cl

from Kernel import Kernel

from Buffer.IBuffer import IBuffer16x16x16Factory
from Buffer.TempBuffer import TempBuffer
from Buffer.TBuffer import TBuffer
from Buffer.UBuffer import UBuffer

class Float16x16x16Kernel(Kernel):
    
    def getNbWorkGroups(self, I, R, n):
        return pow(math.ceil(I/16.0), 3)
    
    def getBasicElements(self, I, R, n):
        return (I*I*I*R)
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getLocalSize()[0],
                 self.I[1]/self.getLocalSize()[1],
                 self.I[2]/self.getLocalSize()[2])
    
    def getLocalSize(self):
        return (4, 4, 4)
    
    def updatedUBuffer(self, UBuffer):       
        self.kernel.set_arg(4, UBuffer.R)
        
        for i in range(3):
            self.kernel.set_arg(i + 1, UBuffer.U[i])
    
    def setIBuffer(self, IBuffer):
        self.kernel.set_arg(5, IBuffer.IBuffer0)
        self.kernel.set_arg(6, IBuffer.IBuffer1)
        self.kernel.set_arg(7, IBuffer.IBuffer2)
        
    def setSumBuffer(self, sumBuffer):
        self.kernel.set_arg(9, sumBuffer.buffer)
    
class Float16x16x16UnmappedKernel(Float16x16x16Kernel):
    def updatedTBuffer(self, Tbuffer):
        """
        @type Tbuffer:TBuffer
        """ 
        self.kernel.set_arg(0, Tbuffer.T)


class Float16x16x16KernelFactory():
    def __init__(self, gcBlocker, contextQueue):
        '''
        @type gcBlocker: Buffer.GCBlocker.GCBlocker
        '''
        self.gcBlocker = gcBlocker
        self.contextQueue = contextQueue
        
    def initKernelAndCreateCommonBuffers(self, U, f):
        
        cq = self.contextQueue

        f.compile()
        f.init()
        
        ub = UBuffer(cq.context)
        ub.setU(U)
        f.setUBuffer(ub)

        sm = TempBuffer(cq.context)
        sm.init(f.getNbWGs())
        f.setSumBuffer(sm)
        
        ifac = IBuffer16x16x16Factory()
        ib = ifac.createFromU(U)
        f.setIBuffer(ib)
        
        #Avoid garbage collection
        self.gcBlocker.remember(ub)
        self.gcBlocker.remember(sm)
        self.gcBlocker.remember(ib)
    
class Float16x16x16UnmappedKernelFactory(Float16x16x16KernelFactory):
    def initKernelAndCreateCommonBuffers(self, U, T, f):
        Float16x16x16KernelFactory.initKernelAndCreateCommonBuffers(self, U, f)
        tb = TBuffer(self.contextQueue.context)
        tb.setT(T)
        f.setTBuffer(tb)
        
        #Avoid garbage collection
        self.gcBlocker.remember(tb)