from Common import BlockKernel, AbstractBufferFactory, AbstractTMapper
from Platform.ContextQueue import ContextQueue
from Kernel.NumpySum import NumpySum
from Kernel.Sum.Sum16 import Sum16

class BufferFactory(AbstractBufferFactory):
    def blockSize(self):
        return 16
        
class FCommon(BlockKernel):
    def getNbFloatsPerWorkitem(self):
        return 4

class Float16x16x16E(FCommon):
    def getName(self):
        return 'float16x16x16E'

class F(FCommon):
    def getName(self):
        return 'float16x16x16'

class Float16x16x16Remapped(FCommon):
    def getName(self):
        return 'float16x16x16R'

class Float16x16x16Remapped2(FCommon):
    def getName(self):
        return 'float16x16x16R2'

class TMapper(AbstractTMapper):
    
    def getNbFloatsPerWorkitem(self):
            return 4
        
    def getName(self):
        return 'float16x16x16Mapper'
        
class Factory:
    def init(self):
        self.cq = ContextQueue()
        self.cq.init()
        
        self.b = BufferFactory(self.cq)
        
        self.summer = NumpySum(self.cq.queue)
        
        
    def setTU(self, T, U):
        self.b.init(T, U)
        self.summer.init(self.b.SumArray)
        
    def createF(self):
        f = F(self.cq)
        f.compile()
        f.init(self.b.T, self.b.R, self.b.U, self.b.I, self.b.SumArray)
        return f
    
    def createRemapper(self):
        r = TMapper(self.cq)
        r.compile()
        r.init(self.b.T, self.b.TMapped, self.b.I)
        return r
    
    def createR(self):
        r = Float16x16x16Remapped(self.cq)
        r.compile()
        r.init(self.b.TMapped, self.b.R, self.b.U, self.b.I, self.b.SumArray)
        return r
        
    def createR2(self):
        r = Float16x16x16Remapped2(self.cq)
        r.compile()
        r.init(self.b.TMapped, self.b.R, self.b.U, self.b.I, self.b.SumArray)
        return r
    
    def createSum16(self):
        s = Sum16(self.cq)
        s.compile()
        s.init(self.b.SumArray, self.b.Sum)
        return s
        
    def getF(self):
        return self.summer.getF()

