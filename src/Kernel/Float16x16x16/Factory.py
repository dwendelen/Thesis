from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory
from Kernel.Float16x16x16.F import F
from Kernel.NumpySum import NumpySum
from Kernel.Float16x16x16.FRemapped import Float16x16x16Remapped
from Kernel.Float16x16x16.FRemapped2 import Float16x16x16Remapped2
from Kernel.Float16x16x16.TMapper import TMapper
from Kernel.Sum.Sum16 import Sum16

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
