from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory
from Kernel.Float16x16x16.F import F
from Kernel.NumpySum import NumpySum
from Kernel.Float16x16x16.FRemapped import Float16x16x16Remapped
from Kernel.Float16x16x16.TMapper import TMapper

class Factory:
    def init(self):
        self.cq = ContextQueue()
        self.cq.init()
        
        self.b = BufferFactory(self.cq)
        
        self.summer = NumpySum(self.cq.queue)
        
        
    def setTU(self, T, U):
        self.b.init(T, U)
        self.summer.init(self.b.Sum)
        
    def createF(self):
        f = F(self.cq)
        f.compile()
        f.init(self.b.T, self.b.R, self.b.U, self.b.I, self.b.Sum)
        return f
    
    def createRemapper(self):
        r = TMapper(self.cq)
        r.compile()
        r.init(self.b.T, self.b.TMapped, self.b.I)
        return r
    
    def createR(self):
        r = Float16x16x16Remapped(self.cq)
        r.compile()
        r.init(self.b.TMapped, self.b.R, self.b.U, self.b.I, self.b.Sum)
        return r
        
    def getF(self):
        return self.summer.getF()
