from Float16x16x16Kernel import Float16x16x16Kernel

import math
from BlockPadder import blockPad
import numpy as np
import pyopencl as cl
from Platform.ContextQueue import ContextQueue
from Buffer.TBuffer import TBuffer
from Buffer.UBuffer import UBuffer
from Buffer.SumBuffer import SumBuffer


class Float16x16x16(Float16x16x16Kernel):
    def getNbOperaties(self, I, R, n):
        return self.getNbWorkGroups(I, R, n) * (9216*R + 12287)
        
    def getDataTransferZonderCache(self, I, R, n):
        return 4*self.getNbWorkGroups(I, R, n) * (16*16*16 + 3 * 16 * R)
        
    def getDataTransferMetCache(self, I, R, n):
        i = math.ceil(I/16.0) * 16
        return (i*i*i + 3*i*R) * 4
    
    def getName(self):
        return 'float16x16x16'

    def getNbWGs(self):
        return (self.I[0]*self.I[1]*self.I[2])/(16*16*16)
    
class Factory():
    def create(self, U, T):
        cq = ContextQueue()
        f = Float16x16x16(cq)
        
        cq.init()
        f.compile()
        f.init()
        
        tb = TBuffer(cq.context)
        tb.setT(T)
        f.setTBuffer(tb)
        
        ub = UBuffer(cq.context)
        ub.setU(U)
        f.setUBuffer(ub)
        
        sm = SumBuffer(cq.context)
        sm.init(f.getNbWGs())
        f.setSumBuffer(sm)
        
        return f
