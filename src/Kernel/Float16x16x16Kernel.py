from Kernel import Kernel
import math
from BlockPadder import blockPad
import numpy as np
import pyopencl as cl


class Float16x16x16Kernel(Kernel):
    
    def getNbWorkGroups(self, I, R, n):
        return pow(math.ceil(I/16.0), 3)
    
    def getBasicElements(self, I, R, n):
        return (I*I*I*R)
    
    def run(self):
        e = cl.enqueue_nd_range_kernel(self.contextQueue.queue, self.kernel, self.I, (4,4,4))
        self.time = (e.profile.end - e.profile.start)/ 1000000.0
    
    def setUBuffer(self, UBuffer):       
        self.kernel.set_arg(5, UBuffer.R)

        for i in range(3):
            self.kernel.set_arg(i + 1, UBuffer.U[i]) 
      
    def setTBuffer(self, Tbuffer):
        """
        @type Tbuffer:TBuffer
        """ 
        self.kernel.set_arg(0, Tbuffer.T)
        self.I = Tbuffer.I

        for i in range(3):
            self.kernel.set_arg(6 + i, Tbuffer.I[i]) 
            
    def setSumBuffer(self, sumBuffer):
        self.kernel.set_arg(9, sumBuffer.sum)
    
    def init(self):
        self.time = np.inf
        l_buf = cl.LocalMemory(64*2)
        self.kernel.set_arg(4, l_buf)
        
        sum_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=4)
        