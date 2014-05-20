import pyopencl as cl
import numpy as np
import numpy.linalg as la
from kr import kr
from code import *
from math import *

from Platform import Platform

class OpenCLPlatform (Platform):        
    def init(self):
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        context = cl.Context([devices[0]])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        file = open('../opencl/16x16x16float.cl', 'r')
        
        prg = cl.Program(context, file.read()).build()
            
        self.prg = prg
        self.queue = queue
        self.context = context
    
    def globalSize(self, (i1, i2, i3)):
        return (int(ceil(i1/16.0))*4, int(ceil(i2/16.0))*4, int(ceil(i3/16.0))*4)

    def getTShape(self, (i1, i2, i3)):
        return (int(ceil(i1/16.0))*16, int(ceil(i2/16.0))*16, int(ceil(i3/16.0))*16)

    def createU(self, U):
        elements = int(ceil(U.shape[0]/16.0))*16
        r = np.zeros((elements, U.shape[1]), order='F', dtype=np.float32)
        r[:U.shape[0],:] = U
        return r

    def f(self):
        U0 = self.createU(self.U[0])
        U1 = self.createU(self.U[1])
        U2 = self.createU(self.U[2])
        
        g = self.globalSize(self.T.shape)
        gs = self.getTShape(self.T.shape)

        T = np.zeros(gs, order='F', dtype=np.float32)      

        T[:self.T.shape[0], :self.T.shape[1], :self.T.shape[2]] = self.T        

        mf = cl.mem_flags
        T_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
        U0_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U0)
        U1_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U1)
        U2_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=U2)
        l_buf = cl.LocalMemory(64*4)
        sum_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=4)

        kernel = self.prg.float16x16x16

        kernel.set_arg(0, T_buf)
        kernel.set_arg(1, U0_buf)
        kernel.set_arg(2, U1_buf)
        kernel.set_arg(3, U2_buf)
        kernel.set_arg(4, l_buf)
        kernel.set_arg(5, np.int32(self.R))
        kernel.set_arg(6, np.int32(g[0]))
        kernel.set_arg(7, np.int32(g[1]))
        kernel.set_arg(8, np.int32(g[2]))
        kernel.set_arg(9, sum_buf)
        
        e = cl.enqueue_nd_range_kernel(self.queue, kernel, g, (4,4,4))
        
        s = np.zeros((1), dtype = np.float32)
        cl.enqueue_copy(self.queue, s, sum_buf)
        
        print str((e.profile.end - e.profile.start)/ 1000000.0)
        
        return s[0]/2
        
    def g(self):
        return []
