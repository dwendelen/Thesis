import pyopencl as cl
import numpy as np
import numpy.linalg as la
from kr import kr

'''
PLATFORM
--------

INTERFACE
f(U)
init()
'''
class Platform:
    def __init__(self, T):
        self.T = T;

class NumPyPlatform (Platform):
    def __init__(self, T, M):
        super.__init__(T)
        self.M = M
    
    def init(self):
        return
    
    def f(self, U):
        D = self.M[0] - U[0].dot(kr(U[:0:-1]).T)
        fval = 0.5 * np.sum(D*D)
        return fval
    
class OpenCLPlatform (Platform):
    def __init__(self, T):
        super.__init__(T)
        
    def init(self):
        device = cl.get_platforms[0].get_devices(cl.device_type.GPU)[0]
        context = cl.Context(device)
        queue = cl.CommandQueue(context)
        
        prg = cl.Program(context, """
            __kernel void sum(__global const float *a,
            __global const float *b, __global float *c)
            {
              int gid = get_global_id(0);
              c[gid] = a[gid] + b[gid];
            }
            """).build()
            
        self.prg = prg
        self.queue = queue
    
    def f(self, U):
        


        a = numpy.random.rand(50000).astype(numpy.float32)
        b = numpy.random.rand(50000).astype(numpy.float32)
        
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
        
        self.prg.sum(self.queue, a.shape, None, a_buf, b_buf, dest_buf)
        
        a_plus_b = numpy.empty_like(a)
        cl.enqueue_copy(queue, a_plus_b, dest_buf)
        
        print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
        return 0;
    
    