import numpy as np
import pyopencl as cl

from simulators import simulateKernel
from Platform.ContextQueue import ContextQueue
from Platform.Platform import NumPyPlatform
from Kernel.Float16x16x16.BufferFactory import BufferFactory

import cProfile
from Kernel.Sum.Sum16 import Sum16

cq = ContextQueue(profile = True)
cq.init()

n = 64000005;
a = np.array(np.random.rand(n), dtype=np.float32)

mf = cl.mem_flags
s = cl.Buffer(cq.context, mf.READ_WRITE, size=4*32768)
sa = cl.Buffer(cq.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

k = Sum16(cq)
k.compile()
k.init(sa, s)

k.run()
k.run()
k.run()

r = k.getSum()
e = np.sum(a[:16000000]) + np.sum(a[16000000:32000000]) \
    + np.sum(a[32000000:48000000]) + np.sum(a[48000000:])

print r
print e
print k.time
    
