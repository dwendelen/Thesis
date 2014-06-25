import numpy as np

from Kernel.Float16x16x16.F import F
from Kernel.Float16x16x16.E import Float16x16x16E as E
from Kernel.Float16x16x16.FRemapped import Float16x16x16Remapped
from Kernel.Float16x16x16.FRemapped2 import Float16x16x16Remapped2
from Kernel.Float16x16x16.TMapper import TMapper

from simulators import simulateKernel
from Platform.ContextQueue import ContextQueue
from Platform.Platform import NumPyPlatform
from Kernel.Float16x16x16.BufferFactory import BufferFactory

import cProfile
from Kernel.Sum.Sum16 import Sum16

cq = ContextQueue(profile = True)
cq.init()

n = 64000001;
a = np.array(np.random.rand(64000001), dtype=np.float32)

mf = cl.mem_flags
s = cl.Buffer(cq.context, mf.READ_WRITE, size=4)
sa = cl.Buffer(cq.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

k = Sum16(cq)
k.compile()
k.init(sa, s)
k.run()

r = k.getSum()
e = np.sum(a)

print r
print e
print k.time
    
