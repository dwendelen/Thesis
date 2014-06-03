import numpy as np

from Kernel.Float16x16x16.F import F
from Kernel.Float16x16x16.FRemapped import Float16x16x16Remapped

from simulators import simulateKernel
from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory

R = 4
I = 360

T = np.array(np.random.rand(I, I, I), dtype=np.float32)
U = np.array(np.random.rand(I, R), dtype=np.float32)

cq = ContextQueue(profile = True)

b = BufferFactory()
b.init(T, U)

f = F(cq)
f.compile()
f.init(b.T, b.R, b.U, b.I, b.Sum)

r = Float16x16x16Remapped(cq)
r.compile()
r.init(b.T, b.R, b.U, b.I, b.Sum)

(t0, t1, t2) = simulateKernel(r, I, R, 3, perBasicElement = False)

print str(t1*1000) + ' ~ ' + str(t2*1000) + ', ' + str(t0*1000)

print 'Version UnRemapped'
f.run()
f.run()
f.run()
print f.time

print 'Version ReMapped'
r.run()
r.run()
r.run()
print r.time
