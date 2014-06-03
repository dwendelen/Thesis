import numpy as np

from Kernel.Float16x16x16.F import F
from Kernel.Float16x16x16.E import Float16x16x16E as E
from Kernel.Float16x16x16.FRemapped import Float16x16x16Remapped

from simulators import simulateKernel
from Platform.ContextQueue import ContextQueue
from Kernel.Float16x16x16.BufferFactory import BufferFactory

def run(kernel, (t0, t1, t2), name):
    print name
    kernel.run()
    kernel.run()
    kernel.run()
    
    print kernel.time
    print str(t1*1000) + ' ~ ' + str(t2*1000) + ', ' + str(t0*1000)
    print str(t1*1000/kernel.time) + ' ~ ' + str(t2*1000/kernel.time) + ', ' + str(t0*1000/kernel.time)
    
def do(R, I):
    print 'R: ' + str(R) + ' I: ' + str(I)
    T = np.array(np.random.rand(I, I, I), dtype=np.float32)
    U = np.array(np.random.rand(I, R), dtype=np.float32)
    
    b.init(T, (U, U, U))
    
    e.init(b.T, b.R, b.U, b.I, b.Sum)
    f.init(b.T, b.R, b.U, b.I, b.Sum)
    r.init(b.T, b.R, b.U, b.I, b.Sum)
    
    (t0, t1, t2) = simulateKernel(r, I, R, 3, perBasicElement = False)

    run(e, (t0, t1, t2), 'Version Empty')
    run(f, (t0, t1, t2), 'Version UnRemapped')
    run(r, (t0, t1, t2), 'Version ReMapped')
    print ''
    print ''

cq = ContextQueue(profile = True)
cq.init()

b = BufferFactory(cq)

e = E(cq)
e.compile()

f = F(cq)
f.compile()

r = Float16x16x16Remapped(cq)
r.compile()

do(16,100)
do(4,100)
do(16,360)
do(4,360)

do(6000,16)
do(6000,100)
do(6000,360)

do(1,1)
do(1,4)
do(1,16)
do(1,64)
do(1,256)
do(1,400)
    
