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
from Kernel.Float4x4x4.BufferFactory import BufferFactory as BufferFactory4
from Kernel.Float8x8x8.BufferFactory import BufferFactory as BufferFactory8
from Kernel.BufferFactory import BufferFactory as BufferFactory1

#import cProfile
from Kernel.Sum.Sum16 import Sum16
from Kernel.Float4x4x4.FRemapped import FRemapped
from Kernel.Float8x8x8.FRemapped import FRemapped as FRemapped8
from Kernel.FloatSingle3D import FloatSingle3D


def run(kernel, (t0, t1, t2), name):
    print name
    kernel.run()
    kernel.run()
    kernel.run()
    
    print kernel.time
    #print str(t1*1000) + ' ~ ' + str(t2*1000) + ', ' + str(t0*1000)
    #print str(t1*1000/kernel.time) + ' ~ ' + str(t2*1000/kernel.time) + ', ' + str(t0*1000/kernel.time)
    
def do(R, I):
    print 'R: ' + str(R) + ' I: ' + str(I)
    T = np.array(np.random.rand(I, I, I), dtype=np.float32)
    U = np.array(np.random.rand(I, R), dtype=np.float32)
    
    npl.setT(T)
    npl.setU((U, U, U))
    
    b.init(T, (U, U, U))
    b4.init(T, (U, U, U))
    b8.init(T, (U, U, U))
    b1.init(T, (U, U, U))
    
    e.init(b.T, b.R, b.U, b.I, b.Sum)
    f.init(b.T, b.R, b.U, b.I, b.Sum)
    r.init(b.T, b.R, b.U, b.I, b.Sum)
    r2.init(b.T, b.R, b.U, b.I, b.Sum)
    r4.init(b4.T, b4.R, b4.U, b4.I, b4.Sum)
    r8.init(b8.T, b8.R, b8.U, b8.I, b8.Sum)
    rst.init(b1.T, b1.R, b1.U, b1.I, b1.Sum, b1.I0, b1.I1, b1.I2)    
    rm.init(b.T, b.TMapped, b.I)
    
    #(t0, t1, t2) = simulateKernel(r, I, R, 3, perBasicElement = False)

    (t0, t1, t2) = (0,0,0)

    run(e, (t0, t1, t2), 'Version Empty')
    #cProfile.run('npl.f()')
    run(f, (t0, t1, t2), 'Version UnRemapped')
    run(r, (t0, t1, t2), 'Version ReMapped')
    run(rst, (t0, t1, t2), 'Version Single T')
    run(r4, (t0, t1, t2), 'Version 4x4x4')
    run(r8, (t0, t1, t2), 'Version 8x8x8')
    run(rm, (t0, t1, t2), 'Version Remapper')
    print ''
    print ''
    
    b.release()
    b1.release()
    b4.release()
    b8.release()

npl = NumPyPlatform()
npl.init()

cq = ContextQueue(profile = True)
cq.init()

b = BufferFactory(cq)
b4 = BufferFactory4(cq)
b1 = BufferFactory1(cq)
b8 = BufferFactory8(cq)

e = E(cq)
e.compile()

f = F(cq)
f.compile()

r = Float16x16x16Remapped(cq)
r.compile()

r2 = Float16x16x16Remapped2(cq)
r2.compile()

rm = TMapper(cq)
rm.compile()

r4 = FRemapped(cq)
r4.compile()

r8 = FRemapped8(cq)
r8.compile()

rst = FloatSingle3D(cq)
rst.compile()

do(4,1)
do(6000,1)
do(4,10)
do(600,10)

do(16,100)
do(4,100)
do(16,360)
do(4,360)

do(6000,16)
do(6000,100)
do(6000,360)

do(1024,360)
do(100,360)
do(128,360)
do(1,360)

do(1,1)
do(1,4)
do(1,16)
do(1,64)
do(1,256)
do(1,400)
    
