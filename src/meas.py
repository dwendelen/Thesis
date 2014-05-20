import numpy as np
from OpenCLPlatform import OpenCLPlatform
from OpenCLPlatform2 import OpenCLPlatform2
from OpenCLPlatformE import OpenCLPlatformE
from simulators import float16x16x16SlechtsEenKeerFetchen,\
    float16x16x16ElkeKeerNieuweFetch
R = 16
I = 360

(_, _, t0, t1) = float16x16x16SlechtsEenKeerFetchen(I, R)
(_, _, _, t2) = float16x16x16ElkeKeerNieuweFetch(I, R)

print str(t1*1000) + ' ~ ' + str(t2*1000) + ', ' + str(t0*1000)

T = np.array(np.random.rand(I, I, I), dtype=np.float32)

U = np.array(np.random.rand(I, R), dtype=np.float32)
U2 = np.array(np.random.rand(I, R), dtype=np.float32)

print 'Version one'
p = OpenCLPlatform()
p.init()
p.setT(T)
p.setU([U, U, U])
p.setU2([U2, U2, U2])
p.f()
print p.time

print 'Empty'
p = OpenCLPlatformE()
p.init()
p.setT(T)
p.setU([U, U, U])
p.f()
print p.time

print 'Empty'
p = OpenCLPlatformE()
p.init()
p.setT(T)
p.setU([U, U, U])
p.f()
eTime = p.time
print eTime

T = np.array(np.random.rand(I, I, I), dtype=np.float32)

U = np.array(np.random.rand(I, R), dtype=np.float32)

print 'Version one'

p = OpenCLPlatform()
p.init()
p.setT(T)
p.setU([U, U, U])
p.setU2([U2, U2, U2])
p.f()
print p.time
print p.time-eTime

print 'Version two'

p = OpenCLPlatform2()
p.init()
p.setT(T)
p.setU([U, U, U])
p.f()
