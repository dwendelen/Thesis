import numpy as np
from OpenCLPlatform import OpenCLPlatform
from simulators import float16x16x16SlechtsEenKeerFetchen,\
    float16x16x16ElkeKeerNieuweFetch
R = 16
I = 400

(_, _, _, t1) = float16x16x16SlechtsEenKeerFetchen(I, R)
(_, _, _, t2) = float16x16x16ElkeKeerNieuweFetch(I, R)

print str(t1*1000) + ' ~ ' + str(t2*1000)

T = np.array(np.random.rand(I, I, I), dtype=np.float32)

U = np.array(np.random.rand(I, R), dtype=np.float32)

p = OpenCLPlatform()
p.init()
p.setT(T)
p.setU([U, U, U])
p.f()
