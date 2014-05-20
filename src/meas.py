import numpy as np
from OpenCLPlatform import OpenCLPlatform
from simulators import float16x16x16SlechtsEenKeerFetchen,\
    float16x16x16ElkeKeerNieuweFetch

(_, _, t1, _) = float16x16x16SlechtsEenKeerFetchen(160, 100)
(_, _, t2, _) = float16x16x16ElkeKeerNieuweFetch(160, 100)

print str(t1*1000) + ' ~ ' + str(t2*1000)

T = np.array(np.random.rand(160, 160, 160), dtype=np.float32)

U = np.array(np.random.rand(160, 100), dtype=np.float32)

p = OpenCLPlatform()
p.setT(T)
p.setU([U, U, U])
p.f()

(_, _, t1, _) = float16x16x16SlechtsEenKeerFetchen(160, 100)
(_, _, t2, _) = float16x16x16ElkeKeerNieuweFetch(160, 100)

print str(t1*1000) + ' ~ ' + str(t2*1000)